//! Compiled evaluation of [`Ast`]s.
//!
//! [`Ast`]s are trees whose leaves are (rotated) registered polynomials. Evaluating one
//! recursively re-reads, copies and re-combines every leaf and shared subtree each time
//! it appears, and allocates a temporary vector for every node of every chunk.
//!
//! Instead, we lower the tree into a small register program once per evaluation:
//!
//! - The tree is converted into a DAG in which structurally identical nodes (including
//!   repeated column/rotation queries) are created once ("hash-consing").
//! - While building the DAG we apply exact algebraic identities that only depend on the
//!   shape of the expression and on public constants: scaling by zero, one or minus one,
//!   adding zero, folding constants, and the initial zero step of a Horner chain. Field
//!   arithmetic is exact, so these rewrites do not change the result.
//! - Each remaining operation becomes an instruction. Values that are used again later are
//!   kept in scratch registers, and a register is reused once its value has been read for
//!   the last time.
//! - The program is then executed row by row, [`LANES`] independent rows per instruction,
//!   reading the leaves directly out of the registered polynomials.

use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
};

use ff::{PrimeField, WithSmallOrderMulGroup};

use super::{get_chunk_params, Ast, AstMul, BasisOps};
use crate::{
    multicore,
    poly::{EvaluationDomain, Polynomial},
};

/// The number of rows evaluated by each instruction.
pub(super) const LANES: usize = 4;

type NodeId = usize;

/// A node in the expression DAG.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Node<F> {
    /// A registered polynomial, read at `index + offset` (modulo the polynomial length).
    Poly {
        poly: usize,
        offset: usize,
    },
    /// The constant polynomial with the given value.
    Constant(F),
    /// The degree-1 polynomial `X`.
    X,
    Neg(NodeId),
    Add(NodeId, NodeId),
    Sub(NodeId, NodeId),
    Mul(NodeId, NodeId),
    Scale(NodeId, F),
    /// `acc * base + term`.
    Horner(NodeId, F, NodeId),
}

/// A [`Node`] used as a hash map key. Field elements don't implement [`Hash`], so they are
/// hashed through their canonical encoding.
#[derive(Clone, Copy, Debug)]
struct NodeKey<F>(Node<F>);

impl<F: PartialEq> PartialEq for NodeKey<F> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<F: Eq> Eq for NodeKey<F> {}

impl<F: PrimeField> Hash for NodeKey<F> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(&self.0).hash(state);
        let hash_field = |f: &F, state: &mut H| f.to_repr().as_ref().hash(state);
        match &self.0 {
            Node::Poly { poly, offset } => (poly, offset).hash(state),
            Node::Constant(c) => hash_field(c, state),
            Node::X => (),
            Node::Neg(a) => a.hash(state),
            Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) => (a, b).hash(state),
            Node::Scale(a, c) => {
                a.hash(state);
                hash_field(c, state);
            }
            Node::Horner(a, c, b) => {
                (a, b).hash(state);
                hash_field(c, state);
            }
        }
    }
}

impl<F: Copy> Node<F> {
    /// The nodes this node reads.
    fn operands(&self) -> [Option<NodeId>; 2] {
        match *self {
            Node::Poly { .. } | Node::Constant(_) | Node::X => [None, None],
            Node::Neg(a) | Node::Scale(a, _) => [Some(a), None],
            Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) | Node::Horner(a, _, b) => {
                [Some(a), Some(b)]
            }
        }
    }
}

/// Builds a de-duplicated, simplified expression DAG. Nodes are always created after their
/// children, so node indices are a topological order.
struct Dag<F: PrimeField> {
    nodes: Vec<Node<F>>,
    ids: HashMap<NodeKey<F>, NodeId>,
    /// The polynomial length, used to normalize rotations into offsets.
    poly_len: usize,
    /// The number of rows of the polynomial per unit of rotation, or `None` if the
    /// polynomials cannot be rotated.
    rotation_scale: Option<usize>,
}

impl<F: PrimeField> Dag<F> {
    fn intern(&mut self, node: Node<F>) -> NodeId {
        let nodes = &mut self.nodes;
        *self.ids.entry(NodeKey(node)).or_insert_with(|| {
            nodes.push(node);
            nodes.len() - 1
        })
    }

    fn constant_value(&self, id: NodeId) -> Option<F> {
        match self.nodes[id] {
            Node::Constant(c) => Some(c),
            _ => None,
        }
    }

    fn is_zero(&self, id: NodeId) -> bool {
        self.constant_value(id)
            .map_or(false, |c| c.is_zero_vartime())
    }

    fn poly(&mut self, poly: usize, rotation: i32) -> NodeId {
        let offset = if rotation == 0 {
            0
        } else {
            let scale = self
                .rotation_scale
                .expect("Can't rotate polynomials in the standard basis");
            (rotation as i64 * scale as i64).rem_euclid(self.poly_len as i64) as usize
        };
        self.intern(Node::Poly { poly, offset })
    }

    fn constant(&mut self, c: F) -> NodeId {
        self.intern(Node::Constant(c))
    }

    fn neg(&mut self, a: NodeId) -> NodeId {
        match self.nodes[a] {
            Node::Constant(c) => self.constant(-c),
            Node::Neg(x) => x,
            Node::Scale(x, c) => self.scale(x, -c),
            _ => self.intern(Node::Neg(a)),
        }
    }

    fn add(&mut self, a: NodeId, b: NodeId) -> NodeId {
        if let (Some(x), Some(y)) = (self.constant_value(a), self.constant_value(b)) {
            return self.constant(x + y);
        }
        if self.is_zero(a) {
            return b;
        }
        if self.is_zero(b) {
            return a;
        }
        if let Node::Neg(y) = self.nodes[b] {
            return self.sub(a, y);
        }
        if let Node::Neg(x) = self.nodes[a] {
            return self.sub(b, x);
        }
        // Addition is commutative; order the operands so that `a + b` and `b + a` share
        // a node.
        self.intern(Node::Add(a.min(b), a.max(b)))
    }

    /// `a - b`, which is only produced from `a + (-b)` in [`Dag::add`]. There, `b` is never
    /// a constant (the negation of a constant is folded into a constant) and `a` is never
    /// the zero constant; `a` may be a non-zero constant, e.g. in `1 - z`.
    fn sub(&mut self, a: NodeId, b: NodeId) -> NodeId {
        self.intern(Node::Sub(a, b))
    }

    /// Pointwise multiplication. A constant operand is turned into a scale, which is only
    /// valid in bases where a constant polynomial has the same value at every index. That
    /// holds because [`Ast::Mul`] can only be constructed in the Lagrange bases.
    fn mul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        if let Some(c) = self.constant_value(a) {
            return self.scale(b, c);
        }
        if let Some(c) = self.constant_value(b) {
            return self.scale(a, c);
        }
        self.intern(Node::Mul(a.min(b), a.max(b)))
    }

    fn scale(&mut self, a: NodeId, c: F) -> NodeId {
        if c.is_zero_vartime() {
            return self.constant(F::ZERO);
        }
        if c == F::ONE {
            return a;
        }
        if c == -F::ONE {
            return self.neg(a);
        }
        match self.nodes[a] {
            Node::Constant(x) => self.constant(x * c),
            Node::Scale(x, d) => self.scale(x, d * c),
            Node::Neg(x) => self.scale(x, -c),
            _ => self.intern(Node::Scale(a, c)),
        }
    }

    fn horner(&mut self, acc: NodeId, base: F, term: NodeId) -> NodeId {
        if base.is_zero_vartime() || self.is_zero(acc) {
            return term;
        }
        if let (Some(a), Some(t)) = (self.constant_value(acc), self.constant_value(term)) {
            return self.constant(a * base + t);
        }
        if self.is_zero(term) {
            return self.scale(acc, base);
        }
        if base == F::ONE {
            return self.add(acc, term);
        }
        self.intern(Node::Horner(acc, base, term))
    }

    /// Lowers an [`Ast`] into this DAG, returning the id of its root. `memo` caches the
    /// lowering of shared subtrees by address so that DAG-shaped [`Ast`]s are traversed
    /// once.
    fn lower<E, B: BasisOps>(
        &mut self,
        ast: &Ast<E, F, B>,
        memo: &mut HashMap<*const Ast<E, F, B>, NodeId>,
    ) -> NodeId {
        let ptr = ast as *const _;
        if let Some(id) = memo.get(&ptr) {
            return *id;
        }
        let id = match ast {
            Ast::Poly(leaf) => self.poly(leaf.index, leaf.rotation.0),
            Ast::Add(a, b) => {
                let a = self.lower(a, memo);
                let b = self.lower(b, memo);
                self.add(a, b)
            }
            Ast::Mul(AstMul(a, b)) => {
                debug_assert!(
                    B::CONSTANT_IS_UNIFORM,
                    "multiplication is only defined in the Lagrange bases"
                );
                let a = self.lower(a, memo);
                let b = self.lower(b, memo);
                self.mul(a, b)
            }
            Ast::Scale(a, scalar) => {
                let a = self.lower(a, memo);
                self.scale(a, *scalar)
            }
            Ast::DistributePowers(terms, base) => {
                let mut acc = self.constant(F::ZERO);
                for term in terms.iter() {
                    let term = self.lower(term, memo);
                    acc = self.horner(acc, *base, term);
                }
                acc
            }
            Ast::LinearTerm(scalar) => {
                let x = self.intern(Node::X);
                self.scale(x, *scalar)
            }
            Ast::ConstantTerm(scalar) => self.constant(*scalar),
        };
        memo.insert(ptr, id);
        id
    }
}

/// A value read by an instruction. Indices are `u32` and constants live in
/// [`Program::constants`], keeping instructions small: the whole instruction stream is read
/// once per block of rows, so it should stay in the L1 cache.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Src {
    /// A scratch register holding the value of an earlier instruction.
    Reg(u32),
    /// A registered polynomial, read at `row + offset`.
    Poly { poly: u32, offset: u32 },
    /// A constant polynomial, by index into [`Program::constants`].
    Constant(u32),
    /// The degree-1 polynomial `X`.
    X,
}

/// An operation. Scalars are indices into [`Program::constants`].
#[derive(Clone, Copy, Debug)]
enum Op {
    Neg(Src),
    Add(Src, Src),
    Sub(Src, Src),
    Mul(Src, Src),
    Scale(Src, u32),
    /// `acc * base + term`.
    Horner(Src, u32, Src),
}

/// An operation whose result is written to the scratch register `dst`.
#[derive(Clone, Copy, Debug)]
struct Instr {
    dst: u32,
    op: Op,
}

#[inline(always)]
fn lanes_map<F>(mut a: [F; LANES], f: impl Fn(&mut F)) -> [F; LANES] {
    a.iter_mut().for_each(f);
    a
}

#[inline(always)]
fn lanes_zip<F>(mut a: [F; LANES], b: [F; LANES], f: impl Fn(&mut F, &F)) -> [F; LANES] {
    for (a, b) in a.iter_mut().zip(b.iter()) {
        f(a, b);
    }
    a
}

/// A lowered [`Ast`], ready to be executed over the rows of the registered polynomials.
#[derive(Debug)]
pub(super) struct Program<F> {
    instrs: Vec<Instr>,
    /// The scalars and constant polynomial values the instructions refer to.
    constants: Vec<F>,
    /// Where the result is read from after all instructions have run.
    result: Src,
    /// The number of scratch registers needed.
    num_regs: usize,
    /// Whether the program reads [`Src::X`].
    uses_x: bool,
    /// The length of the polynomials the program reads and produces.
    poly_len: usize,
}

impl<F: WithSmallOrderMulGroup<3>> Program<F> {
    /// Compiles `ast` for polynomials of length `poly_len`, where one unit of rotation is
    /// `rotation_scale` rows (`None` if the polynomials cannot be rotated).
    pub(super) fn compile<E, B: BasisOps>(
        ast: &Ast<E, F, B>,
        poly_len: usize,
        rotation_scale: Option<usize>,
    ) -> Self {
        let mut dag = Dag {
            nodes: vec![],
            ids: HashMap::new(),
            poly_len,
            rotation_scale,
        };
        let root = dag.lower(ast, &mut HashMap::new());
        let nodes = &dag.nodes;

        // Find the nodes the result depends on, and the last node (in topological order)
        // that reads each of them.
        let mut live = vec![false; nodes.len()];
        live[root] = true;
        let mut last_use = vec![0; nodes.len()];
        for id in (0..nodes.len()).rev() {
            if !live[id] {
                continue;
            }
            for child in nodes[id].operands().into_iter().flatten() {
                if !live[child] {
                    live[child] = true;
                    last_use[child] = id;
                }
            }
        }

        // Emit instructions in topological order, assigning scratch registers.
        let mut constants = vec![];
        let index = |i: usize| u32::try_from(i).expect("program fits in u32 indices");
        let mut constant = |c: F| {
            constants.push(c);
            index(constants.len() - 1)
        };
        let mut src_of: Vec<Option<Src>> = vec![None; nodes.len()];
        let mut free_regs: Vec<u32> = vec![];
        let mut num_regs = 0;
        let mut instrs = vec![];
        for id in (0..nodes.len()).filter(|id| live[*id]) {
            let leaf = match nodes[id] {
                Node::Poly { poly, offset } => Some(Src::Poly {
                    poly: index(poly),
                    offset: index(offset),
                }),
                Node::Constant(c) => Some(Src::Constant(constant(c))),
                Node::X => Some(Src::X),
                _ => None,
            };
            if leaf.is_some() {
                src_of[id] = leaf;
                continue;
            }

            // Operands are read before the destination is written, so the registers of
            // operands that are not used again can be reused for the destination. Both
            // operands may be the same node (e.g. `a * a`); only free its register once.
            for op in nodes[id].operands().into_iter().flatten() {
                if let Some(Src::Reg(r)) = src_of[op] {
                    if last_use[op] == id && !free_regs.contains(&r) {
                        free_regs.push(r);
                    }
                }
            }
            let dst = free_regs.pop().unwrap_or_else(|| {
                num_regs += 1;
                index(num_regs - 1)
            });

            let src = |n: NodeId| src_of[n].expect("operands precede their users");
            let op = match nodes[id] {
                Node::Neg(a) => Op::Neg(src(a)),
                Node::Add(a, b) => Op::Add(src(a), src(b)),
                Node::Sub(a, b) => Op::Sub(src(a), src(b)),
                Node::Mul(a, b) => Op::Mul(src(a), src(b)),
                Node::Scale(a, c) => Op::Scale(src(a), constant(c)),
                Node::Horner(a, c, b) => Op::Horner(src(a), constant(c), src(b)),
                Node::Poly { .. } | Node::Constant(_) | Node::X => unreachable!(),
            };
            instrs.push(Instr { dst, op });
            src_of[id] = Some(Src::Reg(dst));
        }

        let result = src_of[root].expect("root is live");
        Program {
            instrs,
            constants,
            result,
            num_regs,
            uses_x: src_of.contains(&Some(Src::X)),
            poly_len,
        }
    }

    /// The number of instructions in the program.
    #[cfg(test)]
    pub(super) fn num_instrs(&self) -> usize {
        self.instrs.len()
    }

    /// Evaluates the program over every row of `polys`.
    pub(super) fn evaluate<B: BasisOps>(
        &self,
        polys: &[Polynomial<F, B>],
        domain: &EvaluationDomain<F>,
    ) -> Polynomial<F, B> {
        // Make every chunk except the last a whole number of row blocks.
        let (chunk_size, _) = get_chunk_params(self.poly_len);
        let chunk_size = (chunk_size + LANES - 1) / LANES * LANES;
        self.evaluate_in_chunks(polys, domain, chunk_size)
    }

    /// Evaluates the program over every row of `polys`, processing `chunk_size` rows per
    /// task.
    pub(super) fn evaluate_in_chunks<B: BasisOps>(
        &self,
        polys: &[Polynomial<F, B>],
        domain: &EvaluationDomain<F>,
        chunk_size: usize,
    ) -> Polynomial<F, B> {
        let mut result = B::empty_poly(domain);
        assert_eq!(result.len(), self.poly_len);
        assert!(polys.iter().all(|p| p.len() == self.poly_len));

        multicore::scope(|scope| {
            for (chunk_index, out) in result.chunks_mut(chunk_size).enumerate() {
                scope.spawn(move |_| {
                    self.evaluate_chunk(polys, domain, chunk_index * chunk_size, out)
                });
            }
        });
        result
    }

    fn evaluate_chunk<B: BasisOps>(
        &self,
        polys: &[Polynomial<F, B>],
        domain: &EvaluationDomain<F>,
        start: usize,
        out: &mut [F],
    ) {
        let poly_len = self.poly_len;
        let mut regs = vec![[F::ZERO; LANES]; self.num_regs];

        // `X` at the next row and the step between rows, when `X` is a geometric sequence
        // in this basis.
        let mut x_geometric = if self.uses_x {
            B::x_geometric(domain)
                .map(|(x0, omega)| (x0 * omega.pow_vartime([start as u64]), omega))
        } else {
            None
        };

        for (block, out) in out.chunks_mut(LANES).enumerate() {
            let row0 = start + block * LANES;

            // The rows evaluated by each lane. Lanes past the end of the polynomial (only
            // in a short final block) wrap around; their results are discarded.
            let mut rows = [0; LANES];
            for (lane, row) in rows.iter_mut().enumerate() {
                *row = row0 + lane;
                if row0 + LANES > poly_len {
                    *row %= poly_len;
                }
            }

            let mut xs = [F::ZERO; LANES];
            if self.uses_x {
                match &mut x_geometric {
                    Some((x, omega)) => {
                        for value in xs.iter_mut() {
                            *value = *x;
                            *x *= *omega;
                        }
                    }
                    None => {
                        for (value, row) in xs.iter_mut().zip(rows.iter()) {
                            *value = B::x_at_row(*row);
                        }
                    }
                }
            }

            let load = |src: Src, regs: &[[F; LANES]]| -> [F; LANES] {
                match src {
                    Src::Reg(r) => regs[r as usize],
                    Src::Poly { poly, offset } => {
                        let (values, offset) = (&polys[poly as usize][..], offset as usize);
                        let mut lanes = [F::ZERO; LANES];
                        for (value, row) in lanes.iter_mut().zip(rows.iter()) {
                            let mut index = row + offset;
                            if index >= poly_len {
                                index -= poly_len;
                            }
                            *value = values[index];
                        }
                        lanes
                    }
                    Src::Constant(c) => {
                        let mut lanes = [self.constants[c as usize]; LANES];
                        if !B::CONSTANT_IS_UNIFORM {
                            for (value, row) in lanes.iter_mut().zip(rows.iter()) {
                                if *row != 0 {
                                    *value = F::ZERO;
                                }
                            }
                        }
                        lanes
                    }
                    Src::X => xs,
                }
            };

            for instr in self.instrs.iter() {
                regs[instr.dst as usize] = match instr.op {
                    Op::Neg(a) => lanes_map(load(a, &regs), |a| *a = -*a),
                    Op::Scale(a, c) => {
                        let c = self.constants[c as usize];
                        lanes_map(load(a, &regs), |a| *a *= c)
                    }
                    Op::Add(a, b) => lanes_zip(load(a, &regs), load(b, &regs), |a, b| *a += b),
                    Op::Sub(a, b) => lanes_zip(load(a, &regs), load(b, &regs), |a, b| *a -= b),
                    Op::Mul(a, b) => lanes_zip(load(a, &regs), load(b, &regs), |a, b| *a *= b),
                    Op::Horner(acc, base, term) => {
                        let base = self.constants[base as usize];
                        lanes_zip(load(acc, &regs), load(term, &regs), |acc, term| {
                            *acc *= base;
                            *acc += term;
                        })
                    }
                };
            }

            let result = load(self.result, &regs);
            out.copy_from_slice(&result[..out.len()]);
        }
    }
}
