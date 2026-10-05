use std::{
    fmt,
    hash::{Hash, Hasher},
    marker::PhantomData,
    ops::{Add, Mul, MulAssign, Neg, Sub},
    sync::Arc,
};

use ff::WithSmallOrderMulGroup;
use group::ff::Field;

use super::{
    Basis, Coeff, EvaluationDomain, ExtendedLagrangeCoeff, LagrangeCoeff, Polynomial, Rotation,
};
use crate::multicore;

mod compiled;

/// Returns `(chunk_size, num_chunks)` suitable for processing the given polynomial length
/// in the current parallelization environment.
fn get_chunk_params(poly_len: usize) -> (usize, usize) {
    // Check the level of parallelization we have available.
    let num_threads = multicore::current_num_threads();
    // We scale the number of chunks by a constant factor, to ensure that if not all
    // threads are available, we can achieve more uniform throughput and don't end up
    // waiting on a couple of threads to process the last chunks.
    let num_chunks = num_threads * 4;
    // Calculate the ideal chunk size for the desired throughput. We use ceiling
    // division to ensure the minimum chunk size is 1.
    //     chunk_size = ceil(poly_len / num_chunks)
    let chunk_size = (poly_len + num_chunks - 1) / num_chunks;
    // Now re-calculate num_chunks from the actual chunk size.
    //     num_chunks = ceil(poly_len / chunk_size)
    let num_chunks = (poly_len + chunk_size - 1) / chunk_size;

    (chunk_size, num_chunks)
}

/// A reference to a polynomial registered with an [`Evaluator`].
#[derive(Clone, Copy)]
pub(crate) struct AstLeaf<E, B: Basis> {
    index: usize,
    rotation: Rotation,
    _evaluator: PhantomData<(E, B)>,
}

impl<E, B: Basis> fmt::Debug for AstLeaf<E, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AstLeaf")
            .field("index", &self.index)
            .field("rotation", &self.rotation)
            .finish()
    }
}

impl<E, B: Basis> PartialEq for AstLeaf<E, B> {
    fn eq(&self, rhs: &Self) -> bool {
        // We compare rotations by offset, which doesn't account for equivalent rotations.
        self.index.eq(&rhs.index) && self.rotation.0.eq(&rhs.rotation.0)
    }
}

impl<E, B: Basis> Eq for AstLeaf<E, B> {}

impl<E, B: Basis> Hash for AstLeaf<E, B> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.index.hash(state);
        self.rotation.0.hash(state);
    }
}

impl<E, B: Basis> AstLeaf<E, B> {
    /// Produces a new `AstLeaf` node corresponding to the underlying polynomial at a
    /// _new_ rotation. Existing rotations applied to this leaf node are ignored and the
    /// returned polynomial is not rotated _relative_ to the previous structure.
    pub(crate) fn with_rotation(&self, rotation: Rotation) -> Self {
        AstLeaf {
            index: self.index,
            rotation,
            _evaluator: PhantomData::default(),
        }
    }
}

/// An evaluation context for polynomial operations.
///
/// This context enables us to de-duplicate queries of circuit columns (and the rotations
/// they might require), by storing a list of all the underlying polynomials involved in
/// any query (which are almost certainly column polynomials). We use the context like so:
///
/// - We register each underlying polynomial with the evaluator, which returns a reference
///   to it as a [`AstLeaf`].
/// - The references are then used to build up a [`Ast`] that represents the overall
///   operations to be applied to the polynomials.
/// - Finally, we call [`Evaluator::evaluate`] passing in the [`Ast`].
pub(crate) struct Evaluator<E, F: Field, B: Basis> {
    polys: Vec<Polynomial<F, B>>,
    _context: E,
}

/// Constructs a new `Evaluator`.
///
/// The `context` parameter is used to provide type safety for evaluators. It ensures that
/// an evaluator will only be used to evaluate [`Ast`]s containing [`AstLeaf`]s obtained
/// from itself. It should be set to the empty closure `|| {}`, because anonymous closures
/// all have unique types.
pub(crate) fn new_evaluator<E: Fn() + Clone, F: Field, B: Basis>(context: E) -> Evaluator<E, F, B> {
    Evaluator {
        polys: vec![],
        _context: context,
    }
}

impl<E, F: Field, B: Basis> Evaluator<E, F, B> {
    /// Registers the given polynomial for use in this evaluation context.
    ///
    /// This API treats each registered polynomial as unique, even if the same polynomial
    /// is added multiple times.
    pub(crate) fn register_poly(&mut self, poly: Polynomial<F, B>) -> AstLeaf<E, B> {
        let index = self.polys.len();
        self.polys.push(poly);

        AstLeaf {
            index,
            rotation: Rotation::cur(),
            _evaluator: PhantomData::default(),
        }
    }

    /// Evaluates the given polynomial operation against this context.
    pub(crate) fn evaluate(
        &self,
        ast: &Ast<E, F, B>,
        domain: &EvaluationDomain<F>,
    ) -> Polynomial<F, B>
    where
        E: Copy + Send + Sync,
        F: WithSmallOrderMulGroup<3>,
        B: BasisOps,
    {
        // We're working in a single basis, so all polynomials are the same length.
        let poly_len = self.polys.first().unwrap().len();
        compiled::Program::compile(ast, poly_len, B::rotation_scale(domain))
            .evaluate(&self.polys, domain)
    }

    /// Evaluates the given polynomial operation against this context by recursively
    /// walking the [`Ast`]. This is the original evaluator, kept as a reference for tests.
    #[cfg(test)]
    pub(crate) fn evaluate_recursive(
        &self,
        ast: &Ast<E, F, B>,
        domain: &EvaluationDomain<F>,
    ) -> Polynomial<F, B>
    where
        E: Copy + Send + Sync,
        F: WithSmallOrderMulGroup<3>,
        B: BasisOps,
    {
        // We're working in a single basis, so all polynomials are the same length.
        let poly_len = self.polys.first().unwrap().len();
        let (chunk_size, _num_chunks) = get_chunk_params(poly_len);

        struct AstContext<'a, F: Field, B: Basis> {
            domain: &'a EvaluationDomain<F>,
            poly_len: usize,
            chunk_size: usize,
            chunk_index: usize,
            polys: &'a [Polynomial<F, B>],
        }

        fn recurse<E, F: WithSmallOrderMulGroup<3>, B: BasisOps>(
            ast: &Ast<E, F, B>,
            ctx: &AstContext<'_, F, B>,
        ) -> Vec<F> {
            match ast {
                Ast::Poly(leaf) => B::get_chunk_of_rotated(
                    ctx.domain,
                    ctx.chunk_size,
                    ctx.chunk_index,
                    &ctx.polys[leaf.index],
                    leaf.rotation,
                ),
                Ast::Add(a, b) => {
                    let mut lhs = recurse(a, ctx);
                    let rhs = recurse(b, ctx);
                    for (lhs, rhs) in lhs.iter_mut().zip(rhs.iter()) {
                        *lhs += *rhs;
                    }
                    lhs
                }
                Ast::Mul(AstMul(a, b)) => {
                    let mut lhs = recurse(a, ctx);
                    let rhs = recurse(b, ctx);
                    for (lhs, rhs) in lhs.iter_mut().zip(rhs.iter()) {
                        *lhs *= *rhs;
                    }
                    lhs
                }
                Ast::Scale(a, scalar) => {
                    let mut lhs = recurse(a, ctx);
                    for lhs in lhs.iter_mut() {
                        *lhs *= scalar;
                    }
                    lhs
                }
                Ast::DistributePowers(terms, base) => terms.iter().fold(
                    B::constant_term(ctx.poly_len, ctx.chunk_size, ctx.chunk_index, F::ZERO),
                    |mut acc, term| {
                        let term = recurse(term, ctx);
                        for (acc, term) in acc.iter_mut().zip(term) {
                            *acc *= base;
                            *acc += term;
                        }
                        acc
                    },
                ),
                Ast::LinearTerm(scalar) => B::linear_term(
                    ctx.domain,
                    ctx.poly_len,
                    ctx.chunk_size,
                    ctx.chunk_index,
                    *scalar,
                ),
                Ast::ConstantTerm(scalar) => {
                    B::constant_term(ctx.poly_len, ctx.chunk_size, ctx.chunk_index, *scalar)
                }
            }
        }

        // Apply `ast` to each chunk in parallel, writing the result into an output
        // polynomial.
        let mut result = B::empty_poly(domain);
        multicore::scope(|scope| {
            for (chunk_index, out) in result.chunks_mut(chunk_size).enumerate() {
                scope.spawn(move |_| {
                    let ctx = AstContext {
                        domain,
                        poly_len,
                        chunk_size,
                        chunk_index,
                        polys: &self.polys,
                    };
                    out.copy_from_slice(&recurse(ast, &ctx));
                });
            }
        });
        result
    }
}

/// Struct representing the [`Ast::Mul`] case.
///
/// This struct exists to make the internals of this case private so that we don't
/// accidentally construct this case directly, because it can only be implemented for the
/// [`ExtendedLagrangeCoeff`] basis.
#[derive(Clone)]
pub(crate) struct AstMul<E, F: Field, B: Basis>(Arc<Ast<E, F, B>>, Arc<Ast<E, F, B>>);

impl<E, F: Field, B: Basis> fmt::Debug for AstMul<E, F, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("AstMul")
            .field(&self.0)
            .field(&self.1)
            .finish()
    }
}

/// A polynomial operation backed by an [`Evaluator`].
#[derive(Clone)]
pub(crate) enum Ast<E, F: Field, B: Basis> {
    Poly(AstLeaf<E, B>),
    Add(Arc<Ast<E, F, B>>, Arc<Ast<E, F, B>>),
    Mul(AstMul<E, F, B>),
    Scale(Arc<Ast<E, F, B>>, F),
    /// Represents a linear combination of a vector of nodes and the powers of a
    /// field element, where the nodes are ordered from highest to lowest degree
    /// terms.
    DistributePowers(Arc<Vec<Ast<E, F, B>>>, F),
    /// The degree-1 term of a polynomial.
    ///
    /// The field element is the coefficient of the term in the standard basis, not the
    /// coefficient basis.
    LinearTerm(F),
    /// The degree-0 term of a polynomial.
    ///
    /// The field element is the same in both the standard and evaluation bases.
    ConstantTerm(F),
}

impl<E, F: Field, B: Basis> Ast<E, F, B> {
    pub fn distribute_powers<I: IntoIterator<Item = Self>>(i: I, base: F) -> Self {
        Ast::DistributePowers(Arc::new(i.into_iter().collect()), base)
    }
}

impl<E, F: Field, B: Basis> fmt::Debug for Ast<E, F, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Poly(leaf) => f.debug_tuple("Poly").field(leaf).finish(),
            Self::Add(lhs, rhs) => f.debug_tuple("Add").field(lhs).field(rhs).finish(),
            Self::Mul(x) => f.debug_tuple("Mul").field(x).finish(),
            Self::Scale(base, scalar) => f.debug_tuple("Scale").field(base).field(scalar).finish(),
            Self::DistributePowers(terms, base) => f
                .debug_tuple("DistributePowers")
                .field(terms)
                .field(base)
                .finish(),
            Self::LinearTerm(x) => f.debug_tuple("LinearTerm").field(x).finish(),
            Self::ConstantTerm(x) => f.debug_tuple("ConstantTerm").field(x).finish(),
        }
    }
}

impl<E, F: Field, B: Basis> From<AstLeaf<E, B>> for Ast<E, F, B> {
    fn from(leaf: AstLeaf<E, B>) -> Self {
        Ast::Poly(leaf)
    }
}

impl<E, F: Field, B: Basis> Ast<E, F, B> {
    pub(crate) fn one() -> Self {
        Self::ConstantTerm(F::ONE)
    }
}

impl<E, F: Field, B: Basis> Neg for Ast<E, F, B> {
    type Output = Ast<E, F, B>;

    fn neg(self) -> Self::Output {
        Ast::Scale(Arc::new(self), -F::ONE)
    }
}

impl<E: Clone, F: Field, B: Basis> Neg for &Ast<E, F, B> {
    type Output = Ast<E, F, B>;

    fn neg(self) -> Self::Output {
        -(self.clone())
    }
}

impl<E, F: Field, B: Basis> Add for Ast<E, F, B> {
    type Output = Ast<E, F, B>;

    fn add(self, other: Self) -> Self::Output {
        Ast::Add(Arc::new(self), Arc::new(other))
    }
}

impl<'a, E: Clone, F: Field, B: Basis> Add<&'a Ast<E, F, B>> for &'a Ast<E, F, B> {
    type Output = Ast<E, F, B>;

    fn add(self, other: &'a Ast<E, F, B>) -> Self::Output {
        self.clone() + other.clone()
    }
}

impl<E, F: Field, B: Basis> Add<AstLeaf<E, B>> for Ast<E, F, B> {
    type Output = Ast<E, F, B>;

    fn add(self, other: AstLeaf<E, B>) -> Self::Output {
        Ast::Add(Arc::new(self), Arc::new(other.into()))
    }
}

impl<E, F: Field, B: Basis> Sub for Ast<E, F, B> {
    type Output = Ast<E, F, B>;

    fn sub(self, other: Self) -> Self::Output {
        self + (-other)
    }
}

impl<'a, E: Clone, F: Field, B: Basis> Sub<&'a Ast<E, F, B>> for &'a Ast<E, F, B> {
    type Output = Ast<E, F, B>;

    fn sub(self, other: &'a Ast<E, F, B>) -> Self::Output {
        self + &(-other)
    }
}

impl<E, F: Field, B: Basis> Sub<AstLeaf<E, B>> for Ast<E, F, B> {
    type Output = Ast<E, F, B>;

    fn sub(self, other: AstLeaf<E, B>) -> Self::Output {
        self + (-Ast::from(other))
    }
}

impl<E, F: Field> Mul for Ast<E, F, LagrangeCoeff> {
    type Output = Ast<E, F, LagrangeCoeff>;

    fn mul(self, other: Self) -> Self::Output {
        Ast::Mul(AstMul(Arc::new(self), Arc::new(other)))
    }
}

impl<'a, E: Clone, F: Field> Mul<&'a Ast<E, F, LagrangeCoeff>> for &'a Ast<E, F, LagrangeCoeff> {
    type Output = Ast<E, F, LagrangeCoeff>;

    fn mul(self, other: &'a Ast<E, F, LagrangeCoeff>) -> Self::Output {
        self.clone() * other.clone()
    }
}

impl<E, F: Field> Mul<AstLeaf<E, LagrangeCoeff>> for Ast<E, F, LagrangeCoeff> {
    type Output = Ast<E, F, LagrangeCoeff>;

    fn mul(self, other: AstLeaf<E, LagrangeCoeff>) -> Self::Output {
        Ast::Mul(AstMul(Arc::new(self), Arc::new(other.into())))
    }
}

impl<E, F: Field> Mul for Ast<E, F, ExtendedLagrangeCoeff> {
    type Output = Ast<E, F, ExtendedLagrangeCoeff>;

    fn mul(self, other: Self) -> Self::Output {
        Ast::Mul(AstMul(Arc::new(self), Arc::new(other)))
    }
}

impl<'a, E: Clone, F: Field> Mul<&'a Ast<E, F, ExtendedLagrangeCoeff>>
    for &'a Ast<E, F, ExtendedLagrangeCoeff>
{
    type Output = Ast<E, F, ExtendedLagrangeCoeff>;

    fn mul(self, other: &'a Ast<E, F, ExtendedLagrangeCoeff>) -> Self::Output {
        self.clone() * other.clone()
    }
}

impl<E, F: Field> Mul<AstLeaf<E, ExtendedLagrangeCoeff>> for Ast<E, F, ExtendedLagrangeCoeff> {
    type Output = Ast<E, F, ExtendedLagrangeCoeff>;

    fn mul(self, other: AstLeaf<E, ExtendedLagrangeCoeff>) -> Self::Output {
        Ast::Mul(AstMul(Arc::new(self), Arc::new(other.into())))
    }
}

impl<E, F: Field, B: Basis> Mul<F> for Ast<E, F, B> {
    type Output = Ast<E, F, B>;

    fn mul(self, other: F) -> Self::Output {
        Ast::Scale(Arc::new(self), other)
    }
}

impl<E: Clone, F: Field, B: Basis> Mul<F> for &Ast<E, F, B> {
    type Output = Ast<E, F, B>;

    fn mul(self, other: F) -> Self::Output {
        Ast::Scale(Arc::new(self.clone()), other)
    }
}

impl<E: Clone, F: Field> MulAssign for Ast<E, F, ExtendedLagrangeCoeff> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone().mul(rhs)
    }
}

/// Operations which can be performed over a given basis.
pub(crate) trait BasisOps: Basis {
    /// Whether a constant polynomial has the same value at every index in this basis.
    const CONSTANT_IS_UNIFORM: bool;
    /// The index shift corresponding to a rotation by one row, or `None` if polynomials in
    /// this basis cannot be rotated.
    fn rotation_scale<F: WithSmallOrderMulGroup<3>>(domain: &EvaluationDomain<F>) -> Option<usize>;
    /// Returns `(x_0, omega)` such that the degree-1 polynomial `X` has the value
    /// `x_0 * omega^i` at index `i`, or `None` if it is not a geometric sequence in this
    /// basis.
    fn x_geometric<F: WithSmallOrderMulGroup<3>>(domain: &EvaluationDomain<F>) -> Option<(F, F)>;
    /// The value of the degree-1 polynomial `X` at index `i`, for bases where
    /// [`BasisOps::x_geometric`] returns `None`.
    fn x_at_row<F: Field>(_i: usize) -> F {
        unreachable!("X is a geometric sequence in this basis")
    }
    fn empty_poly<F: WithSmallOrderMulGroup<3>>(
        domain: &EvaluationDomain<F>,
    ) -> Polynomial<F, Self>;
    #[cfg(test)]
    fn constant_term<F: Field>(
        poly_len: usize,
        chunk_size: usize,
        chunk_index: usize,
        scalar: F,
    ) -> Vec<F>;
    #[cfg(test)]
    fn linear_term<F: WithSmallOrderMulGroup<3>>(
        domain: &EvaluationDomain<F>,
        poly_len: usize,
        chunk_size: usize,
        chunk_index: usize,
        scalar: F,
    ) -> Vec<F>;
    #[cfg(test)]
    fn get_chunk_of_rotated<F: WithSmallOrderMulGroup<3>>(
        domain: &EvaluationDomain<F>,
        chunk_size: usize,
        chunk_index: usize,
        poly: &Polynomial<F, Self>,
        rotation: Rotation,
    ) -> Vec<F>;
}

impl BasisOps for Coeff {
    const CONSTANT_IS_UNIFORM: bool = false;

    fn rotation_scale<F: WithSmallOrderMulGroup<3>>(_: &EvaluationDomain<F>) -> Option<usize> {
        None
    }

    fn x_geometric<F: WithSmallOrderMulGroup<3>>(_: &EvaluationDomain<F>) -> Option<(F, F)> {
        None
    }

    fn x_at_row<F: Field>(i: usize) -> F {
        if i == 1 {
            F::ONE
        } else {
            F::ZERO
        }
    }

    fn empty_poly<F: WithSmallOrderMulGroup<3>>(
        domain: &EvaluationDomain<F>,
    ) -> Polynomial<F, Self> {
        domain.empty_coeff()
    }

    #[cfg(test)]
    fn constant_term<F: Field>(
        poly_len: usize,
        chunk_size: usize,
        chunk_index: usize,
        scalar: F,
    ) -> Vec<F> {
        let mut chunk =
            vec![F::ZERO; std::cmp::min(chunk_size, poly_len - chunk_size * chunk_index)];
        if chunk_index == 0 {
            chunk[0] = scalar;
        }
        chunk
    }

    #[cfg(test)]
    fn linear_term<F: WithSmallOrderMulGroup<3>>(
        _: &EvaluationDomain<F>,
        poly_len: usize,
        chunk_size: usize,
        chunk_index: usize,
        scalar: F,
    ) -> Vec<F> {
        let mut chunk =
            vec![F::ZERO; std::cmp::min(chunk_size, poly_len - chunk_size * chunk_index)];
        // If the chunk size is 1 (e.g. if we have a small k and many threads), then the
        // linear coefficient is the second chunk. Otherwise, the chunk size is greater
        // than one, and the linear coefficient is the second element of the first chunk.
        // Note that we check against the original chunk size, not the potentially-short
        // actual size of the current chunk, because we want to know whether the size of
        // the previous chunk was 1.
        if chunk_size == 1 {
            if chunk_index == 1 {
                chunk[0] = scalar;
            }
        } else if chunk_index == 0 {
            chunk[1] = scalar;
        }
        chunk
    }

    #[cfg(test)]
    fn get_chunk_of_rotated<F: WithSmallOrderMulGroup<3>>(
        _: &EvaluationDomain<F>,
        _: usize,
        _: usize,
        _: &Polynomial<F, Self>,
        _: Rotation,
    ) -> Vec<F> {
        panic!("Can't rotate polynomials in the standard basis")
    }
}

impl BasisOps for LagrangeCoeff {
    const CONSTANT_IS_UNIFORM: bool = true;

    fn rotation_scale<F: WithSmallOrderMulGroup<3>>(_: &EvaluationDomain<F>) -> Option<usize> {
        Some(1)
    }

    fn x_geometric<F: WithSmallOrderMulGroup<3>>(domain: &EvaluationDomain<F>) -> Option<(F, F)> {
        Some((F::ONE, domain.get_omega()))
    }

    fn empty_poly<F: WithSmallOrderMulGroup<3>>(
        domain: &EvaluationDomain<F>,
    ) -> Polynomial<F, Self> {
        domain.empty_lagrange()
    }

    #[cfg(test)]
    fn constant_term<F: Field>(
        poly_len: usize,
        chunk_size: usize,
        chunk_index: usize,
        scalar: F,
    ) -> Vec<F> {
        vec![scalar; std::cmp::min(chunk_size, poly_len - chunk_size * chunk_index)]
    }

    #[cfg(test)]
    fn linear_term<F: WithSmallOrderMulGroup<3>>(
        domain: &EvaluationDomain<F>,
        poly_len: usize,
        chunk_size: usize,
        chunk_index: usize,
        scalar: F,
    ) -> Vec<F> {
        // Take every power of omega within the chunk, and multiply by scalar.
        let omega = domain.get_omega();
        let start = chunk_size * chunk_index;
        (0..std::cmp::min(chunk_size, poly_len - start))
            .scan(omega.pow_vartime([start as u64]) * scalar, |acc, _| {
                let ret = *acc;
                *acc *= omega;
                Some(ret)
            })
            .collect()
    }

    #[cfg(test)]
    fn get_chunk_of_rotated<F: WithSmallOrderMulGroup<3>>(
        _: &EvaluationDomain<F>,
        chunk_size: usize,
        chunk_index: usize,
        poly: &Polynomial<F, Self>,
        rotation: Rotation,
    ) -> Vec<F> {
        poly.get_chunk_of_rotated(rotation, chunk_size, chunk_index)
    }
}

impl BasisOps for ExtendedLagrangeCoeff {
    const CONSTANT_IS_UNIFORM: bool = true;

    fn rotation_scale<F: WithSmallOrderMulGroup<3>>(domain: &EvaluationDomain<F>) -> Option<usize> {
        Some(domain.extended_rotation_scale())
    }

    fn x_geometric<F: WithSmallOrderMulGroup<3>>(domain: &EvaluationDomain<F>) -> Option<(F, F)> {
        Some((F::ZETA, domain.get_extended_omega()))
    }

    fn empty_poly<F: WithSmallOrderMulGroup<3>>(
        domain: &EvaluationDomain<F>,
    ) -> Polynomial<F, Self> {
        domain.empty_extended()
    }

    #[cfg(test)]
    fn constant_term<F: Field>(
        poly_len: usize,
        chunk_size: usize,
        chunk_index: usize,
        scalar: F,
    ) -> Vec<F> {
        vec![scalar; std::cmp::min(chunk_size, poly_len - chunk_size * chunk_index)]
    }

    #[cfg(test)]
    fn linear_term<F: WithSmallOrderMulGroup<3>>(
        domain: &EvaluationDomain<F>,
        poly_len: usize,
        chunk_size: usize,
        chunk_index: usize,
        scalar: F,
    ) -> Vec<F> {
        // Take every power of the extended omega within the chunk, and multiply by scalar.
        let omega = domain.get_extended_omega();
        let start = chunk_size * chunk_index;
        (0..std::cmp::min(chunk_size, poly_len - start))
            .scan(
                omega.pow_vartime([start as u64]) * F::ZETA * scalar,
                |acc, _| {
                    let ret = *acc;
                    *acc *= omega;
                    Some(ret)
                },
            )
            .collect()
    }

    #[cfg(test)]
    fn get_chunk_of_rotated<F: WithSmallOrderMulGroup<3>>(
        domain: &EvaluationDomain<F>,
        chunk_size: usize,
        chunk_index: usize,
        poly: &Polynomial<F, Self>,
        rotation: Rotation,
    ) -> Vec<F> {
        domain.get_chunk_of_rotated_extended(poly, rotation, chunk_size, chunk_index)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use group::ff::Field;
    use pasta_curves::pallas;

    use super::{
        compiled::Program, get_chunk_params, new_evaluator, Ast, AstLeaf, AstMul, BasisOps,
        Evaluator,
    };
    use crate::poly::{
        Coeff, EvaluationDomain, ExtendedLagrangeCoeff, LagrangeCoeff, Polynomial, Rotation,
    };

    #[test]
    fn short_chunk_regression_test() {
        // Pick the smallest polynomial length that is guaranteed to produce a short chunk
        // on this machine.
        let k = match (1..16)
            .map(|k| (k, get_chunk_params(1 << k)))
            .find(|(k, (chunk_size, num_chunks))| (1 << k) < chunk_size * num_chunks)
            .map(|(k, _)| k)
        {
            Some(k) => k,
            None => {
                // We are on a machine with a power-of-two number of threads, and cannot
                // trigger the bug.
                eprintln!(
                    "can't find a polynomial length for short_chunk_regression_test; skipping"
                );
                return;
            }
        };
        eprintln!("Testing short-chunk regression with k = {}", k);

        fn test_case<E: Copy + Send + Sync, B: BasisOps>(
            k: u32,
            mut evaluator: Evaluator<E, pallas::Base, B>,
        ) {
            // Instantiate the evaluator with a trivial polynomial.
            let domain = EvaluationDomain::new(1, k);
            evaluator.register_poly(B::empty_poly(&domain));

            // With the bug present, these will panic.
            let _ = evaluator.evaluate(&Ast::ConstantTerm(pallas::Base::ZERO), &domain);
            let _ = evaluator.evaluate(&Ast::LinearTerm(pallas::Base::ZERO), &domain);
        }

        test_case(k, new_evaluator::<_, _, Coeff>(|| {}));
        test_case(k, new_evaluator::<_, _, LagrangeCoeff>(|| {}));
        test_case(k, new_evaluator::<_, _, ExtendedLagrangeCoeff>(|| {}));
    }

    /// A small deterministic xorshift generator, so failures are reproducible.
    struct TestRng(u64);

    impl TestRng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }

        /// Returns a scalar, often one of the values the compiler special-cases.
        fn scalar(&mut self) -> pallas::Base {
            match self.below(6) {
                0 => pallas::Base::ZERO,
                1 => pallas::Base::ONE,
                2 => -pallas::Base::ONE,
                _ => pallas::Base::from(self.next()) * pallas::Base::from(self.next()),
            }
        }
    }

    /// Generates a random [`Ast`] over `leaves`. Subtrees are frequently shared, both by
    /// cloning (which shares the `Arc`ed children) and by repeating structurally equal
    /// queries.
    fn random_ast<E: Copy, B: BasisOps>(
        rng: &mut TestRng,
        leaves: &[AstLeaf<E, B>],
        max_rotation: i32,
        mul: bool,
        depth: usize,
        pool: &mut Vec<Ast<E, pallas::Base, B>>,
    ) -> Ast<E, pallas::Base, B> {
        if !pool.is_empty() && rng.below(5) == 0 {
            return pool[rng.below(pool.len())].clone();
        }
        let ast = if depth == 0 {
            match rng.below(4) {
                0 => Ast::ConstantTerm(rng.scalar()),
                1 => Ast::LinearTerm(rng.scalar()),
                _ => {
                    let leaf = leaves[rng.below(leaves.len())];
                    let rotation = rng.below(2 * max_rotation as usize + 1) as i32 - max_rotation;
                    Ast::Poly(leaf.with_rotation(Rotation(rotation)))
                }
            }
        } else {
            let mut sub =
                |rng: &mut TestRng| random_ast(rng, leaves, max_rotation, mul, depth - 1, pool);
            match rng.below(if mul { 6 } else { 5 }) {
                0 => Ast::Add(Arc::new(sub(rng)), Arc::new(sub(rng))),
                1 => -sub(rng),
                2 => sub(rng) - sub(rng),
                3 => Ast::Scale(Arc::new(sub(rng)), rng.scalar()),
                4 => {
                    let terms = (0..rng.below(4)).map(|_| sub(rng)).collect::<Vec<_>>();
                    Ast::distribute_powers(terms, rng.scalar())
                }
                _ => Ast::Mul(AstMul(Arc::new(sub(rng)), Arc::new(sub(rng)))),
            }
        };
        pool.push(ast.clone());
        ast
    }

    /// Evaluates `ast` at a single index, directly from the definitions of the operations.
    fn naive_eval<E, B: BasisOps>(
        ast: &Ast<E, pallas::Base, B>,
        polys: &[Polynomial<pallas::Base, B>],
        domain: &EvaluationDomain<pallas::Base>,
        row: usize,
    ) -> pallas::Base {
        let len = polys[0].len();
        match ast {
            Ast::Poly(leaf) => {
                let shift = leaf.rotation.0 as i64 * B::rotation_scale(domain).unwrap_or(0) as i64;
                polys[leaf.index][(row as i64 + shift).rem_euclid(len as i64) as usize]
            }
            Ast::Add(a, b) => naive_eval(a, polys, domain, row) + naive_eval(b, polys, domain, row),
            Ast::Mul(AstMul(a, b)) => {
                naive_eval(a, polys, domain, row) * naive_eval(b, polys, domain, row)
            }
            Ast::Scale(a, c) => naive_eval(a, polys, domain, row) * c,
            Ast::DistributePowers(terms, base) => {
                terms.iter().fold(pallas::Base::ZERO, |acc, t| {
                    acc * base + naive_eval(t, polys, domain, row)
                })
            }
            // Use the recursive evaluator's whole-polynomial terms as the definitions.
            Ast::LinearTerm(c) => B::linear_term(domain, len, len, 0, *c)[row],
            Ast::ConstantTerm(c) => B::constant_term(len, len, 0, *c)[row],
        }
    }

    /// Checks that the compiled evaluator agrees with a direct evaluation (and, where it
    /// supports the expression, with the recursive evaluator) on random expressions, for
    /// every chunk size on small domains.
    ///
    /// `lagrange` selects whether `B` is one of the Lagrange bases, which support rotations
    /// and multiplication.
    fn check_random_expressions<B: BasisOps>(lagrange: bool) {
        let mut rng = TestRng(0x5eed_1234_5678_9abc);
        for (j, k) in [(1, 1), (3, 1), (3, 2), (5, 2), (4, 3), (9, 4)] {
            let domain = EvaluationDomain::<pallas::Base>::new(j, k);
            let mut evaluator = new_evaluator::<_, _, B>(|| {});
            let leaves: Vec<_> = (0..3)
                .map(|_| {
                    let mut poly = B::empty_poly(&domain);
                    for value in poly.iter_mut() {
                        *value = pallas::Base::from(rng.next());
                    }
                    evaluator.register_poly(poly)
                })
                .collect();
            let poly_len = evaluator.polys[0].len();

            for _ in 0..40 {
                let mut pool = vec![];
                let depth = 1 + rng.below(4);
                // The evaluator does not support rotating by more than the domain size.
                let max_rotation = if lagrange { 3.min((1 << k) - 1) } else { 0 };
                let ast = random_ast(&mut rng, &leaves, max_rotation, lagrange, depth, &mut pool);

                let expected: Vec<_> = (0..poly_len)
                    .map(|row| naive_eval(&ast, &evaluator.polys, &domain, row))
                    .collect();
                assert_eq!(evaluator.evaluate(&ast, &domain).values, expected);
                // The recursive evaluator cannot read polynomials in the coefficient basis.
                if lagrange {
                    assert_eq!(evaluator.evaluate_recursive(&ast, &domain).values, expected);
                }

                let program = Program::compile(&ast, poly_len, B::rotation_scale(&domain));
                for chunk_size in 1..=poly_len {
                    let actual: Polynomial<_, B> =
                        program.evaluate_in_chunks(&evaluator.polys, &domain, chunk_size);
                    assert_eq!(actual.values, expected, "chunk size {}", chunk_size);
                }
            }
        }
    }

    #[test]
    fn compiled_matches_reference_coeff() {
        check_random_expressions::<Coeff>(false);
    }

    #[test]
    fn compiled_matches_reference_lagrange() {
        check_random_expressions::<LagrangeCoeff>(true);
    }

    #[test]
    fn compiled_matches_reference_extended() {
        check_random_expressions::<ExtendedLagrangeCoeff>(true);
    }

    #[test]
    fn compiler_deduplicates_and_simplifies() {
        let domain = EvaluationDomain::<pallas::Base>::new(3, 3);
        let mut evaluator = new_evaluator::<_, _, ExtendedLagrangeCoeff>(|| {});
        let a = evaluator.register_poly(domain.constant_extended(pallas::Base::from(3)));
        let b = evaluator.register_poly(domain.constant_extended(pallas::Base::from(5)));
        let poly_len = domain.extended_len();
        let compile = |ast: &Ast<_, _, _>| {
            Program::compile(
                ast,
                poly_len,
                ExtendedLagrangeCoeff::rotation_scale(&domain),
            )
            .num_instrs()
        };

        // Separately constructed copies of the same product are evaluated once.
        let ab = || Ast::from(a) * Ast::from(b);
        assert_eq!(compile(&(ab() + ab())), 2);
        assert_eq!(compile(&(ab() * ab())), 2);

        // Scaling by zero, one and minus one, adding zero, and the leading zero step of a
        // Horner chain need no instructions of their own.
        let x = Ast::from(a);
        assert_eq!(compile(&(x.clone() * pallas::Base::ONE)), 0);
        assert_eq!(compile(&(x.clone() * pallas::Base::ZERO)), 0);
        assert_eq!(
            compile(&(Ast::ConstantTerm(pallas::Base::ZERO) + x.clone())),
            0
        );
        assert_eq!(
            compile(&(x.clone() + (-(Ast::from(b) * -pallas::Base::ONE)))),
            1
        );
        assert_eq!(
            compile(&Ast::distribute_powers(
                [x.clone(), Ast::from(b)],
                pallas::Base::from(7)
            )),
            1
        );
        // A product with a constant polynomial is a scale.
        assert_eq!(compile(&(x * Ast::ConstantTerm(pallas::Base::from(2)))), 1);
    }

    #[test]
    fn coeff_linear_term_short_chunks() {
        // With one coefficient per chunk, the linear coefficient is the whole second
        // chunk. This used to index past the end of the first chunk.
        let domain = EvaluationDomain::<pallas::Base>::new(1, 1);
        let s = pallas::Base::from(9);
        assert_eq!(
            Coeff::linear_term(&domain, 2, 1, 0, s),
            vec![pallas::Base::ZERO]
        );
        assert_eq!(Coeff::linear_term(&domain, 2, 1, 1, s), vec![s]);
        assert_eq!(
            Coeff::linear_term(&domain, 2, 2, 0, s),
            vec![pallas::Base::ZERO, s]
        );
    }
}
