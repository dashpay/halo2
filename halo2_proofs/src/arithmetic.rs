//! This module provides common utilities, traits and structures for group,
//! field and polynomial arithmetic.

pub use ff::Field;
use group::{
    ff::{BatchInvert, PrimeField},
    Group as _, GroupOpsOwned, ScalarMulOwned,
};
use maybe_rayon::prelude::*;
pub use pasta_curves::arithmetic::*;

use crate::multicore;

/// This represents an element of a group with basic operations that can be
/// performed. This allows an FFT implementation (for example) to operate
/// generically over either a field or elliptic curve group.
pub trait FftGroup<Scalar: Field>:
    Copy + Send + Sync + 'static + GroupOpsOwned + ScalarMulOwned<Scalar>
{
}

impl<T, Scalar> FftGroup<Scalar> for T
where
    Scalar: Field,
    T: Copy + Send + Sync + 'static + GroupOpsOwned + ScalarMulOwned<Scalar>,
{
}

#[derive(Clone, Copy)]
enum Bucket<C: CurveAffine> {
    None,
    Affine(C),
    Projective(C::Curve),
}

impl<C: CurveAffine> Bucket<C> {
    fn add_assign(&mut self, other: C) {
        *self = match *self {
            Bucket::None => Bucket::Affine(other),
            Bucket::Affine(a) => Bucket::Projective(a + other),
            Bucket::Projective(mut a) => {
                a += other;
                Bucket::Projective(a)
            }
        }
    }

    fn add(self, mut other: C::Curve) -> C::Curve {
        match self {
            Bucket::None => other,
            Bucket::Affine(a) => {
                other += a;
                other
            }
            Bucket::Projective(a) => other + &a,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BoothDigit {
    magnitude: usize,
    negative: bool,
}

fn booth_digit(bytes: &[u8], window_bits: usize, window: usize) -> BoothDigit {
    debug_assert!(window_bits > 0);
    debug_assert!(window_bits < u64::BITS as usize);

    let window_start = window * window_bits;
    let byte_start = window_start / 8;
    let mut encoded = [0; 8];
    if byte_start < bytes.len() {
        for (slot, byte) in encoded.iter_mut().zip(&bytes[byte_start..]) {
            *slot = *byte;
        }
    }

    let bit_offset = window_start - byte_start * 8;
    let radix = 1usize << window_bits;
    let value = ((u64::from_le_bytes(encoded) >> bit_offset) as usize) & (radix - 1);
    let overlap = if window_start == 0 {
        0
    } else {
        let bit = window_start - 1;
        bytes
            .get(bit / 8)
            .map_or(0, |byte| usize::from((byte >> (bit % 8)) & 1))
    };

    // The bit below each window is its carry-in, while the window's high bit
    // is its carry-out. These terms cancel between adjacent windows, leaving
    // a signed digit whose magnitude is at most half the radix.
    if value < radix / 2 {
        BoothDigit {
            magnitude: value + overlap,
            negative: false,
        }
    } else {
        let magnitude = radix - value - overlap;
        BoothDigit {
            magnitude,
            negative: magnitude != 0,
        }
    }
}

fn booth_window_count<R: AsRef<[u8]>>(reprs: &[R], window_bits: usize) -> usize {
    debug_assert!(window_bits > 0);

    let repr_bytes = reprs
        .iter()
        .map(|repr| repr.as_ref().len())
        .max()
        .unwrap_or(0);
    let repr_bits = repr_bytes
        .checked_mul(u8::BITS as usize)
        .expect("scalar representation is too large");

    // Integer division reserves an additional carry window exactly when the
    // representation ends on a window boundary. Otherwise, the last partial
    // window cannot produce a carry.
    repr_bits / window_bits + 1
}

#[derive(Clone)]
struct BoothBuckets<C: CurveAffine> {
    c: usize,
    coeffs: Vec<Bucket<C>>,
}

impl<C: CurveAffine> BoothBuckets<C> {
    fn new(c: usize) -> Self {
        Self {
            c,
            coeffs: vec![Bucket::None; 1 << (c - 1)],
        }
    }

    fn sum(
        &mut self,
        coeffs: &[<C::Scalar as PrimeField>::Repr],
        bases: &[C],
        i: usize,
    ) -> C::Curve {
        // Recode each scalar window into a signed digit and add its base to
        // the bucket for the digit's magnitude.
        for (coeff, base) in coeffs.iter().zip(bases.iter()) {
            let digit = booth_digit(coeff.as_ref(), self.c, i);
            if digit.magnitude != 0 {
                let base = if digit.negative { -*base } else { *base };
                self.coeffs[digit.magnitude - 1].add_assign(base);
            }
        }
        // Summation by parts
        // e.g. 3a + 2b + 1c = a +
        //                    (a) + b +
        //                    ((a) + b) + c
        let mut acc = C::Curve::identity();
        let mut sum = C::Curve::identity();
        self.coeffs.iter().rev().for_each(|b| {
            sum = b.add(sum);
            acc += sum;
        });
        acc
    }
}

/// Performs a small multi-exponentiation operation.
/// Uses the double-and-add algorithm with doublings shared across points.
pub fn small_multiexp<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
    let coeffs: Vec<_> = coeffs.iter().map(|a| a.to_repr()).collect();
    let mut acc = C::Curve::identity();

    // for byte idx
    for byte_idx in (0..32).rev() {
        // for bit idx
        for bit_idx in (0..8).rev() {
            acc = acc.double();
            // for each coeff
            for coeff_idx in 0..coeffs.len() {
                let byte = coeffs[coeff_idx].as_ref()[byte_idx];
                if ((byte >> bit_idx) & 1) != 0 {
                    acc += bases[coeff_idx];
                }
            }
        }
    }

    acc
}

/// Performs a multi-exponentiation operation.
///
/// This function will panic if coeffs and bases have a different length.
///
/// This will use multithreading if beneficial.
pub fn best_multiexp<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
    assert_eq!(coeffs.len(), bases.len());

    // Convert to canonical representations once instead of once per window.
    let coeffs = coeffs.iter().map(PrimeField::to_repr).collect::<Vec<_>>();

    let c = if bases.len() < 4 {
        1
    } else if bases.len() < 32 {
        3
    } else {
        (f64::from(bases.len() as u32)).ln().ceil() as usize
    };

    let mut multi_buckets: Vec<BoothBuckets<C>> =
        vec![BoothBuckets::new(c); booth_window_count(&coeffs, c)];
    let num_threads = multicore::current_num_threads();
    let window_sums: Vec<C::Curve> = if coeffs.len() > num_threads {
        multi_buckets
            .par_iter_mut()
            .enumerate()
            .map(|(i, buckets)| buckets.sum(&coeffs, bases, i))
            .collect()
    } else {
        multi_buckets
            .iter_mut()
            .enumerate()
            .map(|(i, buckets)| buckets.sum(&coeffs, bases, i))
            .collect()
    };

    // Combine the window sums with Horner's rule, from the most significant
    // window down: acc = acc * 2^c + window_sum. This costs c doublings per
    // window in total, rather than c * i doublings for window i if each
    // window were shifted into place independently.
    let mut window_sums = window_sums.into_iter().rev();
    let mut acc = window_sums
        .next()
        .expect("multi_buckets always contains at least 1 window");
    for window_sum in window_sums {
        for _ in 0..c {
            acc = acc.double();
        }
        acc += window_sum;
    }
    acc
}

/// Performs a radix-$2$ Fast-Fourier Transformation (FFT) on a vector of size
/// $n = 2^k$, when provided `log_n` = $k$ and an element of multiplicative
/// order $n$ called `omega` ($\omega$). The result is that the vector `a`, when
/// interpreted as the coefficients of a polynomial of degree $n - 1$, is
/// transformed into the evaluations of this polynomial at each of the $n$
/// distinct powers of $\omega$. This transformation is invertible by providing
/// $\omega^{-1}$ in place of $\omega$ and dividing each resulting field element
/// by $n$.
///
/// This will use multithreading if beneficial.
pub fn best_fft<Scalar: Field, G: FftGroup<Scalar>>(a: &mut [G], omega: Scalar, log_n: u32) {
    fn bitreverse(mut n: usize, l: usize) -> usize {
        let mut r = 0;
        for _ in 0..l {
            r = (r << 1) | (n & 1);
            n >>= 1;
        }
        r
    }

    let threads = multicore::current_num_threads();
    let log_threads = log2_floor(threads);
    let n = a.len();
    assert_eq!(n, 1 << log_n);

    for k in 0..n {
        let rk = bitreverse(k, log_n as usize);
        if k < rk {
            a.swap(rk, k);
        }
    }

    // precompute twiddle factors
    let twiddles: Vec<_> = (0..(n / 2))
        .scan(Scalar::ONE, |w, _| {
            let tw = *w;
            *w *= &omega;
            Some(tw)
        })
        .collect();

    if log_n <= log_threads {
        let mut chunk = 2_usize;
        let mut twiddle_chunk = n / 2;
        for _ in 0..log_n {
            a.chunks_mut(chunk).for_each(|coeffs| {
                let (left, right) = coeffs.split_at_mut(chunk / 2);

                // case when twiddle factor is one
                let (a, left) = left.split_at_mut(1);
                let (b, right) = right.split_at_mut(1);
                let t = b[0];
                b[0] = a[0];
                a[0] += &t;
                b[0] -= &t;

                left.iter_mut()
                    .zip(right.iter_mut())
                    .enumerate()
                    .for_each(|(i, (a, b))| {
                        let mut t = *b;
                        t *= &twiddles[(i + 1) * twiddle_chunk];
                        *b = *a;
                        *a += &t;
                        *b -= &t;
                    });
            });
            chunk *= 2;
            twiddle_chunk /= 2;
        }
    } else {
        recursive_butterfly_arithmetic(a, n, 1, &twiddles)
    }
}

/// This perform recursive butterfly arithmetic
pub fn recursive_butterfly_arithmetic<Scalar: Field, G: FftGroup<Scalar>>(
    a: &mut [G],
    n: usize,
    twiddle_chunk: usize,
    twiddles: &[Scalar],
) {
    if n == 2 {
        let t = a[1];
        a[1] = a[0];
        a[0] += &t;
        a[1] -= &t;
    } else {
        let (left, right) = a.split_at_mut(n / 2);
        multicore::join(
            || recursive_butterfly_arithmetic(left, n / 2, twiddle_chunk * 2, twiddles),
            || recursive_butterfly_arithmetic(right, n / 2, twiddle_chunk * 2, twiddles),
        );

        // case when twiddle factor is one
        let (a, left) = left.split_at_mut(1);
        let (b, right) = right.split_at_mut(1);
        let t = b[0];
        b[0] = a[0];
        a[0] += &t;
        b[0] -= &t;

        left.iter_mut()
            .zip(right.iter_mut())
            .enumerate()
            .for_each(|(i, (a, b))| {
                let mut t = *b;
                t *= &twiddles[(i + 1) * twiddle_chunk];
                *b = *a;
                *a += &t;
                *b -= &t;
            });
    }
}

/// This evaluates a provided polynomial (in coefficient form) at `point`.
pub fn eval_polynomial<F: Field>(poly: &[F], point: F) -> F {
    // TODO: parallelize?
    poly.iter()
        .rev()
        .fold(F::ZERO, |acc, coeff| acc * point + coeff)
}

/// This computes the inner product of two vectors `a` and `b`.
///
/// This function will panic if the two vectors are not the same size.
pub fn compute_inner_product<F: Field>(a: &[F], b: &[F]) -> F {
    // TODO: parallelize?
    assert_eq!(a.len(), b.len());

    let mut acc = F::ZERO;
    for (a, b) in a.iter().zip(b.iter()) {
        acc += (*a) * (*b);
    }

    acc
}

/// Divides polynomial `a` in `X` by `X - b` with
/// no remainder.
pub fn kate_division<'a, F: Field, I: IntoIterator<Item = &'a F>>(a: I, mut b: F) -> Vec<F>
where
    I::IntoIter: DoubleEndedIterator + ExactSizeIterator,
{
    b = -b;
    let a = a.into_iter();

    let mut q = vec![F::ZERO; a.len() - 1];

    let mut tmp = F::ZERO;
    for (q, r) in q.iter_mut().rev().zip(a.rev()) {
        let mut lead_coeff = *r;
        lead_coeff.sub_assign(&tmp);
        *q = lead_coeff;
        tmp = lead_coeff;
        tmp.mul_assign(&b);
    }

    q
}

/// This simple utility function will parallelize an operation that is to be
/// performed over a mutable slice.
pub fn parallelize<T: Send, F: Fn(&mut [T], usize) + Send + Sync + Clone>(v: &mut [T], f: F) {
    let n = v.len();
    let num_threads = multicore::current_num_threads();
    let mut chunk = n / num_threads;
    if chunk < num_threads {
        chunk = n;
    }

    multicore::scope(|scope| {
        for (chunk_num, v) in v.chunks_mut(chunk).enumerate() {
            let f = f.clone();
            scope.spawn(move |_| {
                let start = chunk_num * chunk;
                f(v, start);
            });
        }
    });
}

fn log2_floor(num: usize) -> u32 {
    assert!(num > 0);

    let mut pow = 0;

    while (1 << (pow + 1)) <= num {
        pow += 1;
    }

    pow
}

/// Returns coefficients of an n - 1 degree polynomial given a set of n points
/// and their evaluations. This function will panic if two values in `points`
/// are the same.
pub fn lagrange_interpolate<F: Field>(points: &[F], evals: &[F]) -> Vec<F> {
    assert_eq!(points.len(), evals.len());
    if points.len() == 1 {
        // Constant polynomial
        vec![evals[0]]
    } else {
        let mut denoms = Vec::with_capacity(points.len());
        for (j, x_j) in points.iter().enumerate() {
            let mut denom = Vec::with_capacity(points.len() - 1);
            for x_k in points
                .iter()
                .enumerate()
                .filter(|&(k, _)| k != j)
                .map(|a| a.1)
            {
                denom.push(*x_j - x_k);
            }
            denoms.push(denom);
        }
        // Compute (x_j - x_k)^(-1) for each j != i
        denoms.iter_mut().flat_map(|v| v.iter_mut()).batch_invert();

        let mut final_poly = vec![F::ZERO; points.len()];
        for (j, (denoms, eval)) in denoms.into_iter().zip(evals.iter()).enumerate() {
            let mut tmp: Vec<F> = Vec::with_capacity(points.len());
            let mut product = Vec::with_capacity(points.len() - 1);
            tmp.push(F::ONE);
            for (x_k, denom) in points
                .iter()
                .enumerate()
                .filter(|&(k, _)| k != j)
                .map(|a| a.1)
                .zip(denoms.into_iter())
            {
                product.resize(tmp.len() + 1, F::ZERO);
                for ((a, b), product) in tmp
                    .iter()
                    .chain(std::iter::once(&F::ZERO))
                    .zip(std::iter::once(&F::ZERO).chain(tmp.iter()))
                    .zip(product.iter_mut())
                {
                    *product = *a * (-denom * x_k) + *b * denom;
                }
                std::mem::swap(&mut tmp, &mut product);
            }
            assert_eq!(tmp.len(), points.len());
            assert_eq!(product.len(), points.len() - 1);
            for (final_coeff, interpolation_coeff) in final_poly.iter_mut().zip(tmp.into_iter()) {
                *final_coeff += interpolation_coeff * eval;
            }
        }
        final_poly
    }
}

#[cfg(test)]
use rand_core::OsRng;

#[cfg(test)]
use crate::pasta::{Ep, EpAffine, Eq, EqAffine, Fp, Fq};

#[cfg(test)]
use group::{prime::PrimeCurveAffine, Curve};

#[cfg(test)]
fn assert_multiexp_matches_naive<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) {
    assert_eq!(best_multiexp(coeffs, bases), naive_multiexp(coeffs, bases));
}

/// The reference result: each term computed with the curve's own scalar
/// multiplication, then summed.
#[cfg(test)]
fn naive_multiexp<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
    coeffs
        .iter()
        .zip(bases)
        .map(|(coeff, base)| *base * coeff)
        .fold(C::Curve::identity(), |acc, val| acc + val)
}

#[cfg(test)]
fn evaluate_le_bytes(bytes: &[u8]) -> Fp {
    bytes.iter().rev().fold(Fp::ZERO, |mut acc, byte| {
        for bit in (0..u8::BITS).rev() {
            acc = acc.double();
            if (byte >> bit) & 1 != 0 {
                acc += Fp::ONE;
            }
        }
        acc
    })
}

#[cfg(test)]
fn evaluate_booth_digits(bytes: &[u8], window_bits: usize) -> Fp {
    (0..booth_window_count(&[bytes], window_bits))
        .rev()
        .fold(Fp::ZERO, |mut acc, window| {
            for _ in 0..window_bits {
                acc = acc.double();
            }

            let digit = booth_digit(bytes, window_bits, window);
            let magnitude = Fp::from(digit.magnitude as u64);
            if digit.negative {
                acc -= magnitude;
            } else {
                acc += magnitude;
            }
            acc
        })
}

#[test]
fn test_multiexp() {
    let rng = OsRng;
    for len in [0, 1, 2, 3, 4, 31, 32, 33, 255, 256, 257] {
        let coeffs = (0..len).map(|_| Fp::random(rng)).collect::<Vec<_>>();
        let bases = (0..len)
            .map(|_| EqAffine::from(Eq::random(rng)))
            .collect::<Vec<_>>();

        assert_multiexp_matches_naive(&coeffs, &bases);
    }
}

#[test]
fn test_booth_digit_boundaries() {
    let assert_digit = |bytes: &[u8], window_bits, window, magnitude, negative| {
        assert_eq!(
            booth_digit(bytes, window_bits, window),
            BoothDigit {
                magnitude,
                negative,
            }
        );
    };

    let repr_bytes = <Fp as PrimeField>::Repr::default().as_ref().len();
    let mut bytes = vec![0; repr_bytes];
    assert_digit(&bytes, 3, 0, 0, false);

    bytes[0] = 3;
    assert_digit(&bytes, 3, 0, 3, false);

    bytes[0] = 4;
    assert_digit(&bytes, 3, 0, 4, true);
    assert_digit(&bytes, 3, 1, 1, false);

    bytes[0] = 7;
    assert_digit(&bytes, 3, 0, 1, true);
    assert_digit(&bytes, 3, 1, 1, false);

    let bytes = vec![u8::MAX; repr_bytes];
    assert_digit(&bytes, 8, 31, 0, false);
    assert_digit(&bytes, 8, 32, 1, false);
}

#[test]
fn test_booth_digit_wide_representation() {
    const WIDE_REPR_BYTES: usize = 40;
    const FIRST_WIDE_BYTE: usize = 32;
    const WINDOW_BITS: usize = 8;
    const HALF_RADIX: usize = 1 << (WINDOW_BITS - 1);

    // A high-half value in the first byte above 256 bits produces a negative
    // digit whose carry must be consumed by the following window.
    let mut bytes = [0; WIDE_REPR_BYTES];
    let short_bytes = [0; FIRST_WIDE_BYTE];
    assert_eq!(
        booth_window_count(&[short_bytes.as_slice(), bytes.as_slice()], WINDOW_BITS),
        WIDE_REPR_BYTES + 1
    );

    bytes[FIRST_WIDE_BYTE] = HALF_RADIX as u8;
    assert_eq!(
        booth_digit(&bytes, WINDOW_BITS, FIRST_WIDE_BYTE),
        BoothDigit {
            magnitude: HALF_RADIX,
            negative: true,
        }
    );
    assert_eq!(
        booth_digit(&bytes, WINDOW_BITS, FIRST_WIDE_BYTE + 1),
        BoothDigit {
            magnitude: 1,
            negative: false,
        }
    );

    // The final data window likewise has a following carry-only window.
    bytes = [0; WIDE_REPR_BYTES];
    bytes[WIDE_REPR_BYTES - 1] = HALF_RADIX as u8;
    assert_eq!(
        booth_digit(&bytes, WINDOW_BITS, WIDE_REPR_BYTES - 1),
        BoothDigit {
            magnitude: HALF_RADIX,
            negative: true,
        }
    );
    assert_eq!(
        booth_digit(&bytes, WINDOW_BITS, WIDE_REPR_BYTES),
        BoothDigit {
            magnitude: 1,
            negative: false,
        }
    );

    let first_high_bit = {
        let mut bytes = [0; WIDE_REPR_BYTES];
        bytes[FIRST_WIDE_BYTE + 1] = 1;
        bytes
    };
    let all_ones = [u8::MAX; WIDE_REPR_BYTES];
    // Folding into `Fp` avoids a big-integer test dependency while exercising
    // the complete 320-bit recoding schedule.
    for bytes in [&bytes, &first_high_bit, &all_ones] {
        for window_bits in [1, 3, 5, 8, 9] {
            assert_eq!(
                evaluate_booth_digits(bytes, window_bits),
                evaluate_le_bytes(bytes)
            );
        }
    }
}

#[test]
fn test_multiexp_booth_boundaries() {
    let rng = OsRng;
    let mut fp_high_bit = Fp::ONE;
    for _ in 0..254 {
        fp_high_bit = fp_high_bit.double();
    }
    let fp_scalars = [
        Fp::ZERO,
        Fp::ONE,
        -Fp::ONE,
        Fp::from(3),
        Fp::from(4),
        Fp::from(7),
        Fp::from(8),
        fp_high_bit,
    ];
    let fp_coeffs = fp_scalars.into_iter().cycle().take(32).collect::<Vec<_>>();
    let fp_bases = (0..fp_coeffs.len())
        .map(|_| EqAffine::from(Eq::random(rng)))
        .collect::<Vec<_>>();
    assert_multiexp_matches_naive(&fp_coeffs, &fp_bases);

    let mut fq_high_bit = Fq::ONE;
    for _ in 0..254 {
        fq_high_bit = fq_high_bit.double();
    }
    let fq_scalars = [
        Fq::ZERO,
        Fq::ONE,
        -Fq::ONE,
        Fq::from(3),
        Fq::from(4),
        Fq::from(7),
        Fq::from(8),
        fq_high_bit,
    ];
    let fq_coeffs = fq_scalars.into_iter().cycle().take(32).collect::<Vec<_>>();
    let fq_bases = (0..fq_coeffs.len())
        .map(|_| EpAffine::from(Ep::random(rng)))
        .collect::<Vec<_>>();
    assert_multiexp_matches_naive(&fq_coeffs, &fq_bases);
}

/// Scalars that stress the signed-digit recoding: zero, small values, values
/// next to every window size's digit boundaries at several window offsets,
/// all-ones and alternating bit patterns, top-bit values, and values next to
/// the field modulus.
#[cfg(test)]
fn multiexp_edge_scalars<F: PrimeField>() -> Vec<F> {
    let pow2 = |e: usize| (0..e).fold(F::ONE, |acc, _| acc.double());
    let mut scalars = vec![F::ZERO, F::ONE, -F::ONE, F::from(2), -F::from(2)];
    for c in 1..=13 {
        for shift in [0, c, 3 * c, 64, 128 - c, 250 - c] {
            let base = pow2(shift);
            for v in [
                pow2(c) - F::ONE,
                pow2(c),
                pow2(c - 1) - F::ONE,
                pow2(c - 1),
                pow2(c - 1) + F::ONE,
            ] {
                scalars.push(v * base);
                scalars.push(-(v * base));
            }
        }
    }
    let bits = F::NUM_BITS as usize;
    let top = pow2(bits - 1);
    scalars.extend([top, top - F::ONE, top + F::ONE, -top, pow2(bits - 2)]);
    // Byte patterns over the low `bits - 1` bits, so they are always canonical.
    let pattern_bits = bits - 1;
    for pattern in [0xffu8, 0x80, 0x7f, 0xaa, 0x55, 0x01, 0xfe] {
        let mut repr = F::Repr::default();
        for (i, byte) in repr.as_mut().iter_mut().enumerate() {
            let bit_start = i * 8;
            if bit_start + 8 <= pattern_bits {
                *byte = pattern;
            } else if bit_start < pattern_bits {
                *byte = pattern & ((1u8 << (pattern_bits - bit_start)) - 1);
            }
        }
        let value = F::from_repr(repr).unwrap();
        scalars.extend([value, -value]);
    }
    scalars
}

/// Bases with identities, repeated points (bucket doublings) and negated
/// points (bucket cancellations) mixed into random points.
#[cfg(test)]
fn multiexp_test_bases<C: CurveAffine>(len: usize) -> Vec<C> {
    let rng = OsRng;
    let mut bases: Vec<C> = (0..len)
        .map(|_| C::Curve::random(rng).to_affine())
        .collect();
    for i in 0..len {
        match i % 11 {
            4 => bases[i] = C::identity(),
            7 => bases[i] = bases[i / 2],
            9 => bases[i] = -bases[i / 3],
            _ => {}
        }
    }
    bases
}

/// Checks `best_multiexp` against the naive sum for many lengths, under
/// several thread counts (which also moves the serial/parallel threshold), for
/// a mix of random and edge-case scalars.
#[cfg(test)]
fn check_multiexp_sizes_and_threads<C: CurveAffine>(lengths: &[usize], thread_counts: &[usize]) {
    let rng = OsRng;
    let edge = multiexp_edge_scalars::<C::Scalar>();
    let pools: Vec<_> = thread_counts
        .iter()
        .map(|&threads| {
            maybe_rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
        })
        .collect();

    for (round, &len) in lengths.iter().enumerate() {
        let bases = multiexp_test_bases::<C>(len);
        let coeff_sets: [Vec<C::Scalar>; 4] = [
            (0..len).map(|_| C::Scalar::random(rng)).collect(),
            // Edge scalars at rotating offsets interleaved with random ones.
            (0..len)
                .map(|i| {
                    if i % 3 == 2 {
                        C::Scalar::random(rng)
                    } else {
                        edge[(i + round * 7) % edge.len()]
                    }
                })
                .collect(),
            // All scalars identical (every point lands in the same buckets).
            vec![edge[(round * 5 + 3) % edge.len()]; len],
            vec![C::Scalar::ZERO; len],
        ];
        for coeffs in coeff_sets.iter() {
            let expected = naive_multiexp(coeffs, &bases);
            for pool in &pools {
                assert_eq!(
                    pool.install(|| best_multiexp(coeffs, &bases)),
                    expected,
                    "len {}",
                    len
                );
            }
        }
    }
}

#[test]
fn test_multiexp_sizes_and_threads() {
    let mut lengths: Vec<usize> = (0..=40).collect();
    lengths.extend([
        63, 64, 65, 127, 128, 129, 255, 256, 257, 1023, 1024, 1025, 2049,
    ]);
    let thread_counts = [1, 2, 3, 6, 14, 64];
    check_multiexp_sizes_and_threads::<EqAffine>(&lengths, &thread_counts);
    check_multiexp_sizes_and_threads::<EpAffine>(
        &[0, 1, 2, 3, 4, 13, 31, 32, 33, 100, 2048],
        &thread_counts,
    );
}

#[test]
fn test_multiexp_wide_windows() {
    // The window size is ceil(ln n): 8 up to n = 2980, 9 from n = 2981 and 10
    // from n = 8104. Cover the wider windows that large (e.g. batched)
    // verifier MSMs use.
    check_multiexp_sizes_and_threads::<EqAffine>(&[2980, 2981, 8104], &[1, 6]);
}

#[test]
fn test_multiexp_edge_scalars() {
    // Every edge scalar on its own, so that a wrong digit cannot be masked by
    // the other terms of the sum.
    let rng = OsRng;
    let base = EqAffine::from(Eq::random(rng));
    for scalar in multiexp_edge_scalars::<Fp>() {
        for len in [1, 2, 4, 32, 33, 64] {
            let coeffs = vec![scalar; len];
            let mut bases = vec![EqAffine::identity(); len];
            bases[len - 1] = base;
            assert_eq!(best_multiexp(&coeffs, &bases), base * scalar);
        }
    }
}

#[test]
fn test_lagrange_interpolate() {
    let rng = OsRng;

    let points = (0..5).map(|_| Fp::random(rng)).collect::<Vec<_>>();
    let evals = (0..5).map(|_| Fp::random(rng)).collect::<Vec<_>>();

    for coeffs in 0..5 {
        let points = &points[0..coeffs];
        let evals = &evals[0..coeffs];

        let poly = lagrange_interpolate(points, evals);
        assert_eq!(poly.len(), points.len());

        for (point, eval) in points.iter().zip(evals) {
            assert_eq!(eval_polynomial(&poly, *point), *eval);
        }
    }
}
