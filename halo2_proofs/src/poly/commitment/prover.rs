use ff::Field;
use rand_core::RngCore;

use super::super::{Coeff, Polynomial};
use super::{Blind, Params};
use crate::arithmetic::{
    best_multiexp, compute_inner_product, eval_polynomial, parallelize, CurveAffine,
};
use crate::multicore;
use crate::pasta::{
    glv::{Decomposed, GlvParams, Table},
    pallas, vesta,
};
use crate::transcript::{EncodedChallenge, TranscriptWrite};

use group::{prime::PrimeCurveAffine, Curve};
use std::any::Any;
use std::io;

/// Create a polynomial commitment opening proof for the polynomial defined
/// by the coefficients `px`, the blinding factor `blind` used for the
/// polynomial commitment, and the point `x` that the polynomial is
/// evaluated at.
///
/// This function will panic if the provided polynomial is too large with
/// respect to the polynomial commitment parameters.
///
/// **Important:** This function assumes that the provided `transcript` has
/// already seen the common inputs: the polynomial commitment P, the claimed
/// opening v, and the point x. It's probably also nice for the transcript
/// to have seen the elliptic curve description and the URS, if you want to
/// be rigorous.
pub fn create_proof<
    C: CurveAffine,
    E: EncodedChallenge<C>,
    R: RngCore,
    T: TranscriptWrite<C, E>,
>(
    params: &Params<C>,
    mut rng: R,
    transcript: &mut T,
    p_poly: &Polynomial<C::Scalar, Coeff>,
    p_blind: Blind<C::Scalar>,
    x_3: C::Scalar,
) -> io::Result<()> {
    // We're limited to polynomials of degree n - 1.
    assert_eq!(p_poly.len(), params.n as usize);

    // Sample a random polynomial (of same degree) that has a root at x_3, first
    // by setting all coefficients to random values.
    let mut s_poly = (*p_poly).clone();
    for coeff in s_poly.iter_mut() {
        *coeff = C::Scalar::random(&mut rng);
    }
    // Evaluate the random polynomial at x_3
    let s_at_x3 = eval_polynomial(&s_poly[..], x_3);
    // Subtract constant coefficient to get a random polynomial with a root at x_3
    s_poly[0] -= &s_at_x3;
    // And sample a random blind
    let s_poly_blind = Blind(C::Scalar::random(&mut rng));

    // Write a commitment to the random polynomial to the transcript
    let s_poly_commitment = params.commit(&s_poly, s_poly_blind).to_affine();
    transcript.write_point(s_poly_commitment)?;

    // Challenge that will ensure that the prover cannot change P but can only
    // witness a random polynomial commitment that agrees with P at x_3, with high
    // probability.
    let xi = *transcript.squeeze_challenge_scalar::<()>();

    // Challenge that ensures that the prover did not interfere with the U term
    // in their commitments.
    let z = *transcript.squeeze_challenge_scalar::<()>();

    // We'll be opening `P' = P - [v] G_0 + [ξ] S` to ensure it has a root at
    // zero.
    let mut p_prime_poly = s_poly * xi + p_poly;
    let v = eval_polynomial(&p_prime_poly, x_3);
    p_prime_poly[0] -= &v;
    let p_prime_blind = s_poly_blind * Blind(xi) + p_blind;

    // This accumulates the synthetic blinding factor `f` starting
    // with the blinding factor for `P'`.
    let mut f = p_prime_blind.0;

    // Initialize the vector `p_prime` as the coefficients of the polynomial.
    let mut p_prime = p_prime_poly.values;
    assert_eq!(p_prime.len(), params.n as usize);

    // Initialize the vector `b` as the powers of `x_3`. The inner product of
    // `p_prime` and `b` is the evaluation of the polynomial at `x_3`.
    let mut b = Vec::with_capacity(1 << params.k);
    {
        let mut cur = C::Scalar::ONE;
        for _ in 0..(1 << params.k) {
            b.push(cur);
            cur *= &x_3;
        }
    }

    // Initialize the vector `G'` from the URS. We'll be progressively collapsing
    // this vector into smaller and smaller vectors until it is of length 1.
    let mut g_prime = params.g.clone();

    // Perform the inner product argument, round by round.
    for j in 0..params.k {
        let half = 1 << (params.k - j - 1); // half the length of `p_prime`, `b`, `G'`

        // Compute L, R
        //
        // TODO: If we modify multiexp to take "extra" bases, we could speed
        // this piece up a bit by combining the multiexps.
        let l_j = best_multiexp(&p_prime[half..], &g_prime[0..half]);
        let r_j = best_multiexp(&p_prime[0..half], &g_prime[half..]);
        let value_l_j = compute_inner_product(&p_prime[half..], &b[0..half]);
        let value_r_j = compute_inner_product(&p_prime[0..half], &b[half..]);
        let l_j_randomness = C::Scalar::random(&mut rng);
        let r_j_randomness = C::Scalar::random(&mut rng);
        let l_j = l_j + &best_multiexp(&[value_l_j * &z, l_j_randomness], &[params.u, params.w]);
        let r_j = r_j + &best_multiexp(&[value_r_j * &z, r_j_randomness], &[params.u, params.w]);
        let l_j = l_j.to_affine();
        let r_j = r_j.to_affine();

        // Feed L and R into the real transcript
        transcript.write_point(l_j)?;
        transcript.write_point(r_j)?;

        let u_j = *transcript.squeeze_challenge_scalar::<()>();
        let u_j_inv = u_j.invert().unwrap(); // TODO, bubble this up

        // Collapse `p_prime` and `b`.
        // TODO: parallelize
        #[allow(clippy::assign_op_pattern)]
        for i in 0..half {
            p_prime[i] = p_prime[i] + &(p_prime[i + half] * &u_j_inv);
            b[i] = b[i] + &(b[i + half] * &u_j);
        }
        p_prime.truncate(half);
        b.truncate(half);

        // Collapse `G'`
        parallel_generator_collapse(&mut g_prime, u_j);
        g_prime.truncate(half);

        // Update randomness (the synthetic blinding factor at the end)
        f += &(l_j_randomness * &u_j_inv);
        f += &(r_j_randomness * &u_j);
    }

    // We have fully collapsed `p_prime`, `b`, `G'`
    assert_eq!(p_prime.len(), 1);
    let c = p_prime[0];

    transcript.write_scalar(c)?;
    transcript.write_scalar(f)?;

    Ok(())
}

/// Collapses the generator vector `g` of even length `2m` in place, so that its
/// first `m` entries become `g[i] + challenge * g[m + i]`. The caller truncates
/// `g` to `m` entries afterwards.
///
/// `challenge` is a public Fiat-Shamir challenge and `g` is derived only from
/// the public parameters and earlier public challenges, so variable-time scalar
/// multiplication is acceptable here: nothing secret flows into this function.
fn parallel_generator_collapse<C: CurveAffine>(g: &mut Vec<C>, challenge: C::Scalar) {
    // Pallas and Vesta (the only curves `halo2_proofs` is used with in
    // practice) take the GLV path; any other curve keeps the generic one.
    if !glv_generator_collapse::<C, vesta::Point>(g, &challenge)
        && !glv_generator_collapse::<C, pallas::Point>(g, &challenge)
    {
        generic_generator_collapse(g, challenge);
    }
}

/// The original, curve-agnostic collapse, using the curve implementation's
/// (double-and-add) scalar multiplication.
fn generic_generator_collapse<C: CurveAffine>(g: &mut [C], challenge: C::Scalar) {
    let len = g.len() / 2;
    let (g_lo, g_hi) = g.split_at_mut(len);

    parallelize(g_lo, |g_lo, start| {
        let g_hi = &g_hi[start..];
        let mut tmp = Vec::with_capacity(g_lo.len());
        for (g_lo, g_hi) in g_lo.iter().zip(g_hi.iter()) {
            tmp.push(g_lo.to_curve() + &(*g_hi * challenge));
        }
        C::Curve::batch_normalize(&tmp, g_lo);
    });
}

/// Minimum number of points handled by one task of [`glv_generator_collapse`].
/// A GLV multiplication costs tens of microseconds, so even small chunks
/// amortize the task overhead and the one field inversion per chunk, and the
/// later IPA rounds (which have few points) still run in parallel.
const GLV_COLLAPSE_MIN_CHUNK: usize = 4;

/// Generator collapse using `pasta_curves`' variable-time GLV multiplication,
/// if `C` is the affine form of `P`. Returns `false`, leaving `g` untouched,
/// if it is not.
///
/// The challenge is decomposed once per round and shared by every point; each
/// point's table of odd multiples of `P` and `phi(P)` is built in batches that
/// share a single field inversion. The results are exact group elements, so
/// the collapsed (normalized) vector is identical to the generic path's.
// `g` is a `Vec` rather than a slice because only sized types can be
// downcast through `Any`.
#[allow(clippy::ptr_arg)]
fn glv_generator_collapse<C: CurveAffine, P: GlvParams>(
    g: &mut Vec<C>,
    challenge: &C::Scalar,
) -> bool {
    let (g, challenge) = match (
        (g as &mut dyn Any).downcast_mut::<Vec<P::AffineExt>>(),
        (challenge as &dyn Any).downcast_ref::<P::ScalarExt>(),
    ) {
        (Some(g), Some(challenge)) => (g, challenge),
        _ => return false,
    };

    let half = g.len() / 2;
    let (g_lo, g_hi) = g.split_at_mut(half);
    let decomposed = Decomposed::<P>::new(challenge);
    let decomposed = &decomposed;

    let num_threads = multicore::current_num_threads();
    let chunk = ((half + num_threads - 1) / num_threads).max(GLV_COLLAPSE_MIN_CHUNK);
    multicore::scope(|scope| {
        for (g_lo, g_hi) in g_lo.chunks_mut(chunk).zip(g_hi.chunks(chunk)) {
            scope.spawn(move |_| {
                let g_hi: Vec<P> = g_hi.iter().map(PrimeCurveAffine::to_curve).collect();
                let collapsed: Vec<P> = Table::batch(&g_hi)
                    .iter()
                    .zip(g_lo.iter())
                    .map(|(table, g_lo)| table.mul_decomposed(decomposed) + g_lo)
                    .collect();
                P::batch_normalize(&collapsed, g_lo);
            });
        }
    });
    true
}

#[cfg(test)]
mod tests {
    use super::{generic_generator_collapse, glv_generator_collapse, parallel_generator_collapse};
    use crate::arithmetic::CurveAffine;
    use crate::pasta::{pallas, vesta, EpAffine, EqAffine, Fp};
    use ff::Field;
    use group::{Curve, Group};
    use rand_core::OsRng;

    /// `g_lo[i] + challenge * g_hi[i]`, computed point by point.
    fn naive_collapse<C: CurveAffine>(g: &[C], challenge: C::Scalar) -> Vec<C> {
        let (g_lo, g_hi) = g.split_at(g.len() / 2);
        g_lo.iter()
            .zip(g_hi.iter())
            .map(|(lo, hi)| (lo.to_curve() + *hi * challenge).to_affine())
            .collect()
    }

    fn check_collapse<C: CurveAffine>() {
        let mut high_bit = C::Scalar::ONE;
        for _ in 0..254 {
            high_bit = high_bit.double();
        }
        let challenges = [
            C::Scalar::ZERO,
            C::Scalar::ONE,
            -C::Scalar::ONE,
            C::Scalar::from(2),
            C::Scalar::from(u64::MAX),
            high_bit,
            -high_bit,
            C::Scalar::random(OsRng),
            C::Scalar::random(OsRng),
        ];
        for len in [2usize, 4, 6, 8, 16, 30, 64, 130, 512, 2048] {
            let mut g: Vec<C> = (0..len)
                .map(|i| match i % 7 {
                    // Identity entries in both halves.
                    3 => C::identity(),
                    // A low/high pair that may cancel for challenge = -1.
                    5 => C::generator(),
                    _ => C::Curve::random(OsRng).to_affine(),
                })
                .collect();
            if len >= 4 {
                g[len / 2] = g[0];
                g[len / 2 + 1] = -g[1];
            }
            for challenge in challenges {
                let expected = naive_collapse(&g, challenge);

                let mut generic = g.clone();
                generic_generator_collapse(&mut generic, challenge);
                assert_eq!(&generic[..len / 2], &expected[..]);

                let mut fast = g.clone();
                parallel_generator_collapse(&mut fast, challenge);
                assert_eq!(&fast[..len / 2], &expected[..]);
                // The upper half is left for the caller to truncate.
                assert_eq!(&fast[len / 2..], &g[len / 2..]);
            }
        }
    }

    #[test]
    fn glv_generator_collapse_dispatch() {
        let g: Vec<EqAffine> = (0..8)
            .map(|_| vesta::Point::random(OsRng).to_affine())
            .collect();
        let challenge = Fp::random(OsRng);

        // A mismatched curve is rejected without touching the generators.
        let mut mismatched = g.clone();
        assert!(!glv_generator_collapse::<EqAffine, pallas::Point>(
            &mut mismatched,
            &challenge
        ));
        assert_eq!(mismatched, g);

        // The matching curve takes the GLV path.
        let mut collapsed = g.clone();
        assert!(glv_generator_collapse::<EqAffine, vesta::Point>(
            &mut collapsed,
            &challenge
        ));
        assert_eq!(&collapsed[..4], &naive_collapse(&g, challenge)[..]);
    }

    #[test]
    fn generator_collapse_matches_naive_vesta() {
        check_collapse::<EqAffine>();
    }

    #[test]
    fn generator_collapse_matches_naive_pallas() {
        check_collapse::<EpAffine>();
    }
}
