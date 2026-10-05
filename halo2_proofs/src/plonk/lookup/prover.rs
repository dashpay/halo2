use super::super::{
    circuit::Expression, ChallengeBeta, ChallengeGamma, ChallengeTheta, ChallengeX, Error,
    ProvingKey,
};
use super::Argument;
use crate::{
    arithmetic::{eval_polynomial, parallelize, CurveAffine},
    poly::{
        self,
        commitment::{Blind, Params},
        multiopen::ProverQuery,
        Coeff, EvaluationDomain, ExtendedLagrangeCoeff, LagrangeCoeff, Polynomial, Rotation,
    },
    transcript::{EncodedChallenge, TranscriptWrite},
};
use ff::WithSmallOrderMulGroup;
use group::{
    ff::{BatchInvert, Field},
    Curve,
};
use rand_core::RngCore;
use std::{
    collections::BTreeMap,
    iter,
    ops::{Mul, MulAssign},
};

#[derive(Debug)]
pub(in crate::plonk) struct Permuted<C: CurveAffine, Ev> {
    compressed_input_expression: Polynomial<C::Scalar, LagrangeCoeff>,
    permuted_input_expression: Polynomial<C::Scalar, LagrangeCoeff>,
    compressed_input_coset: poly::Ast<Ev, C::Scalar, ExtendedLagrangeCoeff>,
    permuted_input_poly: Polynomial<C::Scalar, Coeff>,
    permuted_input_coset: poly::AstLeaf<Ev, ExtendedLagrangeCoeff>,
    permuted_input_blind: Blind<C::Scalar>,
    compressed_table_expression: Polynomial<C::Scalar, LagrangeCoeff>,
    compressed_table_coset: poly::Ast<Ev, C::Scalar, ExtendedLagrangeCoeff>,
    permuted_table_expression: Polynomial<C::Scalar, LagrangeCoeff>,
    permuted_table_poly: Polynomial<C::Scalar, Coeff>,
    permuted_table_coset: poly::AstLeaf<Ev, ExtendedLagrangeCoeff>,
    permuted_table_blind: Blind<C::Scalar>,
}

/// The randomness that [`Argument::commit_permuted`] uses for one lookup argument.
///
/// It is drawn up front, in the order in which committing to the permuted
/// expressions has always drawn it, so that several lookup arguments can be
/// committed to concurrently without changing the proof for a given RNG.
#[derive(Debug)]
pub(in crate::plonk) struct PermutedBlinding<F: Field> {
    input_rows: Vec<F>,
    table_rows: Vec<F>,
    input_blind: Blind<F>,
    table_blind: Blind<F>,
}

impl<F: Field> PermutedBlinding<F> {
    pub(in crate::plonk) fn draw(blinding_factors: usize, mut rng: impl RngCore) -> Self {
        // Blinding rows of the permuted input expression, then of the permuted
        // table expression, then the blinds of their commitments.
        let input_rows = (0..=blinding_factors)
            .map(|_| F::random(&mut rng))
            .collect();
        let table_rows = (0..=blinding_factors)
            .map(|_| F::random(&mut rng))
            .collect();
        let input_blind = Blind(F::random(&mut rng));
        let table_blind = Blind(F::random(&mut rng));
        PermutedBlinding {
            input_rows,
            table_rows,
            input_blind,
            table_blind,
        }
    }
}

/// A lookup argument whose permuted expressions have been committed to, but whose
/// commitments have not been written to the transcript yet.
#[derive(Debug)]
pub(in crate::plonk) struct PermutedPending<C: CurveAffine, Ev> {
    compressed_input_expression: Polynomial<C::Scalar, LagrangeCoeff>,
    permuted_input_expression: Polynomial<C::Scalar, LagrangeCoeff>,
    compressed_input_coset: poly::Ast<Ev, C::Scalar, ExtendedLagrangeCoeff>,
    permuted_input_poly: Polynomial<C::Scalar, Coeff>,
    permuted_input_coset: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
    permuted_input_blind: Blind<C::Scalar>,
    permuted_input_commitment: C,
    compressed_table_expression: Polynomial<C::Scalar, LagrangeCoeff>,
    compressed_table_coset: poly::Ast<Ev, C::Scalar, ExtendedLagrangeCoeff>,
    permuted_table_expression: Polynomial<C::Scalar, LagrangeCoeff>,
    permuted_table_poly: Polynomial<C::Scalar, Coeff>,
    permuted_table_coset: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
    permuted_table_blind: Blind<C::Scalar>,
    permuted_table_commitment: C,
}

/// The randomness that [`Permuted::commit_product`] uses for one lookup argument,
/// drawn up front for the same reason as [`PermutedBlinding`].
#[derive(Debug)]
pub(in crate::plonk) struct ProductBlinding<F: Field> {
    rows: Vec<F>,
    blind: Blind<F>,
}

impl<F: Field> ProductBlinding<F> {
    pub(in crate::plonk) fn draw(blinding_factors: usize, mut rng: impl RngCore) -> Self {
        // Blinding rows of the grand product, then the blind of its commitment.
        let rows = (0..blinding_factors).map(|_| F::random(&mut rng)).collect();
        let blind = Blind(F::random(&mut rng));
        ProductBlinding { rows, blind }
    }
}

/// A lookup argument whose grand product has been committed to, but whose
/// commitment has not been written to the transcript yet.
#[derive(Debug)]
pub(in crate::plonk) struct CommittedPending<C: CurveAffine, Ev> {
    permuted: Permuted<C, Ev>,
    product_poly: Polynomial<C::Scalar, Coeff>,
    product_coset: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
    product_blind: Blind<C::Scalar>,
    product_commitment: C,
}

#[derive(Debug)]
pub(in crate::plonk) struct Committed<C: CurveAffine, Ev> {
    permuted: Permuted<C, Ev>,
    product_poly: Polynomial<C::Scalar, Coeff>,
    product_coset: poly::AstLeaf<Ev, ExtendedLagrangeCoeff>,
    product_blind: Blind<C::Scalar>,
}

pub(in crate::plonk) struct Constructed<C: CurveAffine> {
    permuted_input_poly: Polynomial<C::Scalar, Coeff>,
    permuted_input_blind: Blind<C::Scalar>,
    permuted_table_poly: Polynomial<C::Scalar, Coeff>,
    permuted_table_blind: Blind<C::Scalar>,
    product_poly: Polynomial<C::Scalar, Coeff>,
    product_blind: Blind<C::Scalar>,
}

pub(in crate::plonk) struct Evaluated<C: CurveAffine> {
    constructed: Constructed<C>,
}

impl<F: WithSmallOrderMulGroup<3>> Argument<F> {
    /// Given a Lookup with input expressions [A_0, A_1, ..., A_{m-1}] and table expressions
    /// [S_0, S_1, ..., S_{m-1}], this method
    /// - constructs A_compressed = \theta^{m-1} A_0 + theta^{m-2} A_1 + ... + \theta A_{m-2} + A_{m-1}
    ///   and S_compressed = \theta^{m-1} S_0 + theta^{m-2} S_1 + ... + \theta S_{m-2} + S_{m-1},
    /// - permutes A_compressed and S_compressed using permute_expression_pair() helper,
    ///   obtaining A' and S', and
    /// - constructs Permuted<C> struct using permuted_input_value = A', and
    ///   permuted_table_expression = S'.
    /// The Permuted<C> struct is used to update the Lookup, and is then returned.
    ///
    /// This draws no randomness (it is passed in as `blinding`) and does not
    /// touch the transcript, so it can run concurrently for several lookups;
    /// [`PermutedPending::finish`] then writes the commitments to the transcript.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::plonk) fn commit_permuted<'a, C, Ev: Copy + Send + Sync, Ec: Copy + Send + Sync>(
        &self,
        pk: &ProvingKey<C>,
        params: &Params<C>,
        domain: &EvaluationDomain<C::Scalar>,
        value_evaluator: &poly::Evaluator<Ev, C::Scalar, LagrangeCoeff>,
        theta: ChallengeTheta<C>,
        advice_values: &'a [poly::AstLeaf<Ev, LagrangeCoeff>],
        fixed_values: &'a [poly::AstLeaf<Ev, LagrangeCoeff>],
        instance_values: &'a [poly::AstLeaf<Ev, LagrangeCoeff>],
        advice_cosets: &'a [poly::AstLeaf<Ec, ExtendedLagrangeCoeff>],
        fixed_cosets: &'a [poly::AstLeaf<Ec, ExtendedLagrangeCoeff>],
        instance_cosets: &'a [poly::AstLeaf<Ec, ExtendedLagrangeCoeff>],
        blinding: PermutedBlinding<C::Scalar>,
    ) -> Result<PermutedPending<C, Ec>, Error>
    where
        C: CurveAffine<ScalarExt = F>,
        C::Curve: Mul<F, Output = C::Curve> + MulAssign<F>,
    {
        // Closure to get values of expressions and compress them
        let compress_expressions = |expressions: &[Expression<C::Scalar>]| {
            // Values of input expressions involved in the lookup
            let unpermuted_expressions: Vec<_> = expressions
                .iter()
                .map(|expression| {
                    expression.evaluate(
                        &|scalar| poly::Ast::ConstantTerm(scalar),
                        &|_| panic!("virtual selectors are removed during optimization"),
                        &|query| {
                            fixed_values[query.column_index]
                                .with_rotation(query.rotation)
                                .into()
                        },
                        &|query| {
                            advice_values[query.column_index]
                                .with_rotation(query.rotation)
                                .into()
                        },
                        &|query| {
                            instance_values[query.column_index]
                                .with_rotation(query.rotation)
                                .into()
                        },
                        &|a| -a,
                        &|a, b| a + b,
                        &|a, b| a * b,
                        &|a, scalar| a * scalar,
                    )
                })
                .collect();

            let unpermuted_cosets: Vec<_> = expressions
                .iter()
                .map(|expression| {
                    expression.evaluate(
                        &|scalar| poly::Ast::ConstantTerm(scalar),
                        &|_| panic!("virtual selectors are removed during optimization"),
                        &|query| {
                            fixed_cosets[query.column_index]
                                .with_rotation(query.rotation)
                                .into()
                        },
                        &|query| {
                            advice_cosets[query.column_index]
                                .with_rotation(query.rotation)
                                .into()
                        },
                        &|query| {
                            instance_cosets[query.column_index]
                                .with_rotation(query.rotation)
                                .into()
                        },
                        &|a| -a,
                        &|a, b| a + b,
                        &|a, b| a * b,
                        &|a, scalar| a * scalar,
                    )
                })
                .collect();

            // Compressed version of expressions
            let compressed_expression = unpermuted_expressions.iter().fold(
                poly::Ast::ConstantTerm(C::Scalar::ZERO),
                |acc, expression| &(acc * *theta) + expression,
            );

            // Compressed version of cosets
            let compressed_coset = unpermuted_cosets.iter().fold(
                poly::Ast::<_, _, ExtendedLagrangeCoeff>::ConstantTerm(C::Scalar::ZERO),
                |acc, eval| acc * poly::Ast::ConstantTerm(*theta) + eval.clone(),
            );

            (
                compressed_coset,
                value_evaluator.evaluate(&compressed_expression, domain),
            )
        };

        // Get values of input expressions involved in the lookup and compress them
        let (compressed_input_coset, compressed_input_expression) =
            compress_expressions(&self.input_expressions);

        // Get values of table expressions involved in the lookup and compress them
        let (compressed_table_coset, compressed_table_expression) =
            compress_expressions(&self.table_expressions);

        // Permute compressed (InputExpression, TableExpression) pair
        let (permuted_input_expression, permuted_table_expression) = permute_expression_pair::<C>(
            pk,
            params,
            domain,
            blinding.input_rows,
            blinding.table_rows,
            &compressed_input_expression,
            &compressed_table_expression,
        )?;

        // Closure to construct commitment to vector of values
        let commit_values = |values: &Polynomial<C::Scalar, LagrangeCoeff>, blind| {
            let poly = pk.vk.domain.lagrange_to_coeff(values.clone());
            let commitment = params.commit_lagrange(values, blind).to_affine();
            let coset = pk.vk.domain.coeff_to_extended(poly.clone());
            (poly, coset, commitment)
        };

        // Commit to permuted input expression
        let (permuted_input_poly, permuted_input_coset, permuted_input_commitment) =
            commit_values(&permuted_input_expression, blinding.input_blind);

        // Commit to permuted table expression
        let (permuted_table_poly, permuted_table_coset, permuted_table_commitment) =
            commit_values(&permuted_table_expression, blinding.table_blind);

        Ok(PermutedPending {
            compressed_input_expression,
            compressed_input_coset,
            permuted_input_expression,
            permuted_input_poly,
            permuted_input_coset,
            permuted_input_blind: blinding.input_blind,
            permuted_input_commitment,
            compressed_table_expression,
            compressed_table_coset,
            permuted_table_expression,
            permuted_table_poly,
            permuted_table_coset,
            permuted_table_blind: blinding.table_blind,
            permuted_table_commitment,
        })
    }
}

impl<C: CurveAffine, Ev> PermutedPending<C, Ev> {
    /// Writes the commitments to the permuted input and table expressions to the
    /// transcript, and registers their cosets with `coset_evaluator`.
    pub(in crate::plonk) fn finish<E: EncodedChallenge<C>, T: TranscriptWrite<C, E>>(
        self,
        coset_evaluator: &mut poly::Evaluator<Ev, C::Scalar, ExtendedLagrangeCoeff>,
        transcript: &mut T,
    ) -> Result<Permuted<C, Ev>, Error> {
        // Hash permuted input commitment
        transcript.write_point(self.permuted_input_commitment)?;

        // Hash permuted table commitment
        transcript.write_point(self.permuted_table_commitment)?;

        let permuted_input_coset = coset_evaluator.register_poly(self.permuted_input_coset);
        let permuted_table_coset = coset_evaluator.register_poly(self.permuted_table_coset);

        Ok(Permuted {
            compressed_input_expression: self.compressed_input_expression,
            compressed_input_coset: self.compressed_input_coset,
            permuted_input_expression: self.permuted_input_expression,
            permuted_input_poly: self.permuted_input_poly,
            permuted_input_coset,
            permuted_input_blind: self.permuted_input_blind,
            compressed_table_expression: self.compressed_table_expression,
            compressed_table_coset: self.compressed_table_coset,
            permuted_table_expression: self.permuted_table_expression,
            permuted_table_poly: self.permuted_table_poly,
            permuted_table_coset,
            permuted_table_blind: self.permuted_table_blind,
        })
    }
}

impl<C: CurveAffine, Ev: Copy + Send + Sync> Permuted<C, Ev> {
    /// Given a Lookup with input expressions, table expressions, and the permuted
    /// input expression and permuted table expression, this method constructs the
    /// grand product polynomial over the lookup. The grand product polynomial
    /// is used to populate the Product<C> struct. The Product<C> struct is
    /// added to the Lookup and finally returned by the method.
    ///
    /// Like [`Argument::commit_permuted`], this takes its randomness as `blinding`
    /// and leaves writing the commitment to [`CommittedPending::finish`].
    pub(in crate::plonk) fn commit_product(
        self,
        pk: &ProvingKey<C>,
        params: &Params<C>,
        beta: ChallengeBeta<C>,
        gamma: ChallengeGamma<C>,
        blinding: ProductBlinding<C::Scalar>,
    ) -> CommittedPending<C, Ev> {
        let blinding_factors = pk.vk.cs.blinding_factors();
        // Goal is to compute the products of fractions
        //
        // Numerator: (\theta^{m-1} a_0(\omega^i) + \theta^{m-2} a_1(\omega^i) + ... + \theta a_{m-2}(\omega^i) + a_{m-1}(\omega^i) + \beta)
        //            * (\theta^{m-1} s_0(\omega^i) + \theta^{m-2} s_1(\omega^i) + ... + \theta s_{m-2}(\omega^i) + s_{m-1}(\omega^i) + \gamma)
        // Denominator: (a'(\omega^i) + \beta) (s'(\omega^i) + \gamma)
        //
        // where a_j(X) is the jth input expression in this lookup,
        // where a'(X) is the compression of the permuted input expressions,
        // s_j(X) is the jth table expression in this lookup,
        // s'(X) is the compression of the permuted table expressions,
        // and i is the ith row of the expression.
        let mut lookup_product = vec![C::Scalar::ZERO; params.n as usize];
        // Denominator uses the permuted input expression and permuted table expression
        parallelize(&mut lookup_product, |lookup_product, start| {
            for ((lookup_product, permuted_input_value), permuted_table_value) in lookup_product
                .iter_mut()
                .zip(self.permuted_input_expression[start..].iter())
                .zip(self.permuted_table_expression[start..].iter())
            {
                *lookup_product = (*beta + permuted_input_value) * &(*gamma + permuted_table_value);
            }
        });

        // Batch invert to obtain the denominators for the lookup product
        // polynomials
        lookup_product.iter_mut().batch_invert();

        // Finish the computation of the entire fraction by computing the numerators
        // (\theta^{m-1} a_0(\omega^i) + \theta^{m-2} a_1(\omega^i) + ... + \theta a_{m-2}(\omega^i) + a_{m-1}(\omega^i) + \beta)
        // * (\theta^{m-1} s_0(\omega^i) + \theta^{m-2} s_1(\omega^i) + ... + \theta s_{m-2}(\omega^i) + s_{m-1}(\omega^i) + \gamma)
        parallelize(&mut lookup_product, |product, start| {
            for ((product, &input_term), &table_term) in product
                .iter_mut()
                .zip(self.compressed_input_expression[start..].iter())
                .zip(self.compressed_table_expression[start..].iter())
            {
                *product *= &(input_term + &*beta);
                *product *= &(table_term + &*gamma);
            }
        });

        // The product vector is a vector of products of fractions of the form
        //
        // Numerator: (\theta^{m-1} a_0(\omega^i) + \theta^{m-2} a_1(\omega^i) + ... + \theta a_{m-2}(\omega^i) + a_{m-1}(\omega^i) + \beta)
        //            * (\theta^{m-1} s_0(\omega^i) + \theta^{m-2} s_1(\omega^i) + ... + \theta s_{m-2}(\omega^i) + s_{m-1}(\omega^i) + \gamma)
        // Denominator: (a'(\omega^i) + \beta) (s'(\omega^i) + \gamma)
        //
        // where there are m input expressions and m table expressions,
        // a_j(\omega^i) is the jth input expression in this lookup,
        // a'j(\omega^i) is the permuted input expression,
        // s_j(\omega^i) is the jth table expression in this lookup,
        // s'(\omega^i) is the permuted table expression,
        // and i is the ith row of the expression.

        // Compute the evaluations of the lookup product polynomial
        // over our domain, starting with z[0] = 1
        let z = iter::once(C::Scalar::ONE)
            .chain(lookup_product)
            .scan(C::Scalar::ONE, |state, cur| {
                *state *= &cur;
                Some(*state)
            })
            // Take all rows including the "last" row which should
            // be a boolean (and ideally 1, else soundness is broken)
            .take(params.n as usize - blinding_factors)
            // Chain random blinding factors.
            .chain(blinding.rows)
            .collect::<Vec<_>>();
        assert_eq!(z.len(), params.n as usize);
        let z = pk.vk.domain.lagrange_from_vec(z);

        #[cfg(feature = "sanity-checks")]
        // This test works only with intermediate representations in this method.
        // It can be used for debugging purposes.
        {
            // While in Lagrange basis, check that product is correctly constructed
            let u = (params.n as usize) - (blinding_factors + 1);

            // l_0(X) * (1 - z(X)) = 0
            assert_eq!(z[0], C::Scalar::ONE);

            // z(\omega X) (a'(X) + \beta) (s'(X) + \gamma)
            // - z(X) (\theta^{m-1} a_0(X) + ... + a_{m-1}(X) + \beta) (\theta^{m-1} s_0(X) + ... + s_{m-1}(X) + \gamma)
            for i in 0..u {
                let mut left = z[i + 1];
                let permuted_input_value = &self.permuted_input_expression[i];

                let permuted_table_value = &self.permuted_table_expression[i];

                left *= &(*beta + permuted_input_value);
                left *= &(*gamma + permuted_table_value);

                let mut right = z[i];
                let mut input_term = self.compressed_input_expression[i];

                let mut table_term = self.compressed_table_expression[i];

                input_term += &(*beta);
                table_term += &(*gamma);
                right *= &(input_term * &table_term);

                assert_eq!(left, right);
            }

            // l_last(X) * (z(X)^2 - z(X)) = 0
            // Assertion will fail only when soundness is broken, in which
            // case this z[u] value will be zero. (bad!)
            assert_eq!(z[u], C::Scalar::ONE);
        }

        let product_blind = blinding.blind;
        let product_commitment = params.commit_lagrange(&z, product_blind).to_affine();
        let z = pk.vk.domain.lagrange_to_coeff(z);
        let product_coset = pk.vk.domain.coeff_to_extended(z.clone());

        CommittedPending {
            permuted: self,
            product_poly: z,
            product_coset,
            product_blind,
            product_commitment,
        }
    }
}

impl<C: CurveAffine, Ev> CommittedPending<C, Ev> {
    /// Registers the coset of the grand product with `evaluator`, and writes its
    /// commitment to the transcript.
    pub(in crate::plonk) fn finish<E: EncodedChallenge<C>, T: TranscriptWrite<C, E>>(
        self,
        evaluator: &mut poly::Evaluator<Ev, C::Scalar, ExtendedLagrangeCoeff>,
        transcript: &mut T,
    ) -> Result<Committed<C, Ev>, Error> {
        let product_coset = evaluator.register_poly(self.product_coset);

        // Hash product commitment
        transcript.write_point(self.product_commitment)?;

        Ok(Committed::<C, _> {
            permuted: self.permuted,
            product_poly: self.product_poly,
            product_coset,
            product_blind: self.product_blind,
        })
    }
}

impl<'a, C: CurveAffine, Ev: Copy + Send + Sync + 'a> Committed<C, Ev> {
    /// Given a Lookup with input expressions, table expressions, permuted input
    /// expression, permuted table expression, and grand product polynomial, this
    /// method constructs constraints that must hold between these values.
    /// This method returns the constraints as a vector of ASTs for polynomials in
    /// the extended evaluation domain.
    pub(in crate::plonk) fn construct(
        self,
        beta: ChallengeBeta<C>,
        gamma: ChallengeGamma<C>,
        l0: poly::AstLeaf<Ev, ExtendedLagrangeCoeff>,
        l_blind: poly::AstLeaf<Ev, ExtendedLagrangeCoeff>,
        l_last: poly::AstLeaf<Ev, ExtendedLagrangeCoeff>,
    ) -> (
        Constructed<C>,
        impl Iterator<Item = poly::Ast<Ev, C::Scalar, ExtendedLagrangeCoeff>> + 'a,
    ) {
        let permuted = self.permuted;

        let active_rows = poly::Ast::one() - (poly::Ast::from(l_last) + l_blind);
        let beta = poly::Ast::ConstantTerm(*beta);
        let gamma = poly::Ast::ConstantTerm(*gamma);

        let expressions = iter::empty()
            // l_0(X) * (1 - z(X)) = 0
            .chain(Some((poly::Ast::one() - self.product_coset) * l0))
            // l_last(X) * (z(X)^2 - z(X)) = 0
            .chain(Some(
                (poly::Ast::from(self.product_coset) * self.product_coset - self.product_coset)
                    * l_last,
            ))
            // (1 - (l_last(X) + l_blind(X))) * (
            //   z(\omega X) (a'(X) + \beta) (s'(X) + \gamma)
            //   - z(X) (\theta^{m-1} a_0(X) + ... + a_{m-1}(X) + \beta) (\theta^{m-1} s_0(X) + ... + s_{m-1}(X) + \gamma)
            // ) = 0
            .chain({
                // z(\omega X) (a'(X) + \beta) (s'(X) + \gamma)
                let left: poly::Ast<_, _, _> = poly::Ast::<_, C::Scalar, _>::from(
                    self.product_coset.with_rotation(Rotation::next()),
                ) * (poly::Ast::from(permuted.permuted_input_coset)
                    + beta.clone())
                    * (poly::Ast::from(permuted.permuted_table_coset) + gamma.clone());

                //  z(X) (\theta^{m-1} a_0(X) + ... + a_{m-1}(X) + \beta) (\theta^{m-1} s_0(X) + ... + s_{m-1}(X) + \gamma)
                let right: poly::Ast<_, _, _> = poly::Ast::from(self.product_coset)
                    * (permuted.compressed_input_coset + beta)
                    * (permuted.compressed_table_coset + gamma);

                Some((left - right) * active_rows.clone())
            })
            // Check that the first values in the permuted input expression and permuted
            // fixed expression are the same.
            // l_0(X) * (a'(X) - s'(X)) = 0
            .chain(Some(
                (poly::Ast::from(permuted.permuted_input_coset) - permuted.permuted_table_coset)
                    * l0,
            ))
            // Check that each value in the permuted lookup input expression is either
            // equal to the value above it, or the value at the same index in the
            // permuted table expression.
            // (1 - (l_last + l_blind)) * (a′(X) − s′(X))⋅(a′(X) − a′(\omega^{-1} X)) = 0
            .chain(Some(
                (poly::Ast::<_, C::Scalar, _>::from(permuted.permuted_input_coset)
                    - permuted.permuted_table_coset)
                    * (poly::Ast::from(permuted.permuted_input_coset)
                        - permuted
                            .permuted_input_coset
                            .with_rotation(Rotation::prev()))
                    * active_rows,
            ));

        (
            Constructed {
                permuted_input_poly: permuted.permuted_input_poly,
                permuted_input_blind: permuted.permuted_input_blind,
                permuted_table_poly: permuted.permuted_table_poly,
                permuted_table_blind: permuted.permuted_table_blind,
                product_poly: self.product_poly,
                product_blind: self.product_blind,
            },
            expressions,
        )
    }
}

impl<C: CurveAffine> Constructed<C> {
    pub(in crate::plonk) fn evaluate<E: EncodedChallenge<C>, T: TranscriptWrite<C, E>>(
        self,
        pk: &ProvingKey<C>,
        x: ChallengeX<C>,
        transcript: &mut T,
    ) -> Result<Evaluated<C>, Error> {
        let domain = &pk.vk.domain;
        let x_inv = domain.rotate_omega(*x, Rotation::prev());
        let x_next = domain.rotate_omega(*x, Rotation::next());

        let product_eval = eval_polynomial(&self.product_poly, *x);
        let product_next_eval = eval_polynomial(&self.product_poly, x_next);
        let permuted_input_eval = eval_polynomial(&self.permuted_input_poly, *x);
        let permuted_input_inv_eval = eval_polynomial(&self.permuted_input_poly, x_inv);
        let permuted_table_eval = eval_polynomial(&self.permuted_table_poly, *x);

        // Hash each advice evaluation
        for eval in iter::empty()
            .chain(Some(product_eval))
            .chain(Some(product_next_eval))
            .chain(Some(permuted_input_eval))
            .chain(Some(permuted_input_inv_eval))
            .chain(Some(permuted_table_eval))
        {
            transcript.write_scalar(eval)?;
        }

        Ok(Evaluated { constructed: self })
    }
}

impl<C: CurveAffine> Evaluated<C> {
    pub(in crate::plonk) fn open<'a>(
        &'a self,
        pk: &'a ProvingKey<C>,
        x: ChallengeX<C>,
    ) -> impl Iterator<Item = ProverQuery<'a, C>> + Clone {
        let x_inv = pk.vk.domain.rotate_omega(*x, Rotation::prev());
        let x_next = pk.vk.domain.rotate_omega(*x, Rotation::next());

        iter::empty()
            // Open lookup product commitments at x
            .chain(Some(ProverQuery {
                point: *x,
                poly: &self.constructed.product_poly,
                blind: self.constructed.product_blind,
            }))
            // Open lookup input commitments at x
            .chain(Some(ProverQuery {
                point: *x,
                poly: &self.constructed.permuted_input_poly,
                blind: self.constructed.permuted_input_blind,
            }))
            // Open lookup table commitments at x
            .chain(Some(ProverQuery {
                point: *x,
                poly: &self.constructed.permuted_table_poly,
                blind: self.constructed.permuted_table_blind,
            }))
            // Open lookup input commitments at x_inv
            .chain(Some(ProverQuery {
                point: x_inv,
                poly: &self.constructed.permuted_input_poly,
                blind: self.constructed.permuted_input_blind,
            }))
            // Open lookup product commitments at x_next
            .chain(Some(ProverQuery {
                point: x_next,
                poly: &self.constructed.product_poly,
                blind: self.constructed.product_blind,
            }))
    }
}

type ExpressionPair<F> = (Polynomial<F, LagrangeCoeff>, Polynomial<F, LagrangeCoeff>);

/// Given a vector of input values A and a vector of table values S,
/// this method permutes A and S to produce A' and S', such that:
/// - like values in A' are vertically adjacent to each other; and
/// - the first row in a sequence of like values in A' is the row
///   that has the corresponding value in S'.
/// This method returns (A', S') if no errors are encountered.
///
/// `input_blinding_rows` and `table_blinding_rows` are the random values that fill
/// the `blinding_factors + 1` rows at the end of A' and S'.
fn permute_expression_pair<C: CurveAffine>(
    pk: &ProvingKey<C>,
    params: &Params<C>,
    domain: &EvaluationDomain<C::Scalar>,
    input_blinding_rows: Vec<C::Scalar>,
    table_blinding_rows: Vec<C::Scalar>,
    input_expression: &Polynomial<C::Scalar, LagrangeCoeff>,
    table_expression: &Polynomial<C::Scalar, LagrangeCoeff>,
) -> Result<ExpressionPair<C::Scalar>, Error> {
    let blinding_factors = pk.vk.cs.blinding_factors();
    let usable_rows = params.n as usize - (blinding_factors + 1);

    let mut permuted_input_expression: Vec<C::Scalar> = input_expression.to_vec();
    permuted_input_expression.truncate(usable_rows);

    // Sort input lookup expression values
    permuted_input_expression.sort();

    // A BTreeMap of each unique element in the table expression and its count
    let mut leftover_table_map: BTreeMap<C::Scalar, u32> = table_expression
        .iter()
        .take(usable_rows)
        .fold(BTreeMap::new(), |mut acc, coeff| {
            *acc.entry(*coeff).or_insert(0) += 1;
            acc
        });
    let mut permuted_table_coeffs = vec![C::Scalar::ZERO; usable_rows];

    let mut repeated_input_rows = permuted_input_expression
        .iter()
        .zip(permuted_table_coeffs.iter_mut())
        .enumerate()
        .filter_map(|(row, (input_value, table_value))| {
            // If this is the first occurrence of `input_value` in the input expression
            if row == 0 || *input_value != permuted_input_expression[row - 1] {
                *table_value = *input_value;
                // Remove one instance of input_value from leftover_table_map
                if let Some(count) = leftover_table_map.get_mut(input_value) {
                    assert!(*count > 0);
                    *count -= 1;
                    None
                } else {
                    // Return error if input_value not found
                    Some(Err(Error::ConstraintSystemFailure))
                }
            // If input value is repeated
            } else {
                Some(Ok(row))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Populate permuted table at unfilled rows with leftover table elements
    for (coeff, count) in leftover_table_map.iter() {
        for _ in 0..*count {
            permuted_table_coeffs[repeated_input_rows.pop().unwrap()] = *coeff;
        }
    }
    assert!(repeated_input_rows.is_empty());

    permuted_input_expression.extend(input_blinding_rows);
    permuted_table_coeffs.extend(table_blinding_rows);
    assert_eq!(permuted_input_expression.len(), params.n as usize);
    assert_eq!(permuted_table_coeffs.len(), params.n as usize);

    #[cfg(feature = "sanity-checks")]
    {
        let mut last = None;
        for (a, b) in permuted_input_expression
            .iter()
            .zip(permuted_table_coeffs.iter())
            .take(usable_rows)
        {
            if *a != *b {
                assert_eq!(*a, last.unwrap());
            }
            last = Some(*a);
        }
    }

    Ok((
        domain.lagrange_from_vec(permuted_input_expression),
        domain.lagrange_from_vec(permuted_table_coeffs),
    ))
}
