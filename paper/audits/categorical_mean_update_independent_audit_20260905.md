# Independent categorical mean-update and collapse audit, 2026-09-05

Scope: read-only audit of paper/main.tex lines 843–1063 and the corresponding paper/mathai2026/appendix.tex lines 490–709. Line references reflect the source read at approximately 14:48 EDT, before possible concurrent manuscript edits. Manuscripts and scheduler were not modified.

## Verdict

The finite-group mean-gradient formulas, MaxRL potential, independent-logit dynamics, infinite effective-time argument, unique-winner theorem, and tied-winner conclusion are mathematically correct under the stated finite categorical, independently trained Euclidean-logit, unregularized continuous mean-flow model. No proved counterexample to those theorem statements was found. The changes below strengthen precision and remove unproved extrapolation; they do not require replacing the central result.

## Concrete corrections, classified

1. **Missing local scope/metric qualifier in prose, substantive if read as a general parameterized-policy claim.** main.tex 315–320 and mathai appendix 161–166 say merely “gradient ascent on expected binary reward” before asserting d log(p_c/p_d)/dt = (p_c-p_d)(1-P). This equation needs Euclidean ascent in independently parameterized categorical logits. It is false for general neural-network parameters, preconditioning, or shared correctness directions. The appendix overall supplies the intended assumption, but the displayed main-prose claim should state it locally. Suggested wording: “Under Euclidean gradient ascent on expected binary reward in independently trained categorical logits, ...”. Similarly main.tex 1002–1004 / appendix 649–651 should say “Thus these GRPO, Dr.GRPO, and binary-MaxRL mean flows converge ...”, rather than stating convergence of the finite algorithms without the qualifier.

2. **Small proof-completeness gap, not a false conclusion.** main.tex 1041–1046 / appendix 688–693 infer that every limit point is uniform on its support from nondecrease of sum q_c^2. A compact-state invariance or uniform-continuity argument is needed to justify the inference. An elementary ratio bound avoids that theorem entirely: let Q=q_* and r_j=q_j/Q, with r_j(0)<1. Ordering is preserved, so r_j is nonincreasing and Q>=1/m. Consequently

   d log r_j/d tau = -Q(1-r_j) <= -(1-r_j(0))/m,

   hence r_j(tau) <= r_j(0) exp[-(1-r_j(0))tau/m] -> 0. Thus Q=1/(1+sum_j r_j)->1. For k tied leaders symmetry preserves equal Q, every outsider satisfies the same bound, and kQ + Q sum_outside r_j=1 gives Q->1/k. This proves the theorem and tie corollary directly and supplies a quantitative effective-time rate.

3. **Unproved comparative practical claim.** main.tex 1059–1062 / appendix 706–709 say “Finite groups make this instability practically harsher.” The listed sampling effects do not prove a monotone worsening compared with larger groups or deterministic flow. Indeed the Dr.GRPO mean vector field is multiplied by (G-1)/G, so at a fixed ordinary time the finite-G mean flow is slower than the G->infinity limit. This does not negate the possible tie-breaking effect of sampling noise, but it prevents a general comparative claim. Suggested replacement: “Finite groups also introduce sampling effects: ...”. No stochastic convergence theorem is established here.

4. **Minor conventions and implementation mapping.** main.tex 882–887 / appendix 529–534 should specify fixed finite positive mixed-group weights 0<w(l)<infinity and define the all-equal group update as zero, rather than implicitly allowing 0 times an undefined reciprocal standard deviation. These conventions are satisfied by the implementation. The common response-length factor should be acknowledged for both variants, not only Dr.GRPO, if the claim is intended to identify the repository's implementation. Current code uses masked_sum with constant_normalizer=generate_max_length for all critic variants (src/oat_drgrpo/learner/init.py 234–236 and grpo.py 556–560), which fits the categorical reduction. It does not use a response-dependent 1/L_i factor. Such a factor would generally invalidate proportionality to the binary correctness gradient, so the theorem should not silently cover that alternative GRPO normalization. The actual GRPO reward std is sample std plus 1e-8 (grpo.py 4230–4236), which is an admissible positive w(M); no formula correction is needed. MaxRL binary advantages agree exactly with src/oat_drgrpo/maxrl.py and grpo.py 4225–4226.

5. **Minor notation/presentation only.** The MaxRL lemma uses lower-case r_i despite defining R_i in the categorical setup (main.tex 931 / appendix 578). For absent categories the centered score sum has zero direct logit coordinate, because the advantages sum to zero; probabilities can still change via the softmax denominator. “No direct sampled score term” is clearer than “no score carrier.” This is already phrased more carefully in the mathai prose at 171.

## Independent derivation

Let v_i=grad_z log p_{A_i}; conditional on rewards, E[v_i|R_i=1]=grad P/P and E[v_i|R_i=0]=-grad P/(1-P). For fixed M, summing the centered group advantages gives

  E[ghat | M] = w(M) M(G-M)/(G^2 P(1-P)) grad P.

Thus c_G(P) is the manuscript expression. The binomial identity k(G-k) C(G,k)=G(G-1) C(G-2,k-1) additionally gives

  c_G(P) = (G-1)/G E[w(B+1)],  B~Binomial(G-2,P).

For fixed finite positive mixed-group weights this is a polynomial, extends continuously to both boundaries, and is bounded above and away from zero. Its endpoints are (G-1)w(1)/G and (G-1)w(G-1)/G. This also supplies global smooth bounded coefficients for the finite-logit flow. Dr.GRPO has c=(G-1)/G.

For MaxRL, E[ghat | M>0] as a function of realized M is (G-M)/(G P(1-P)) grad P, and E[(G-M)1{M>0}]=G[(1-P)-(1-P)^G]. Therefore

  c_MaxRL(P) = [1-(1-P)^(G-1)]/P = sum_{j=0}^{G-2}(1-P)^j.

Its endpoint values are G-1 and 1, and its integral from zero is sum_{k=1}^{G-1}[1-(1-P)^k]/k. All constants and indexing in the manuscript are correct, including G=2.

Under zdot=c(P)grad P, correct logits obey zdot_c=c p_c(1-P), incorrect logits zdot_a=-c p_a P. Therefore Pdot=c||grad P||^2 and

  ||grad P||^2=(1-P)^2 sum_correct p_c^2 + P^2 sum_incorrect p_a^2
              >=P^2(1-P)^2(1/m+1/n).

Finite logits remain finite at finite time because the vector field is bounded, hence 0<P(t)<1. P is increasing, and any interior limit would have derivative bounded away from zero. Thus P->1.

Define tau_dot=cP(1-P). Then |zdot_j|<=tau_dot for every coordinate. If total tau were finite, every logit would have finite total variation and a finite limit, which contradicts P->1. Thus tau->infinity. Since q is softmax restricted to correct logits, dz_c/dtau=q_c and

  dq_c/dtau=q_c(q_c-sum_d q_d^2),
  d log(q_c/q_d)/dtau=q_c-q_d.

The direct ratio bound above proves the unique and tied leader limits without any omitted asymptotic argument.

## Boundary and interpretation checks

- G>=2 is essential: G=1 produces zero centered group advantage and no learning.
- Finite logits, m>=2, n>=1 imply strictly positive p for every category and 0<P<1. Exact P=1 would freeze the unregularized task flow and preserve arbitrary q; exact P=0 also produces no update and leaves q undefined. These cases are excluded correctly.
- All m correct coordinates tied initially stay uniform for all effective times. Exactly k<m tied leaders survive at equal limiting mass 1/k; the theorem does not break deterministic exact ties.
- Extinction is asymptotic. Finite logits imply strictly positive probability at every finite time; no finite-time exact zero is proved.
- The theorem concerns infinite-time continuous Euclidean mean flow. It does not establish finite-budget or stochastic-algorithm collapse, Adam dynamics, later PPO clipping epochs, other regularizers, response-dependent row weights, or arbitrary shared neural-network geometry.
- Mapping execution modes to independent categories is a model reduction. The conditional distribution is the same observable, but aggregating real text outputs into keys does not itself make mode logits independently trainable.

## Independent arithmetic verification

Enumerated every ordered group for G=2,3,4,5,6 across four categorical distributions, including correctness masses 0.15, 0.25, 0.5 and 0.9 and a case with two incorrect categories. Tested Dr.GRPO, population-standardized GRPO, implementation-style sample-standardized GRPO with epsilon, and MaxRL. All 80 comparisons between exhaustive finite sums and the claimed c(P)grad P agree to floating point precision (maximum absolute discrepancy 9.3259e-15). Reproduce with: python paper/audits/verify_categorical_group_expectations_20260905.py --output paper/audits/verify_categorical_group_expectations_20260905.json. This is a supplementary arithmetic sanity check; the analytic derivation above proves the formulas generally.
