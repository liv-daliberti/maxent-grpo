# Independent review: Mei and SetPO corollaries

Reviewed 2026-09-21 against the integrated main.tex. No manuscript edits.

**Verdict: both new mathematical corollaries pass.** The source reductions, normalizations, step condition, rate and shared-parameter qualifications are correct. Three residual narrative points should be cleaned up; none invalidates either corollary.

## Exact Dr.GRPO plus reference KL (`cor:kl-dr-mei-convergence`)

Primary checks: [Mei et al., published Section 4.2.1, Update 2, Lemma 13, Theorem 5](https://proceedings.mlr.press/v119/mei20b/mei20b.pdf), p.7; [supplement, proof of Lemma 13 and Theorem 5](https://proceedings.mlr.press/v119/mei20b/mei20b-supp.pdf), pp.35–37.

- The gradient is exactly `(diag(p)-pp^T)[alpha R+beta log(mu)-beta log(p)]`; the derivative's extra constant entropy term is annihilated by the softmax Jacobian.
- With shifted reward `rtilde=alpha R+beta log(mu)`, normalization by B gives `J_beta=B J_normalized+constant`, `tau=beta/B`, and `eta'=eta B`. Therefore the published step condition `eta'<=1/tau` becomes precisely `eta<=1/beta`, with no missing B factor.
- The rate exponent transforms as `2 tau eta' c=2 beta eta c`. Lemma 13 gives a positive all-iteration probability lower bound independent of eta. The reported C can also be independent of eta: one permitted choice is `2K(beta||z1||_infty+B)^2/beta`, where K is the number of categories.
- The exact gap is `beta KL(p_n||pstar)` in the stated direction. Its convergence implies policy convergence. Finite initial logits and full-support mu give finite transformed rewards and positive temperature.
- No unique reward maximizer is needed. Both reward classes being nonempty are sufficient here.
- The result explicitly covers independently optimized exact discrete gradients. It does not claim sampled KL, PPO or AdamW convergence. MaxRL with G>=3 is correctly left at a stationary-target result. Its G=2 constant-coefficient exception does not undermine that qualification.

## SetPO verified-key specialization (`cor:pcmd-setpo-neural`)

Primary check: [SetPO v1, Assumption 4.1 and Theorem 4.2](https://arxiv.org/html/2602.01062v1#S4.SS1).

The equality kernel is measurable, symmetric and bounded in [0,1], and `g(u)=1-u` is continuously differentiable, nonincreasing and has bounded derivative. Its local mass is `q_{V(y)}`. Thus the population functional is `1-sum_c q_c^2`, and its mixture influence is `2(S2-q_{V(y)})`.

The conditional-score calculation is correct:

\[
\nabla\log Q_\theta(Y)=s_\theta(Y)-\nabla\log P_\theta,
\qquad E_Q[2(S_2-q_{V(Y)})]=0.
\]

The conditioning correction therefore cancels. The expectation is under Q, so no additional factor of P or 1/P belongs outside it. Fixed correctness/key events, integrable score, and differentiation through the finitely many key masses are stated. These suffice for arbitrary shared neural parameters; the corollary correctly makes no optimizer-improvement claim.

Optional precision: append `y in C` to the influence identity's quantifier, since V is defined there. Its present meaning is already implicit in the expression.

## Residual narrative issues

1. **Following `cor:pmd-limits`:** the sentence “a general convergence proof is not supplied here” is now ambiguous after proving exact Dr.GRPO convergence. Recommended wording: “For exact Dr.GRPO gradient steps, Corollary [Mei] also proves convergence to the KL value; for centered MaxRL with G>=3 we identify the stationary target without a convergence claim.” Keep the flow-versus-discrete distinction explicit.

2. **Measured KL subsection heading:** “What the flow got wrong, twice, for one reason” conflicts with the revised paragraph listing finite training, shared parameters and the sampled adaptive optimizer as distinct unresolved explanations. Rename it to describe disagreement with stationary predictions without asserting a uniquely identified cause.

3. **Later measured interpretation:** “replay attains its bound” and “which is a discovery failure and not a retention failure” reintroduce stronger conclusions than the corrected occupancy paragraph allows. Occupancy is not a held-out neural PCMD bound, and these measurements do not identify discovery versus retention as the unique cause. Use “tracks its occupancy summary” and “is consistent with limited discovery” unless additional evidence identifies the mechanism. Similarly, the later statement that a deployed-model anchor “inherits a macro diversity” should be explicitly limited to the analyzed stationary target, not asserted as a guaranteed neural outcome.

The stationary-policy proposition itself correctly permits Dr.GRPO and MaxRL while the new convergence corollary covers only constant-coefficient Dr.GRPO. The reverse-KL sublevel repair and separate two-mode recovery result are compatible with a positive, initialization-dependent trajectory floor; no contradiction remains in those mathematical statements.
