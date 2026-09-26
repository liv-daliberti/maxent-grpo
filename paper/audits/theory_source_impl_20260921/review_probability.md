# Independent review of integrated probability extensions

Reviewed `paper/main.tex` on 2026-09-21, by theorem/equation label rather than unstable line numbers. No manuscript edits were made.

**Verdict: the three requested results are mathematically correct under their stated assumptions. No blocking defect found.** The sharpness endpoints, prompt normalization, Ville argument, and noise constant all pass. One small clarification is recommended in the optimizer example below.

## Sharp categorical retention

`cor:replay-sharp-kl-certificate` correctly uses the normalized target extended by zero outside the bank. Its KL equals `R_w(p)-H(w)`, so the stated excess budget is nonnegative. Binary coarsening yields the correct inverse-KL coordinate floor; coarsening to the entire bank yields `p(B)>=exp(-D)`. Pinsker has the correct factor `sqrt(D/2)`.

The endpoint conventions are sound: `D=0` forces the target itself; for a singleton bank, the lower inverse is `exp(-D)`. When `0<w_b<1`, the displayed sharpness construction puts the complement in the target's relative proportions and attains the coordinate floor with KL exactly D. A second category is necessary in the singleton construction. With zeros outside the bank, the full-support claim correctly says **infimum**, and restricts the approximation statement to `D>0`. In that case one can first mix the extremizer slightly toward the target to create strict KL slack, then add arbitrarily small full-support mass. The simultaneous-in-time application uses the deterministic replay energy bound correctly.

Rechecked primary attribution: van Erven and Harremoës, [Theorem 9](https://arxiv.org/html/1206.2459v2#S3.SS1) includes order one and deterministic partitions; [Theorem 31](https://arxiv.org/html/1206.2459v2#S4.SS1) gives the cited Pinsker consequence. Their variation convention is full L1 distance; the paper's coordinate constant remains correct.

## Complete-exemplar certificate

`eq:complete-exemplar-key-certificate` correctly normalizes `alpha_j/L_j`, not `alpha_j`. With `A_x=sum_j alpha_j/L_j`, normalized response cross entropy is bounded by `C_x/A_x`, and subtracting `H(w)` gives exactly the response KL budget. Distinct complete responses are disjoint atoms, so their normalized target is a probability law. Coarsening by verifier key gives the target mass `W_b` and the unconditional policy key probability appearing in the binary divergence. Different exemplars can share a key without invalidating the construction. `W_b=1` is handled by the defined endpoint; the inverse is only asserted for `W_b>0`.

The nearby qualifications correctly require complete response/termination events and the same decoding/verifier law, keep normalization separate for each prompt, and distinguish checkpoint certificates from uniform trajectory bounds. A bound on the total nonnegative replay loss does indeed bound each prompt's contribution. These results do not require independent neural parameters or held-out-prompt generalization.

## Stochastic energy certificate

`cor:replay-stochastic-energy-budget` is valid for finite N and N=infinity. The stipulated integrability, adaptedness, deterministic initial value and deterministic summable error budget make `X_t=D_t+sum_{s=t}^{N-1}d_s` an integrable nonnegative supermartingale, with initial value `D_0+B_N`. On its common maximal-bound event, every replay summand is at most `D_t<=X_t`, which yields all exemplar floors simultaneously without a union factor. The zero-initial-budget case is explicitly and correctly separated. For infinite horizon, `Pr(sup_t X_t>=a)<=X_0/a` and `a` tending to infinity establish a finite almost-sure supremum; finite positive weights and lengths then give the asserted positive random infima.

Rechecked primary attribution: [Howard et al., Lemma 1, Eq. (2.11)](https://arxiv.org/html/1808.03204#S2.SS3) is precisely the required maximal inequality. No independence or bounded increments are needed once the displayed conditional drift holds.

## Wang mapping and a recommended precision edit

The stated step threshold and error constant agree with [Wang et al., Lemma 2.1, Eq. (2.13)](https://optimization-online.org/wp-content/uploads/2016/07/5529.pdf). Calling this the **one-step estimate underlying** the lemma is accurate: the proof's one-step inequality does not require the lemma's additional infinite-horizon step-sum assumptions. The paper separately imposes a finite accumulated noise budget when it needs infinite-horizon retention.

For maximal precision, explicitly state the optimizer update and make the gradient moments conditional in that example. Suggested replacement for its opening sentence:

> For the update `theta_{t+1}=theta_t-eta_t H_t g_t`, the one-step estimate underlying Wang et al. [Lemma 2.1, Eq. (2.13)] supplies this drift when F is L-smooth, `E[g_t|F_t]=grad F(theta_t)`, `E[||g_t-grad F(theta_t)||^2|F_t]<=sigma^2/m_t`, and H_t is F_t-measurable with `kappa_lower I <= H_t <= kappa_upper I`.

Then retain the present step-size/error-budget and scope sentences. The current prose strongly implies this update and conditional interpretation, so this is a clarification rather than a counterexample to the corollary. The predicted drift is obtained by discarding the nonpositive gradient term whose coefficient is at least `eta_t*kappa_lower/2` under the stated step limit.

The exclusions concerning current-gradient AdamW moments, clipping, changing banks and cyclic scheduling are appropriate. The surrounding appendix recap correctly says that optimizer-specific energy control has not been established for the experiments. These additions prove conditional retention; they do not prove convergence or practical visibility of the actual training algorithm.

## Mechanical checks

All labels in current `main.tex` are unique. Each of the three requested labels occurs exactly once. Bibliography keys resolve to the intended van Erven–Harremoës, Howard et al., and Wang et al. papers. The uniform-16 illustrative inverse-KL number was independently recomputed below.

Numerical check: lower inverse = 0.033971207627; termwise floor = 4.61948073634e-20. Both match the manuscript rounding.
