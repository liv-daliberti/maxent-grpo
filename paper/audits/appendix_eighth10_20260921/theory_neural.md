# Independent review: P.3 tail, P.4, and P.5 opening

Reviewed the incoming PDF text on pages 95–98, the connected finite-step proof, and `/tmp/paper-appendix-eighth10-20260921/before.tex`, approximately lines 3560–4058. The source snapshot already incorporates substantial repairs that do not appear in the incoming page extracts. This review therefore distinguishes those completed repairs from any additional recommendation. No manuscript, experiment, generator, numerical table, or figure asset was edited by this reviewer.

**Conclusion:** the reviewed mean, covariance, finite-step, response-weighting, verified-key, and diversity-drift identities are correct under their stated assumptions. The snapshot's qualifications prevent the categorical conclusions from being misread as guarantees for shared neural training. No blocking mathematical correction is required. There are no figure captions within pages 95–98.

## Presentation and scope

The incoming PDF contains manuscript narration that is absent from the source snapshot: “The extension here gives…”, “This subsection replaces…”, “We now connect…”, and statements describing what a stronger objective “does buy.” The revised direct statements are appropriate and should be retained. The revised P.4 introduction states the binary-reward allocation issue, the proportional per-prompt means, and the limits caused by covariance, finite steps, shared prompts, and response weighting. The revised P.5 introduction states concentration and finite-budget visibility directly.

The comparison normalized by `c_a(P)` should remain. The sentence explaining that training does not use this normalization is a substantive scope distinction, not editorial process language: deleting it would suggest that the corollary directly bounds the implemented update. Likewise, retain the explicit exclusions of Adam moment estimates and later PPO epochs. The `v1` citation locators identify the versions supporting theorem/section references and are not manuscript-process narration.

“Conditional solution mode probabilities” and “eventual surviving solution modes” in the revised P.4 text satisfy the requested terminology. Mathematical symbols such as “correct mode” or “incorrect response” retain precise meanings and need not be expanded mechanically.

## P.3 tail: coefficient and group size

The coefficient polynomial and the nonnegative leave-one-out gap criterion are sound. Every Bernstein basis coefficient is positive in the interior, so one positive gap and no negative gaps imply `c_a(P)>0` there. The fixed, finite reward-dependent advantage gives a polynomial coefficient on `[0,1]`; its boundedness and smoothness are sufficient for the already cited categorical-flow argument. Mixed-sign gaps alone are indeed inconclusive.

For MaxRL, the ratio to Dr.GRPO is

`G/(G-1) * sum_{j=0}^{G-2} (1-P)^j`.

This is constant at 2 for `G=2`, and strictly decreasing only for `G>2`. The snapshot repairs the incoming unqualified “decreases” claim correctly. Its endpoint limits are `G` and `G/(G-1)`. The revised caption-like corollary title “MaxRL amplification of the correctness gradient” identifies exactly the quantity that is multiplied, without implying equivalent wall-clock progress or general neural correctness trajectories.

The snapshot also correctly limits the group-size equivalence to positive-coefficient deterministic conditional flows. The mixed-group probability `1-P^G-(1-P)^G` is small near both correctness boundaries, not only near zero. This says nothing about a floor for one particular correct solution mode. Preserve the covariance and sampling caveat.

## P.4 mean and covariance: assumptions are sufficient

Theorem `thm:neural-binary-moments` assumes a common positive support, exchange of differentiation with full-support and correct-event sums, and `E||s||^2<infinity`. Since `0<P<1`, both conditional score second moments are finite. The finite detached advantage is bounded because its domain is finite. Consequently the estimator has a finite second moment and every displayed covariance exists.

Differentiating the event probabilities gives the two conditional means:

`E[s | correct] = v/P`, and `E[s | incorrect] = -v/(1-P)`.

Conditioning on the complete binary reward vector preserves row independence. There are `R` correct conditional score draws and `G-R` incorrect draws. Their conditional mean is `beta_R v`; their covariance is the displayed sum of conditional row covariances divided by `G^2`. The law of total covariance therefore gives exactly

`Var(beta_R) vv^T + G^{-2} E[R a(1,R)^2 Sigma_1 + (G-R) a(0,R)^2 Sigma_0]`.

No cross-row covariance term is missing. Conditioning only on the success count can make the row labels dependent, but the proof deliberately conditions on the full reward vector before taking expectations, which avoids that mistake. No categorical-logit or independent neural-parameter assumption is used.

The preconditioned orbit statement is also properly qualified. A common autonomous `M(theta)` and a positive scalar multiplier change the clock of a uniquely existing vector-field solution. The statement does not require `M` to be positive definite, does not assert correctness improvement for arbitrary `M`, and does not cover optimizer history or objective-dependent preconditioners. The current text states only orbit equivalence where the flows exist uniquely, so it is sound.

## Finite neural step

The bounded Hessian along every update segment is the relevant condition. The assumed segment-domain condition is necessary for an unbounded stochastic score and is already explicit. The Taylor remainder is bounded pointwise by `L_f eta^2 ||g_hat||^2/(2 c_a(P)^2)`. The estimator's finite second moment permits expectation, and

`E||g_hat||^2 = c_a(P)^2 ||v||^2 + tr Cov(g_hat)`.

Thus the stated bound and the sum-of-two-remainders comparison are correct. This is a first-order mean-step comparison, not an equality of expected nonlinear observables. The snapshot's text preserves that distinction. Conditional mode probabilities and PCMD may be used only on regions where the stated smoothness bound holds; a global guarantee as `P` approaches zero would need additional assumptions. The existing “where the stated smoothness bounds hold” condition is sufficient and should remain.

## Shared prompts

The revised fixed prompt-sampling law and integrability assumption are correct additions. A parameter-dependent sampling law would introduce a different issue if one interpreted the sum as a gradient of a prompt-weighted scalar objective. Here the sum is explicitly the aggregate update mean under a fixed law.

The counterexample is exact. At `(u,v)=(0,log 3)`, the two prompt gradients are `(3/16,3/16)` and `(3/16,-3/16)`. Equal prompt weights with Dr.GRPO coefficient `2/3` give `(1/8,0)`. MaxRL coefficients `5/4` and `7/4` give `(9/32,-3/64)`. The derivative of `2 sigma(v)(1-sigma(v))` with respect to `v` is `-3/16`, so the MaxRL diversity derivative is `9/1024`; the Dr.GRPO derivative is zero. Dotting either flow with either correctness gradient is positive. This refutes a shared-prompt trajectory-equivalence claim while preserving the valid per-prompt identity.

## Response-dependent normalization

The stated assumptions suffice for the mean and residual bound. Conditional second moments of `omega` and the score imply integrability of `omega s` by Cauchy–Schwarz. They also make each vector covariance `K_r` finite. No second moment of the product `omega s` is needed for the claims actually made. In particular, separate second moments would not by themselves justify a covariance formula for the weighted estimator; the text makes no such claim.

For each outcome class,

`E[omega s | r] = omega_bar_r E[s | r] + Cov(omega,s | r)`.

Substituting the two score means yields the displayed coefficient and residual exactly. The norm bound follows from vector Cauchy–Schwarz and the triangle inequality. The two residual terms can also cancel; the bound does not incorrectly claim that nonzero within-class covariance always changes the direction.

The snapshot appropriately ties “common response length makes it zero” to inverse-length weighting. Equal *mean* lengths do not imply either constant inverse lengths or vanishing covariance with the score. IPS outcome-frequency weights distinguish identities beyond the binary outcome, and a conditional-uniformity objective lies outside the outcome-binary class as stated.

A small optional clarification, not a necessary repair, is to write:

```tex
inverse response length is the case $\omega(Y)=1/L(Y)$ with $L(Y)>0$.
```

This makes the reciprocal's domain explicit. If the repository's response-length convention already guarantees a positive token count, the current text is sufficient. Another optional sharpening is “a length constant within each outcome class makes the residual zero”; a single common length is already a correct sufficient condition, so there is no need to change the existing wording.

## SetPO influence and verified-key gradient

The cached primary source at `paper/audits/reference_audit_20260905/modern_rl_sources/li2026setpo_full.txt`, Assumption 4.1 and Theorem 4.2, supports the specialization. Its kernel must be measurable, symmetric, and in `[0,1]`; its shaping function must be continuously differentiable, nonincreasing, and have bounded derivative. The equality-of-fixed-key kernel and `g(u)=1-u` satisfy all of these conditions.

The functional is `1-sum q_c^2`. Replacing `q` by `(1-epsilon)q + epsilon e_{V(y)}` differentiates to `2(S_2-q_{V(y)})`. Its mean under the correct-conditional law is zero. Differentiating the normalized key probabilities therefore gives exactly the displayed neural gradient, because the conditional-score correction `-grad log P` multiplies a mean-zero influence.

The fixed verifier and key map are necessary and already explicit. Integrability of the score under the correct-conditional law, plus interchange of differentiation with each key sum, is sufficient because there are finitely many keys. There is no unsupported monotonicity conclusion: the sign along a training update still depends on its alignment with this gradient.

## P.5: exact diversity drift and finite expected steps

The variance/Fisher interpretation is used only in the conditional categorical simplex. It is not a claim that a neural optimizer follows a Fisher natural gradient. Directly from `dq_c/dtau = q_c(q_c-S_2)`,

`dS_2/dtau = 2(sum q_c^3-S_2^2) = 2 V(q)`.

Thus `dD/dtau = -2 V(q)`. The positive interior assumption makes the variance zero exactly at a uniform `q`. The logit correctness gradient has correct coordinates `P(1-P)q_c` and incorrect coordinates `-P(1-P)r_a`, hence its squared norm is `P^2(1-P)^2(S_2+T_2)`. Combining this with the effective-time definition gives the stated derivative of correctness log odds and the matched-correctness slope. In effective time the full correct and incorrect logit vector field contains no coefficient `c`; this establishes equality of the entire trajectory from the same initial logits, not merely equality of an initial derivative. The existing local source-reuse and finite-section audits also record primary-source verification of Harper's potential/fitness-variance theorem. No broader Fisher-geometry assertion is needed.

The finite expected-gradient theorem is valid with any finite positive step sizes whose sum diverges, assuming the other categorical hypotheses. For `s>=0`, `q_c exp(s q_c)` preserves and strengthens coordinate ordering. The derivative-of-squared-norm sum is nonnegative term by term, and strictly positive for nonuniform interior `q`. The proof's use of this ordering is limited to integration from zero to positive `h_k`, as required.

Correct logits increase and incorrect logits decrease. The odds increment lower bound `h_k/m` is valid: Jensen gives `log sum q_c exp(h_k q_c) >= h_k S_2`, while `log sum r_a exp(-h_k r_a)<=0`. If correctness converged to an interior value, continuity and positivity of `c` on the resulting compact range would force `sum h_k=infinity`, contradicting bounded odds. If `sum h_k` were finite after correctness tends to one, every logit would have finite variation and a finite limit, again a contradiction. Thus `sum h_k=infinity` and the positive initial leader gap force all competitor-to-leader ratios to vanish.

The finite-step conclusion does not assert matched-correctness path equivalence across estimators; the snapshot explicitly separates these facts in the paragraph following the theorem. Preserve that distinction.

## Validation provenance and action

This review independently checked the derivations and assumptions above and inspected the existing exact-arithmetic validation implementation in `neural/validate.py`. The recorded 102-check neural report includes 18 advantage/group-size cases, 2,016 enumerated weighted response groups, the two-prompt counterexample, the residual, and the influence gradient. The connected finite-section audit records its independent deterministic identity and binomial checks. These previously completed checks were inspected, not rerun or counted as new checks by this reviewer.

No new experiment, training run, model call, bootstrap, or external-source request was made. No further prose replacement is required beyond retaining the snapshot's already integrated repairs. The only optional additional edit is the explicit positive-length condition above. Root should merge only any selected minimal clarification because live `main.tex` is changing concurrently.
