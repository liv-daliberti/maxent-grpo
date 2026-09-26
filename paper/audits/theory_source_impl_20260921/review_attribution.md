# Integrated theory attribution audit — 2026-09-21

Scope: read-only review of the integrated `paper/main.tex`, compared with `main.before.tex` in this directory. Reviewed source attribution and proof compression, the SetPO specialization, finite-budget coverage, and the optimization-geometry distinction. Root handles compilation and the separate optimization review.

**Verdict: no correctness defect found in the assigned changes.** The imported statements supply precisely the intermediate facts used; manuscript-specific extensions retain their derivations and limitations. No manuscript edits were made by this review.

## 1. Exact MaxRL coefficient: valid direct import

Manuscript: `lem:maxrl-mean`, `eq:maxrl-expected-gradient`, and `eq:maxrl-potential-finite-sum`.

Primary source: [Tajwar et al., arXiv:2602.02710v3, Appendix D, Theorem 5](https://arxiv.org/html/2602.02710v3#A4). This theorem is version-sensitive: it is present in v3, and `tajwar2026maxrl` now points to v3 and includes a version note.

The source's practical estimator is exactly

`1{R>0} [(1/R) sum_i R_i s_i − (1/G) sum_i s_i]`.

The manuscript's `G^{-1} sum_i (G R_i/R − 1) s_i`, set to zero when `R=0`, is the identical random vector. With independent on-policy draws and the score identity, Theorem 5 supplies the coefficient `sum_{j=0}^{G−2}(1−P)^j`. The geometric sum and endpoints are therefore correctly imported, including `c(0)=G−1`, `c(1)=1`. Integrating gives the displayed harmonic pass@k mixture with `k=1,...,G−1`. The distinction from the order-G estimator using an unconditional baseline on all-failure groups is preserved. No convergence claim is imported from the estimator identity.

## 2. Harper variance identity: factor two and residual proof retained

Manuscript: `prop:pcmd-drift`, `eq:pcmd-drift`.

Primary source: [Harper, arXiv:0911.1383v1, Theorems 1–2](https://arxiv.org/html/0911.1383v1#S2.SS4). Theorem 1 identifies the gradient potential and Theorem 2 states that its derivative is the fitness variance.

For fitness `f_c(q)=q_c`, the relevant potential is `U(q)=S2/2`, not `S2`. Thus `dU/dτ=Var_q(q_C)` and `D=1−2U` gives `dD/dτ=−2 Var_q(q_C)`, exactly as printed. The matched-correctness identity needs additional manuscript-specific work; that work remains in the proof: `d logit(P)/dτ=S2+T2`, followed by the coefficient-free full-logit dynamics. Positivity of the coefficient and categorical geometry are still explicit. Neither Harper nor the manuscript implies this sign for arbitrary neural optimizers.

## 3. Neural identities and SetPO specialization: valid

Manuscript: `thm:neural-binary-moments`, `cor:pcmd-setpo-neural`.

The conditional-score primitive is correctly credited to [MaxRL v3, Theorem 1](https://arxiv.org/html/2602.02710v3#S4) and [AVSPO v2, Lemma B.1](https://arxiv.org/html/2605.21125v2). Conditioning on the full reward vector and applying total covariance remain explicit in the proof. Thus the covariance extension is proved locally rather than attributed to either source.

For [SetPO v1, Assumption 4.1 and Theorem 4.2](https://arxiv.org/html/2602.01062v1), the fixed deterministic equality kernel is measurable, symmetric, and bounded in `[0,1]`; `g(u)=1−u` is continuously differentiable and nonincreasing with bounded derivative. On the verifier-conditioned law `Qθ`, its functional is exactly `1−sum_c q_c²`. Its influence is `2(S2−q_{V(y)})`. The conditional score is `sθ(y)−∇log Pθ`, and the influence has mean zero, so the normalization term cancels. The printed neural gradient is therefore correct under the stated differentiation/interchange assumptions. Fixed verifier/key map, positive correctness, and arbitrary shared parameters are handled correctly. The final sentence appropriately denies an optimizer-update sign guarantee.

## 4. Coupon coverage: source domain and boundary extension are sound

Manuscript: `cor:coupon-uniform-coverage`, `eq:coupon-uniform-coverage`.

Primary source: [Anceaume et al., arXiv:1504.03878v1, Theorem 3](https://arxiv.org/html/1504.03878v1#S3). Its interior premises are positive coupon masses with total less than one; the remaining mass is a null coupon. At fixed total non-null mass, equal non-null masses minimize the time to collect any specified number `r` of distinct non-null types in stochastic order.

The reduction is exact: correct-key probabilities are `Pq_c`, failures are the null coupon of probability `1−P`, and `{T_r≤K}` equals `{D_K≥r}`. The inequality direction in the manuscript is correct. Zero key probabilities and `P=1` are valid finite-horizon limits: every event probability is a finite sum of products of the `m+1` category probabilities, hence a polynomial continuous on the closed simplex. For example, use `q^(ε)=(1−ε)q+εu` and, at `P=1`, `P^(ε)=1−ε`. Apply the source theorem in the interior and take `ε↓0`.

Summing tails proves the expected distinct-count comparison. The stronger statement is the displayed stochastic dominance, not merely Jensen's expected-count inequality. It applies to an iid fixed sampling law and does not show that a training algorithm reaches uniformity; the manuscript explicitly says so. No majorization or Shannon-entropy premise is substituted for the coverage theorem.

## 5. Natural-gradient finite update: exact ratio preservation

Manuscript: paragraph “Dependence on optimization geometry.”

Primary source: [Agarwal et al., JMLR 22(98), Lemma 15, p. 21](https://jmlr.org/papers/volume22/19-736/19-736.pdf). Its multiplicative NPG update reduces at bandit discount `γ=0` to `p_a^+ ∝ p_a exp(η A_a)`. For the scaled correctness gradient, `A_a=c_G(P)(R_a−P)`; the common baseline exponential cancels in normalization, giving exactly the manuscript's `p_a^+ ∝ p_a exp(η c_G(P)R_a)`. Every correct category receives the same multiplier, so all correct ratios are preserved at every finite step, even when the scalar coefficient changes across steps.

The continuous logit representation is valid up to the usual common-logit gauge. This is a statement about exact categorical NPG, not sampled updates or an arbitrary neural Fisher approximation. The accompanying [Cui et al. v1, Theorems 1–2](https://arxiv.org/html/2505.22617v1) attribution supports the distinction between vanilla and natural-gradient entropy changes without claiming entropy establishes verified-mode behavior.

## 6. Version precision and proof preservation

The bibliography pins the audited arXiv sources: MaxRL v3, SetPO v1, AVSPO v2, DPH v4, Cui v1, Harper v1, and Anceaume v1. Consequently inline citations need not repeat every version. DPH is used only for its valid Appendix D.1 identity `R_w=H(w)+KL(w||p)`; its disputed monotonic-improvement result is not imported. The manuscript-specific categorical replay and neural loss bounds retain their separate premises.

A structural comparison against the saved `main.before.tex` found:

- No pre-existing theorem, lemma, proposition, or corollary statement removed or changed.
- No pre-existing proof environment lost. Total proof environments increased from 29 to 34.
- Exactly five new formal statements: sharp KL certificate, Dr.GRPO reference-KL convergence, SetPO metric bridge, coupon coverage, and stochastic energy-budget retention.
- Exactly three old proof bodies changed: MaxRL's elementary expectation derivation was replaced by the exact source theorem application; PCMD's variance calculation was replaced by Harper with the correct potential scaling; the neural mean/covariance proof added source credit for the conditional-score identity. The remaining parts of those proofs are retained.
- The original finite-step collapse, sampled-PCMD expansion, finite-budget invisibility, replay thresholds and recovery, and KL sublevel arguments are not removed by this integration.

The reader can distinguish imported foundations from the paper-specific consequences. No further correction is required for the points covered here.
