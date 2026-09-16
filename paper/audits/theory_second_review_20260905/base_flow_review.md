# Independent second review: categorical and shared-correctness flows

Reviewed 5 September 2026. Scope: `paper/main.tex:877–1230`, with the corresponding workshop appendix. No TeX edits. The two proof sections differ only in two document cross-references; their mathematics is identical. Base-section SHA256 at review: `fe822cecb1f2ec5af6daf055bc94ec658170af789a9c28fd5a5d47d3e2280156`.

**Verdict:** the stated finite-group expectations, winner-take-all theorem, tied-maxima limit, and matched-accuracy shared-correctness theorem are correct under the stipulated model. I found no incorrect theorem conclusion. The main remaining issue is interpretive: “becoming more right eventually collapses” is true here for a fixed finite shared-correctness coefficient **with the independent-logit component retained**, a unique initial maximum, and unlimited exact mean-flow optimization. It is not a consequence of increasing correctness alone.

## 1. Mean updates and endpoint bookkeeping

At `main.tex:886–941`, the conditional score calculation is correct. With mixed-group weights `w(ℓ)>0`, the coefficient can also be written without endpoint division:

\[
c_G(P)=\frac{G-1}{G}\sum_{j=0}^{G-2}\binom{G-2}{j}w(j+1)P^j(1-P)^{G-2-j}.
\]

The identity follows from `binom(G,ℓ)ℓ(G−ℓ)=G(G−1)binom(G−2,ℓ−1)`. It gives the useful explicit bounds

\[
\frac{G-1}{G}\min_{1\le\ell<G}w(\ell)
\le c_G(P)\le
\frac{G-1}{G}\max_{1\le\ell<G}w(\ell),
\]

on the closed interval `[0,1]`, and endpoint values `(G−1)w(1)/G` and `(G−1)w(G−1)/G`. Thus the scalar field is continuous, positive and bounded, and the finite-logit ODE exists for all finite times. This also makes the later continuity argument completely explicit.

**Small notation fix, not a result error:** `main.tex:911–912` writes `E[w(M)M(G−M)]` although the text avoids defining reciprocal-standard-deviation weights on all-equal groups. State that the product is defined as zero at `M=0,G`, or display the finite sum above. Otherwise the expectation can look like an undefined zero-times-infinity expression despite the correct zero-update convention at lines 900–901.

Dr.GRPO gives `(G−1)/G`. For MaxRL, `main.tex:947–999` correctly distinguishes the implemented centered estimator:

\[
c_G^{\rm MaxRL}(P)=\sum_{j=0}^{G-2}(1-P)^j,
\qquad
\Psi(P)=\sum_{k=1}^{G-1}\frac{1-(1-P)^k}{k}.
\]

The all-failure update is genuinely consequential. Applying the score baseline on that event adds `(1−P)^{G−1}∇P`, giving the order-`G` rather than order-`G−1` potential. The current source attribution and distinction are appropriate. At `G=2`, the implemented MaxRL coefficient is exactly one; none of the formulas requires `G>2`.

The scope statements are essential: categorical execution-mode logits, fixed common length normalization, stop-gradient sample advantages, independent fresh groups, the on-policy ratio-one derivative, zero reference KL, and exact Euclidean mean flow. A response-dependent normalizer, later PPO epochs/clipping, Adam, finite steps or shared-prompt interference does not automatically inherit this ODE. The paper already excludes most of these clearly.

## 2. Collapse, effective time, and ties

The argument at `main.tex:1038–1081` is sound. Writing `r_a=p_a/(1−P)`,

\[
\|\nabla P\|^2=P^2(1-P)^2(S_q+S_r)
\ge P^2(1-P)^2(1/m+1/n).
\]

This rules out an interior correctness limit. Every logit velocity is bounded by `dτ/dt=c_G(P)P(1−P)`, so finite total `τ` would give finite limiting logits and contradict `P→1`. Consequently `τ→∞`; the proof does not silently assume that vanishing advantages leave infinite effective time.

On that clock, `q'_c=q_c(q_c−S_q)`. Ordering is invariant by the log-ratio equation and uniqueness of the smooth flow. For a unique initial maximizer `*`,

\[
\xi_c=q_c/q_*,\quad
(\log\xi_c)'=-q_*(1-\xi_c)
\le-[1-\xi_c(0)]/m.
\]

The displayed exponential bound is in **effective time**, not homogeneous wall-clock/optimizer time. It proves every minority vanishes. Finite logits mean that probabilities remain positive at every finite time: “eliminating” modes denotes the limiting distribution, not finite-time exact zeros.

At `main.tex:1086–1091`, tied maxima persist by symmetry, while every strictly smaller coordinate vanishes by the same ratio argument. The equal maximum coordinates therefore each tend to `1/k`. The sentence “only the exactly uniform initialization ... avoids collapse” is potentially confusing when “collapse” is read as *single*-mode collapse. Suggested replacement:

> Only the exactly uniform initialization preserves all `m` modes in the infinite-time limit. A tie among `1<k<m` maxima yields partial collapse onto those `k` modes; a unique maximum yields single-mode collapse.

The finite-logit assumptions exclude starting at `P=0` or `P=1`. At the correctness boundary `P=1`, all groups are all-correct and the unreplayed task update is zero; arbitrary correct-mode ratios are then stationary. This is another reason not to state the theorem as a claim about every already-perfect policy.

## 3. Shared correctness and sampled breadth

The chain rule at `main.tex:1111–1125` is correct for Euclidean parameter gradient flow: `Kθ=JθJθᵀ`. It does not require a derivative of `Jθ` in the first time derivative of logits. The redundant parameter example gives `I+λvvᵀ` exactly; parameter duplication/scaling at lines 1207–1216 is also correct.

For constant finite `λ≥0`, all three equations at `main.tex:1164–1173` check independently:

\[
q'_c=q_c(q_c-S_q),\quad
r'_a=-r_a(r_a-S_r),\quad
(\operatorname{logit}P)'=S_q+S_r+\lambda.
\]

Thus `q(τ),r(τ)` are independent of both `λ` and the choice of positive scalar `c_G`. The positive interior target is reached in finite physical time. At fixed target `P_*`, the implicit clock equation gives strict decrease of `τλ` with `λ`; the retention bound follows from `d log q_c/dτ≥−1`. The covariance expression at lines 1190–1200 is correct. Strictness needs `K≥2`, nonuniform `q(0)`, and an interior target—all stated. `K=1` gives exactly `P_*` for both sampling metrics. These are comparisons of **expected** independent-draw counts, not sample-by-sample inequalities.

A useful immediate implication is that, within this reduction and fixed geometry, GRPO/Dr.GRPO/MaxRL have the same entire `q(P)` path: their positive coefficients change the optimization clock. Their observed neural differences cannot be explained by a categorical matched-accuracy direction difference alone.

Raw `distinct@K` along training need not be monotone, because increasing `P` competes with concentration of `q`. Indeed, for `K=2`,

\[
D_2=2P-P^2S_q,
\quad
D'_2=2(1-PS_q)P'-P^2S'_q.
\]

At sufficiently small `P`, the positive correctness term dominates for any fixed interior `q,r,λ`. Therefore avoid replacing the precise matched-accuracy theorem by “breadth must decrease whenever accuracy rises.”

## 4. Recommended compact new corollary: rate versus residual error

This quantifies the existing eventual-collapse statement without introducing a new modeling assumption or claiming a physical-time exponential rate.

**Proposed statement.** Under Theorem `thm:shared-correctness`, fix finite `λ≥0` and suppose the initial maximum is unique. For every other correct mode `c`,

\[
\boxed{\lim_{P\uparrow1}\frac{\log q_c^{(\lambda)}(P)}{\log(1-P)}
=\frac{1}{\lambda+1+1/n}.}
\]

Equivalently, `q_c=(1−P)^{1/(λ+1+1/n)+o(1)}`. Larger fixed `λ` decreases the decay exponent but every minority mode still vanishes. This is a logarithmic asymptotic relation, not a claim of an exact power law at finite accuracy.

**Proposed short proof.** The existing theorem gives `q→e_*`, hence `S_q→1`. The incorrect conditional flow tends to the uniform distribution. To see this, its smallest coordinate is nondecreasing, since `S_r≥r_min`, and remains at least `r_min(0)>0`. Ordering is preserved. Put `h=log(r_max/r_min)`. Then

\[
h'=-(r_{\max}-r_{\min})
=-r_{\min}(e^h-1)\le-r_{\min}(0)h.
\]

Thus `h→0` and normalization gives `r→(1/n,...,1/n)`; the case `n=1` is immediate. Therefore

\[
\frac{\log q_c(\tau)}{\tau}\to-1,
\qquad
\frac{\operatorname{logit}P(\tau)}{\tau}\to\lambda+1+1/n,
\]

by integrating `d log q_c/dτ=q_c−S_q` and the correctness-clock equation and taking time averages. Since `log(1−P)=log P−logit P` and `log P→0`, division proves the result.

If a finite-accuracy bound is preferred, the existing ratio proof and `L≤(λ+2)τλ` immediately yield

\[
\frac{q_c(P_*)}{q_*(P_*)}
\le \xi_c(0)\exp\!\left[-\frac{[1-\xi_c(0)]L}{m(\lambda+2)}\right].
\]

This bound and the asymptotic exponent should not be confused: the former is global but conservative; the latter is sharp only in logarithmic asymptotics.

## 5. Critical scope paragraph: shared learning alone does not force collapse

Suggested paragraph following the corollary:

> The independent-logit component is essential to this conclusion. With purely shared geometry `K=bvvᵀ`, `b>0`, all correct logits move equally and all incorrect logits remain fixed. Hence `q(t)=q(0)` while `Pdot=b c_G(P)P²(1−P)²>0` and `P→1`. More generally, `K=aI+bvvᵀ` with fixed `a>0` reduces to the theorem with `λ=b/a` after rescaling time. Increasing correctness therefore does not itself imply collapse: collapse here comes from persistent mode-specific amplification alongside the shared correctness update.

At fixed interior target, taking `λ→∞` gives `τλ→0` and `qλ(P_*)→q(0)`. Taking `P_*→1` first at each fixed finite `λ` instead gives the unique winner. These limits do not commute. A time-varying or unbounded geometry is outside the constant-`λ` result; a finite total learning-rate budget likewise need not produce infinite effective time.

## 6. Optional explanation using majorization; no extra theorem needed

Ordering the correct probabilities decreasingly, the sum of the largest `j` obeys

\[
\frac{d}{d\tau}\sum_{i\le j}q_i
=\sum_{i\le j,\ell>j}q_iq_\ell(q_i-q_\ell)\ge0.
\]

Thus later `q` majorizes earlier `q`. Larger `λ` at matched correctness therefore preserves every symmetric concave diversity measure, including conditional Shannon entropy, and not just the particular distinct-count functional. This can explain the existing covariance calculation in one sentence; an additional standalone theorem would add little. These are standard replicator/majorization consequences, with the paper-specific contribution being their connection to the group estimators, correctness clock, and execution-defined measurement.

Supporting prior checks: [ordered-group enumeration](../verify_categorical_group_expectations_20260905.py) and [results](../verify_categorical_group_expectations_20260905.json) cover 80 cases with maximum absolute error `9.33e−15`; [shared-geometry verifier](../verify_model_size_theorem_20260904.py) and [results](../verify_model_size_theorem_20260904.json) check finite target comparisons. These computations support arithmetic and interpretation; the arguments above establish the claims. No implication for stochastic clipped neural training, causal effects of model size, or execution-key completeness is asserted.
