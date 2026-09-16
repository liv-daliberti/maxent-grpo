# Optimizer-aware retention: what can actually be proved

Independent derivation and primary-source audit, 5 September 2026. No manuscript, optimizer, runtime, or scheduler changes. The proposed standalone fragment is [optimizer.tex](optimizer.tex). It was independently checked by the replay-proof reviewer; the drift and supermartingale argument passed.

**Answer:** yes, a useful theoretical extension beyond exact categorical gradient descent is available. A fixed replay loss can protect complete response events under noisy, biased, preconditioned updates if their accumulated upward energy budget is controlled. This permits shared neural parameters and does not require monotone energy on every sample path. It does **not** establish that the implemented AdamW/PPO/round-robin process satisfies that budget. Existing optimization literature supplies the tools and nearby policy-gradient guarantees, rather than a ready-made theorem for this particular banked-response system.

## 1. Fixed objective and the one-step inequality

Fix finitely many protected complete-response exemplars and define

\[
R(\theta)=\sum_j a_j[-\log\pi_\theta(e_j\mid x_j)],\quad
 a_j=\alpha_j/L_j>0,\quad F=R-J,\quad J\le J_{\max}<\infty.
\]

The complete response must include EOS or a deterministic horizon and use the same temperature, vocabulary mask and truncation law as the bounded event. Then `p(key b_j|x_j)≥π(e_j|x_j)`. Shared parameters and multiple prompts are allowed. Define `D(θ)=F(θ)+Jmax≥R(θ)≥0`.

Assume `F` has globally `L_F`-Lipschitz gradient, and consider

\[
\theta_{t+1}=\theta_t-\eta_tH_t(g_t+b_t+\xi_t),\quad g_t=\nabla F(\theta_t).
\]

The filtration contains history before drawing `ξ_t`; `H_t,b_t,η_t,θ_t` are predictable. Assume `0≼H_t≼MI`, `η_t L_F M≤1`, and conditionally centered, finite-variance `ξ_t`. This also allows singular positive semidefinite matrices: a lower eigenvalue bound is unnecessary for retention, which is weaker than optimization progress.

The exact drift estimate is

\[
\mathbb E_tF_{t+1}\le F_t-\frac{\eta_t}{2}\|g_t\|_{H_t}^2
 +\frac{\eta_t}{2}\|b_t\|_{H_t}^2
 +\frac{L_F\eta_t^2}{2}\mathbb E_t\|H_t\xi_t\|^2. \tag{1}
\]

**Derivation.** Apply smoothness to the complete step and take conditional expectations. Centering removes both terms linear in `ξ_t`. Since `H_t²≼MH_t`, the deterministic quadratic term is at most `η_t||g_t+b_t||²_H/2`. Consequently

\[
-\eta_tg_t^TH_t(g_t+b_t)+\frac{\eta_t}{2}\|g_t+b_t\|_H^2
=-\frac{\eta_t}{2}\|g_t\|_H^2+\frac{\eta_t}{2}\|b_t\|_H^2.
\]

No convexity of `F`, independent-logit parameterization, bounded parameter domain, or lower spectral bound is used. Global smoothness can be replaced by a bound on every actual update segment, but a local bound valid only before exiting a region proves only a stopped-process result until exit; one cannot assume the desired probability floor to prove that exit never occurs.

The structure closely follows the stochastic variable-metric estimate in [Wang et al., Section 2, assumptions AS.1–4 and Lemma 2.1](https://optimization-online.org/wp-content/uploads/2016/07/5529.pdf). Their predictable-matrix requirement is explicit. Their lower spectral bounds and non-summable steps serve stationarity results; they are not needed for the probability consequence derived here.

## 2. Simultaneous high-probability retention, finite or infinite horizon

Let deterministic `d_t≥0` bound the two positive terms in (1), and let `B_N=Σ_{t<N}d_t<∞`, for a chosen finite horizon or `N=∞`. Fix finite initial energy `D0=F0+Jmax`. The process

\[
X_t=D_t+\sum_{s=t}^{N-1}d_s
\]

is adapted, nonnegative, integrable and a supermartingale. Its initial value is `D0+B_N`. Nonnegativity supplies integrability inductively from the finite drift bound. The same construction uses the convergent deterministic tail for an infinite horizon. The nonnegative-supermartingale maximal inequality gives

\[
\Pr\!\left(\sup_{t\le N}X_t\ge(D_0+B_N)/\delta\right)\le\delta.
\]

The degenerate zero-initial-energy case has `X_t=0` almost surely. On the complementary event, all nonnegative replay summands are bounded simultaneously, so

\[
\boxed{\forall t\le N,\ \forall j:\quad
p_{\theta_t}(b_j\mid x_j)\ge\pi_{\theta_t}(e_j\mid x_j)
\ge\exp\!\left[-\frac{L_j(D_0+B_N)}{\alpha_j\delta}\right].} \tag{2}
\]

The event has probability at least `1−δ`. There is **no extra union factor in the number of bank entries**, since one scalar energy bound controls every summand. This is a trajectory guarantee, stronger than a bound on expected likelihood or expected loss at each fixed step.

For `N=∞`, the maximal inequality with threshold tending to infinity gives `sup_t X_t<∞` almost surely. Thus each protected probability has an almost surely positive random infimum over all iterations. This does not promise a useful deterministic floor at confidence one, convergence of parameters, or an optimal policy. The almost-supermartingale tradition is [Robbins–Siegmund (1971)](https://www.sciencedirect.com/science/article/pii/B9780126045505500158); the exact maximal tool used above is stated as Ville's inequality in [Howard et al., Lemma 1](https://arxiv.org/html/1808.03204v5#S2.SS3). These are classical ingredients, not a new stochastic-convergence method.

With zero bias and uniformly bounded variance, `Ση_t²<∞` suffices for a finite infinite-horizon budget. To additionally demand continuing optimization one often also asks `Ση_t=∞`, but retention alone does not need it. Predictable bias contributes `Ση_t||b_t||²_H`; a persistent nonzero bias generally fails this test with non-summable steps. Constant steps and persistent noise give `B_N=O(Nη²)`, hence a finite-horizon guarantee whose floor may decrease with `N`. This argument does not prove infinite-horizon retention in that regime, and its failure is not a proof of extinction.

A simpler deterministic inexact version follows from (1) before expectation: for any realized error `e_t`,

\[
F_{t+1}\le F_t-\eta_t\|g_t\|_H^2/2+\eta_t\|e_t\|_H^2/2.
\]

A finite pathwise error budget directly gives the original deterministic energy floor with that budget added. This accommodates arbitrary update errors but charges noise at order `η` rather than the sharper `η²` available through conditional centering.

## 3. A sharper finite-horizon option, with stronger assumptions

The `1/δ` dependence in (2) is weak. For unbiased updates, assume deterministic bounds `||g_t||≤G_t`, `||ξ_t||≤s_t` almost surely and deterministic `η_t`. Smoothness gives, pathwise,

\[
F_{t+1}-F_t\le-\eta_t\|g_t\|_H^2/2+Z_{t+1}
 +L_F\eta_t^2M^2s_t^2/2,
\]

where

\[
Z_{t+1}=-\eta_tg_t^TH_t(I-L_F\eta_tH_t)\xi_t.
\]

Predictability makes `Z` a martingale difference. Since `0≼I−L_Fη_tH_t≼I`, `|Z_{t+1}|≤c_t:=η_tMG_ts_t`. Conditional Hoeffding's lemma makes `exp(uΣZ−u²Σc_t²/2)` a nonnegative supermartingale. Applying its maximal inequality and optimizing `u` yields, for `V_N=Σ_{t<N}c_t²>0`,

\[
\Pr\left(\max_{t\le N}\sum_{s<t}Z_{s+1}
>\sqrt{2V_N\log(1/\delta)}\right)\le\delta.
\]

For `V_N=0`, the martingale is identically zero. Therefore the complete energy factor `(D_0+B_N)/delta` in (2) can be replaced by

\[
D_0+\frac{L_FM^2}{2}\sum_{t<N}\eta_t^2s_t^2
 +\sqrt{2V_N\log(1/\delta)}. \tag{3}
\]

This has a substantially better confidence dependence, but needs almost-sure gradient/noise bounds. It is an optional companion result, not necessary for the core theorem. The derivation is the standard exponential-supermartingale construction developed systematically by [Howard et al.](https://doi.org/10.1214/18-PS321). It does not follow from bounded variance alone.

## 4. What the actual source establishes, and what remains unproved

Read-only implementation evidence:

- `src/oat_drgrpo/fused_adam_shim.py:23–54` returns `torch.optim.AdamW` with moment parameters, epsilon, optional weight decay, and `amsgrad` forwarded. The factory defaults do not prove the numerical settings for every frozen run.
- `src/oat_drgrpo/learner/grpo.py:487–527` temperature-scales scores and computes the clipped PPO maximum of unclipped/clipped losses unless the separate REINFORCE switch is enabled. A derivative at the ratio-one point does not certify later clipped updates as an unbiased gradient of one fixed `F`.
- `src/oat_drgrpo/learner/grpo.py:247–287` evaluates response-mask mean token log likelihood; `:1245–1280` applies the detached replay score-gradient coefficients in chunks before the optimizer step at `:1855`. The verified-likelihood coefficients are exact for the selected bank, as shown in `src/oat_drgrpo/canonical_replay.py:322–421`.
- The global round-robin path appears at `src/oat_drgrpo/learner/grpo.py:3161–3356`, with admission and scheduling in `src/oat_drgrpo/online_canonical_bank.py`. A per-update logged selected-bank loss is not the full fixed-bank energy used in the proof.

**Must not infer:** Adam's second moment uses the current noisy gradient, so its matrix is not predictable relative to that noise. Momentum changes the direction as well. Gradient/PPO clipping, old-policy ratios, admission, changing replay weights and deterministic cyclic selection produce biases or changing objectives. They cannot be declared mean-zero simply by averaging over the fresh group. Grouping a full scheduling cycle may support a separate incremental-gradient error analysis, but cancellation of first-order cycle errors is not yet a bound for this implementation. Decoupled weight decay, if nonzero, also needs to be included in the objective or error term.

The global neural smoothness constant and the requisite moment/error budgets are not measured. A small nominal learning rate does not verify `η L_F M≤1`, and observed low loss on selected banks does not establish a uniform global energy ceiling. Sparse checkpoint evidence alone cannot exclude excursions between checkpoints.

[Reddi, Kale and Kumar, Theorems 1–3](https://arxiv.org/html/1904.09237v1#S3) give explicit online and stochastic convex counterexamples to unrestricted Adam convergence. Their AMSGrad guarantees require their own assumptions and are not per-exemplar retention theorems. The appropriate conclusion is that additional optimizer-specific hypotheses are necessary, **not** that the current AdamW runs must fail. A simple current-noise-dependent preconditioner counterexample is included in our numerical audit: with `F(θ)=θ²/2`, `θ=1`, centered noise `±2`, choose `H=10` for `−2` and `H=1` for `+2`. At `η=.05`, the claimed predictable drift bound would be `.615`, but the exact expected next energy is `.743125`. This makes the predictability requirement concrete.

## 5. Direct policy-gradient literature: useful precedent, different guarantee

[Zhang, Kim, O'Donoghue and Boyd (AAAI 2021)](https://ojs.aaai.org/index.php/AAAI/article/view/17300) analyze fixed-minibatch REINFORCE with tabular softmax and a log barrier over state-action probabilities. The full primary paper specifies a finite discounted MDP, positive initial-state probabilities, phased regularization/step schedules and post-processing that mixes each phase's policy with a uniform component. Theorem 6 bounds regret and gives almost-sure convergence of average regret; regularization decreases across phases. This is stronger practical precedent than exact-gradient-only results, but it does not establish a uniform all-time probability floor for selected completed responses under a fixed replay bank. It is also not Adam, PPO clipping, arbitrary neural parameter sharing, or an execution-key admission theorem. See the [author-hosted full paper, Algorithms 2–4, Assumptions 1–2, Theorem 6 and Corollary 11](https://stanford.edu/~boyd/papers/pdf/conv_reinforce_aaai_full.pdf).

## 6. Usefulness, validation, and recommended next step

The strongest defensible next theoretical claim is the conditional theorem (2), with finite and infinite horizons separated. It removes the unnecessary demand for sample-path monotonicity and independent logits, while making stochasticity, preconditioning and bias explicit. Its numerical floor can still be useless: it scales exponentially with response length and inverse replay weight, and the basic confidence bound multiplies energy by `1/δ`. Even (3) does not remove the length/weight issue. These are qualitative protection results, not guarantees of seeing every banked mode in eight evaluation draws.

A concrete empirical bridge would freeze a finite bank after admission, record complete response log probabilities under the evaluation sampling law, and measure a fixed full-bank loss at repeated checkpoints. To certify an optimizer guarantee, also establish the intervening energy/error budget or introduce a loss-based acceptance condition for candidate steps. Such safeguards would be new algorithms and experiments, not facts about the existing run. No change to training is proposed or performed here.

[verify_optimizer_drift.py](verify_optimizer_drift.py) and [results](verify_optimizer_drift.json) exhaustively average four finite noise outcomes for 216 cases across Dr.GRPO/implemented MaxRL potentials, four policies, dense/singular PSD matrices, zero/fixed/adversarial bias, and three step fractions including the allowed maximum. Every drift inequality passes; minimum slack is `3.710e−4`. Independent finite differences check the gradient to `2.85e−11`. The explicit nonpredictable-matrix witness fails the would-be extension as expected. These are finite arithmetic checks, not a Monte Carlo proof or certification of neural training.

## Verified source metadata

Machine-readable entries: [optimizer.bib](optimizer.bib). Exact theorem locations and scope above were checked in primary texts. Robbins–Siegmund's original metadata and publisher summary were verified; the full original/reprint body remained access-limited, so the proof here uses the independently stated and proved special case rather than claiming a full-text audit of that chapter.

1. Herbert Robbins and David Siegmund. **A Convergence Theorem for Non Negative Almost Supermartingales and Some Applications.** In *Optimizing Methods in Statistics*, pp. 233–257, Academic Press, 1971. DOI [10.1016/B978-0-12-604550-5.50015-8](https://doi.org/10.1016/B978-0-12-604550-5.50015-8). Do not substitute the 1985 selected-papers reprint's date/pages.
2. Xiao Wang, Shiqian Ma, Donald Goldfarb and Wei Liu. **Stochastic Quasi-Newton Methods for Nonconvex Stochastic Optimization.** *SIAM Journal on Optimization* 27(2):927–956, 2017. DOI [10.1137/15M1053141](https://epubs.siam.org/doi/10.1137/15M1053141); arXiv:1607.01231. The earlier 1412.1196 version has a different three-author list; use the four-author published version.
3. Steven R. Howard, Aaditya Ramdas, Jon McAuliffe and Jasjeet Sekhon. **Time-uniform Chernoff bounds via nonnegative supermartingales.** *Probability Surveys* 17:257–317, 2020. DOI [10.1214/18-PS321](https://doi.org/10.1214/18-PS321); arXiv:1808.03204. Some accessible preprint renderings carry the earlier title *Exponential line-crossing inequalities*; use the published title for the journal citation.
4. Sashank J. Reddi, Satyen Kale and Sanjiv Kumar. **On the Convergence of Adam and Beyond.** ICLR 2018; [OpenReview ryQu7f-RZ](https://openreview.net/forum?id=ryQu7f-RZ); arXiv:1904.09237. The arXiv record explicitly confirms ICLR 2018 despite its 2019 deposit.
5. Junzi Zhang, Jongho Kim, Brendan O'Donoghue and Stephen Boyd. **Sample Efficient Reinforcement Learning with REINFORCE.** *Proceedings of the AAAI Conference on Artificial Intelligence* 35(12):10887–10895, 2021. DOI [10.1609/aaai.v35i12.17300](https://ojs.aaai.org/index.php/AAAI/article/view/17300).
