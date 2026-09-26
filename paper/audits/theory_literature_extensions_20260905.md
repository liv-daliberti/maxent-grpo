# Theory and literature extensions for verified replay — 2026-09-05

**Yes: all four questions yield useful theory, although theoretical certificates cannot replace measurements of actual per-mode survival.** The strongest additions are controlled stochastic retention, admission followed by retention under explicit coverage assumptions, and sharp probability bounds from normalized replay cross entropy.

The [11-page supplement](theory_literature_extensions_20260905/theory_extensions.pdf) contains complete proofs and linked attributions. Its [TeX](theory_literature_extensions_20260905/theory_extensions.tex) and [source bundle](theory_literature_extensions_20260905/theory_extensions_source.zip) are independently reviewable. This investigation produced a standalone note; both reviewed manuscripts remain unchanged, and no new training measurements were generated.

| Reviewer question | Available result | Remaining application premise |
|---|---|---|
| Optimizer-aware retention | Simultaneous high-probability floors for all protected exemplars throughout a finite horizon; infinite-horizon retention under summable noise/bias budgets | Fixed joint potential, smoothness, controlled update error and predictable preconditioning; these are not established for current AdamW/PPO |
| Discovery/admission | Adaptive admission bounds plus retention across bank changes; together, conditional eventual coverage and retention | Actual insertion opportunities, sufficient capacity, persistent weights and a finite switching-energy budget |
| Exact entropy | Exact conditional entropy and full-support uniform replay share an ideal optimum, with different KL geometry; published finite-step exact-entropy theory applies to the linear categorical comparator | Explicit output alphabet and exact estimator; no transfer to the historical semantic score |
| Per-mode survival | Sharp probability certificates and valid finite-count intervals | Complete normalized likelihoods or named-key samples from specified checkpoints; observations still required |

## Optimizer-aware retention from stochastic approximation

For the joint potential `F = weighted exemplar negative log likelihood − bounded fresh objective`, the proved model permits a history-measurable positive-semidefinite preconditioner, conditional-zero-mean noise, and controlled bias. Smoothness gives

`E[F_{t+1} | history] <= F_t − (eta_t/2)||grad F_t||²_H + d_t`.

A finite accumulated drift budget and a nonnegative-supermartingale maximal inequality bound energy across all times, hence every nonnegative exemplar-loss summand. This avoids a union penalty over bank size. Constant steps with persistent variance give the stated finite-horizon result; infinite-horizon floors require additional control. The detailed note also derives a sharper confidence factor under bounded gradients and noise.

The ingredients come from [stochastic variable-metric optimization, Wang et al. (2017)](https://doi.org/10.1137/15M1053141) and [supermartingale inequalities, Howard et al. (2020)](https://doi.org/10.1214/18-PS321). Robbins–Siegmund provides older convergence background. A direct RL precedent, [Zhang et al.'s REINFORCE analysis](https://stanford.edu/~boyd/papers/conv_reinforce.html), uses a tabular, phased, uniformly mixed algorithm and proves a different regret result.

The [optimizer derivation](theory_literature_extensions_20260905/optimizer.md) maps the assumptions to current code. An Adam matrix constructed from the current gradient's noise is not predictable relative to that noise. A verified counterexample shows why dropping predictability invalidates the displayed drift bound. This is an application of established tools, not an existing theorem certifying our learner.

## Discovery must count actual admission

Let `tau_b` be the first insertion of a verified key. If its conditional admission probability is at least `h_b` whenever it remains missing, then

`Pr(tau_b > N) <= exp(−N h_b)`.

For m target keys with `h_b>=h_min`, the probability that some key remains unadmitted is at most `m exp(−N h_min)`. Adaptive policies are allowed. The supplement also proves a stopped-hazard statement for random predictable probabilities and a bound charging positive energy jumps when banks change.

Together these give a conditional discovery-to-retention result: finite correct support fitting within capacity, non-evicting admission, continuing positive admission floors, persistent positive coefficients, and bounded switching energy imply eventual full coverage followed by retention. A fixed full-support proposal mixture is sufficient when it is queried and insertion is guaranteed. Current temperature/priority machinery does not establish that premise.

Positive chances alone are insufficient: independent admission probabilities `h_n=1/(n+1)^2` leave probability **one half** of never admitting the key. A full non-evicting bank can make admission probability zero.

The main probability source is [Durrett's conditional Borel–Cantelli theorem](https://sites.math.duke.edu/~rtd/PTE/PTE5_011119.pdf), Theorem 4.3.4. Coupon collection describes the frozen iid special case. [Go-Explore](https://www.nature.com/articles/s41586-020-03157-9) supplies archive/discovery precedent, and switched-system Lyapunov analysis motivates charging bank changes. The [discovery note](theory_literature_extensions_20260905/discovery.md) verifies sources and details the actual implementation gates.

## Exact entropy gives a fairer comparison

On the full correct support, conditional entropy minimizes `KL(q||u)`, while uniform replay has loss `log m−log P+KL(u||q)`. Both exact objectives have the global optimum `P=1,q=u` when the correctness potential strictly increases. **Conditional entropy has no inherent optimum-level accuracy penalty.** Full-output entropy has a different Gibbs optimum that retains incorrect mass. This is a global-optimizer comparison, not a new convergence theorem for conditional-entropy flow.

The KL-direction distinction is established in [Pereyra et al.'s confidence-penalty/label-smoothing paper](https://arxiv.org/abs/1701.06548). [Agarwal et al., Section 5.2](https://jmlr.org/papers/v22/19-736.html) analyze log-barrier policy regularization. [Mei et al., Lemma 13 and Theorem 5](https://proceedings.mlr.press/v119/mei20b.html) prove positive probability floors and geometric convergence for exact tabular entropy updates. These sources strengthen positioning while ruling out universal “MaxEnt inevitably collapses” language.

An additional result comes from [Fisher/Shahshahani information geometry](https://arxiv.org/abs/0911.1383): exact categorical natural-gradient correctness learning preserves conditional correct-mode ratios. With replay,

`q_dot=(rho/P)(w−q)`.

The distribution contracts toward the bank target, and each banked probability stays at least `P(0) min(q_b(0),w_b)`. This alternate-optimizer calculation is useful as an optional mechanism control. It does not cover Adam or approximate neural natural gradient.

## Sharper per-mode certificates and measurement limits

If normalized bank cross entropy satisfies `R_w(p)<=C`, let `D=C−H(w)`. [KL data processing](https://arxiv.org/abs/1206.2459v2) gives

`kl(w_b||p_b)<=D`.

Its lower root is a sharp positive probability floor. For a hypothetical uniform 16-mode bank with `C=log16+.01`, this gives **0.0339712**, versus **4.62e−20** from the previous termwise bound. These inputs are illustrative, not measured training losses. Large energy budgets can still make even the sharp bound uninformative.

The entropy-only threshold is sharp too: a scalar lower bound on exact `H(q)` excludes every missing mode only if it exceeds `log(m−1)`. With 16 modes, one may be absent while entropy remains `log15`, about **97.7% of maximum**. This describes level sets; exact-entropy dynamics can still prevent extinction.

Zero sightings in 32 draws permits probability as large as **8.94%** at a one-sided 95% level. It does not establish extinction. Named-key counts require a frozen policy; repeated inspection can use a [confidence sequence](https://doi.org/10.1214/20-AOS1991). Mean-token scores require correct length normalization and complete-response/termination semantics before they enter a probability certificate.

Read-only inspection found no telemetry in the five registered E121 run directories. The [survival audit](theory_literature_extensions_20260905/survival.md) identifies unfinished frozen-roster, missing-observation, membership, resume-chain and summary checks in the existing analysis code. Those need completion before claiming the registered mechanism endpoints. No scheduler, cohort or analysis-code changes were made here.

## Recommended incorporation and validation

The strongest compact manuscript additions are the sharp KL certificate, attributed exact-entropy comparison, controlled-stochastic retention theorem, and a short conditional admission corollary. The natural-gradient result is an optional geometry comparison. Their mathematical ingredients should be credited rather than presented as foundational novelty.

Independent proof reviews passed. Numerical corroboration includes 216 exact finite-noise drift cases, 24 independently integrated natural-gradient cases, 126 sharp KL equality constructions, 200 random cross-entropy checks, 24 entropy-threshold constructions and 21 binomial-coverage checks. The tests corroborate algebra; they are not proofs. The supplement compiles without unresolved citations or overfull boxes. The [validation receipt](theory_literature_extensions_20260905/validation.json) records final files and hashes.
