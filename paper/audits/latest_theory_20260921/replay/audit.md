# Latest replay theory: preservation and presentation review

## Scope and handoff

Reviewed the five consecutive live-source subsections from `app:theory-gradient-availability` through `app:theory-neural-replay-response`, ending immediately before `app:theory-kl`. The source was extracted from `/tmp/paper-latest-theory-20260921/before.tex`; no live manuscript, bibliography, empirical file, or figure was edited by this agent.

Integrate `replacement.tex` as one exact block replacement. `changes.patch` records 21 bounded prose edits. The latest retention, finite-time recovery, discrete contraction, complete-response certificate, stochastic energy, and neural interference results are all retained.

Validation passed: **7,018 deterministic assertions**, **35 labels preserved in order**, **42 display-math blocks preserved byte for byte**, and every formal environment retained. The block has 12 proved formal results (3 lemmas, 3 theorems, 4 corollaries, and 2 propositions), 12 proofs, and one additional remark. All environments balance. This scope contains no figures or captions.

Final replacement SHA-256: `91f2ae2ad78158faa11232565b5fc7c78964d5c991c48256961e4b82fad9f0f4`.

## Presentation changes

- Removed both source insertion instructions, novelty claims, and discussion of what “the question below” or “this paper” contributes/trains. The mathematical relationship to IPS and UCPO remains explicitly attributed.
- Replaced “frozen policy,” “frozen bank,” and “admitted banked modes” with the relevant fixed-policy/fixed-bank assumptions.
- Stated the fresh-gradient availability bound directly as a probability bound, preserving the distinction between signal availability and magnitude.
- Rephrased the fixed-bank coefficient class as a **fixed, bounded, reward-measurable advantage**. Its polynomial coefficient and bounded integral justify the retention argument for signed coefficients.
- Made the categorical assumptions explicit in the retention theorem. They already governed its model; no assumption or result was removed.
- Clarified that uniform conditional allocation maximizes **expected** `distinct@K`, in addition to every coverage tail. This removes a possible sample-path reading of the metric statement; the stochastic-order theorem itself is unchanged.
- Rephrased the complete-response bridge as event containment and the neural local-response opening as a direct score-interference statement.
- Preserved optimizer, decoding-law, fixed-bank, complete-response, length-normalization, finite-horizon, and held-out-prompt limitations. These are scientific conditions, not editorial history.

## Mathematical review

### Fresh signal and fixed-bank retention

The mixed-group probability is exactly `1-P^G-(1-P)^G`; the nonzero fresh task component can occur only on that event. Independent enumeration of all binary groups for `G=2,...,8` and nine correctness values matches the expression and verifies both union bounds.

The softmax cross-entropy gradient is `p-w`, including outside-bank coordinates. Independent finite differences verify it under positive nonuniform targets. The energy proof only needs a bounded integral of the fresh coefficient. It therefore continues to hold for signed coefficients, using `Psi_max=max Psi`. The full-coverage and weighted/partial-bank convergence statements explicitly retain **`c_G >= 0`**. This nonnegative hypothesis is sufficient; strict positivity is unnecessary because positive replay itself forces the correct mass to one in the stationary equations. Signed fresh coefficients do not inherit those convergence claims.

The full/partial target equations, gradient-square integrability, bounded-Hessian uniform continuity, and compactness argument are consistent. The fixed partial-bank limit can eliminate unbanked correct alternatives. That limitation remains explicit. Five finite categorical ODE checks cover full/partial banks and positive/zero/negative coefficients; the energy and retention inequalities hold. These examples do not establish the analytic theorems or imply neural-training convergence.

### Normalized KL and finite-budget coverage

The sharp certificate correctly uses `KL(w_extended || p)=R_w-H(w)`, binary data processing, and Pinsker. The equality construction is on the closed simplex; the text correctly treats full-support approximation only for positive excess budget. The single-bank-target convention is `exp(-D)`. The known-alphabet and same-policy premises remain explicit.

Thirty equality constructions and 36 binary inversions pass. The illustrative `k=16`, excess-loss `0.01` values are `0.03397120762694861` and `4.619480736336e-20`, matching the printed rounding. They remain illustrations rather than experimental loss measurements.

Uniform allocation stochastically maximizes every finite-horizon coverage tail at fixed correctness. Exact occupancy recursions verify 108 small-alphabet comparisons, including zero-probability coordinates and `P=1`. The corresponding expected distinct count is the sum of the tails and agrees with the sum of occupancy indicators. This is an evaluation optimum, not an algorithmic convergence guarantee.

### Finite-time and discrete replay recovery

Direct complete-softmax differentiation verifies the log-ratio, conditional-replicator, bank-PCMD, and weighted-target equations. The sign condition is the instantaneous margin `rho-c(P)(1-P)`, with strictness requiring a nonuniform, positive bank-conditional distribution.

The recovery proof preserves bank order, contracts `exp(D)-1`, and converts its bound to a minimum bank-conditional mass. Its duration guarantee assumes the required margin and bank-mass lower bounds throughout the interval. The prose now identifies these as assumptions of the bound, without treating them as necessary conditions for all possible recovery trajectories.

The discrete theorem's order argument correctly uses `r_i-r_j <= d/2`, making `0<=gamma<=2` sufficient. The exponential contraction and its cumulative `gamma<=1` consequence are valid. There are 2,156 deterministic direct-logit contraction checks and 2,156 order checks, including ties and endpoints. A binary near-tie with `gamma=2.1` reverses order, corroborating why the step restriction matters. No clipping, momentum, alternating schedule, or stochastic-gradient conclusion was added.

### Complete responses, energy excursions, and visibility

The exemplar bound correctly requires complete normalized response events under the same prompt/decoding/verifier law. The per-prompt KL certificate uses distinct responses and normalized coefficients `alpha_j/L_j`; multiple exemplars for a key combine by deterministic grouping. Neither prefix probabilities nor mean-token probabilities can replace the complete-response probability. The length factor and distinction between checkpoint and uniform trajectory bounds remain intact.

The stochastic-energy result constructs a nonnegative supermartingale by adding the unused deterministic drift budget. Ville's inequality controls all iterates and all finitely many protected exemplars without a union factor. A finite total infinite-horizon budget gives an almost surely finite energy supremum and hence a positive, possibly random, infimum for each protected probability. The zero-initial-budget case is explicitly valid.

The smooth stochastic preconditioned-step example requires predictable SPD preconditioners, unbiased gradients, variance control, and the displayed step limit. Independent quadratic calculations verify the one-step drift bound. An exact finite nonnegative-supermartingale tree verifies five Ville-tail examples. These conditions still do not certify AdamW/PPO experimental runs.

The occupancy lower bound and union-bound visibility budget use one fixed checkpoint and independent draws. The zero-sighting binomial upper bound is `0.0893681989862648` at 32 draws and 95% one-sided confidence, matching `8.94%`. It supplies no positive lower bound or proof of extinction.

### Neural local response

The Taylor bound and exponentiation preserve the length factor. Positive semidefiniteness makes individual Gram diagonals nonnegative but does not make row averages positive. The interference margin includes the dose-dependent curvature cost, so arbitrarily increasing replay strength is not a guarantee.

The scalar shared-parameter counterexample has exact gradients `(1+s,-2+s)` and replay direction `s-1/2`; the first exemplar declines while the mean replay score increases. Deterministic checks confirm these identities and the stated Taylor lower bound using a global categorical score-curvature bound. The actual AdamW-step decomposition and absence of established experimental interference/curvature margins remain explicit.

## Primary-source checks

Existing downloaded primary sources were available locally, so no external retrieval or bibliography alteration was needed.

- Sinha et al., `arXiv:2601.21669v1`, Theorem 4.1: the detached ideal IPS field is `r_i-p_i sum_j r_j`; normalizing rewards to a fixed target gives `w-p`. Read from `paper/audits/reference_audit_20260905/modern_rl_sources/sinha2026expected_full.txt`.
- Lochab et al., `arXiv:2605.00365v1`, **Sections 5.2 and 6.2**: uniform-correct target and forward conditional-KL objective. The section locator was made explicit in the prose citation after checking the actual source headings/formula in `lochab2026ucpo_full.txt`.
- Anceaume et al., `arXiv:1504.03878`, Theorem 3: every collection size `c=1,...,n` is stochastically fastest under the equal non-null probabilities at fixed null mass. Read from `paper/audits/theory_literature_extensions_20260905/discovery_sources/anceaume2015.txt`; endpoint extension follows by finite-horizon continuity.
- The van Erven–Harremoës data-processing/Pinsker locators, Wang et al. predictable variable-metric drift assumptions, Howard et al. Ville inequality locator, and Clopper–Pearson inversion were cross-checked against the existing primary-source audit records in `paper/audits/theory_appendix_integration_20260905/{survival_integration_notes,optimizer_admission_notes}.md`. The manuscript supplies the actual applications and proof steps.

## Remaining integration work

Root handles whole-manuscript compilation, generated-artifact consistency, TOC/reference checks, and final visual layout. No formula correction or unresolved mathematical defect was identified within this block. No model call, neural training, empirical rerun, bootstrap, or new empirical claim was made.
