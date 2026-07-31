# E72 — Baseline and Decoding Control Suite

Status: DRAFT DESIGN (not frozen, not submitted)
Date opened: 2026-07-31
Owner: od2961
Supersedes: nothing. Extends the E70/E71 headline surface.

---

## 0. Sequencing decision — one arm at a time

Decided 2026-07-31. Arms are added to the campaign strictly in this order, each
landing as a complete, audited, reportable unit before the next begins.

| Order | Arm | Why here | New objective code | Cost |
| --- | --- | --- | --- | --- |
| 1 | **Tier 0 decoding frontier** — DONE 2026-07-31, see 11b | Not a baseline — a prerequisite. It reinterprets every number already in the paper, and until it lands we do not know what a training baseline has to beat. | none | ~13 GPU-h actual |
| 2 | **B3a** count bonus, no replay | Highest probability of actually recovering breadth, so we want it early, not last. Zero new code. Doubles as the replay-necessity ablation that `tab:component-ablations` currently supports only analytically. | none | ~280 GPU-h |
| 3 | **B1a** verified replay only | The reviewer's first-named alternative, and the complement of B3a: together they decompose xGRPO into discovery vs memory. | none | ~280 GPU-h |
| 4 | **B2b** matched token entropy | The paper's central conceptual claim (token entropy vs executed-key entropy), testable against a target that already exists in frozen telemetry. | none (needs T-6 target table) | ~280 GPU-h |
| 5 | **B5b** in-group key-uniform reweighting | Most dangerous single baseline; needs new code, so it follows the zero-code arms. | ~40 lines | ~280 GPU-h |
| 6 | **B4** pass@k | Completes the reviewer's list. | ~120 lines | ~280 GPU-h |
| 7 | Wave B (B1b, B3b, B5a, B6b) | Identity-blind controls and robustness. | see §8 | ~1,100 GPU-h |

Rationale for putting **B3a before B1a**, against the order the review lists
them: the bonus drives discovery and replay only persists it, so B3a is the arm
most likely to close most of the gap on its own. A campaign should surface its
own worst case early, while there is still time to respond to it. B1a is the
safer bet — a baseline that discovers ~1 mode per prompt has little for replay
to preserve — and a safe result learned late is worth less than a threatening
one learned first.

Each arm is 5 domains × 5 seeds × 12 passes = 25 runs ≈ 280 GPU-hours ≈ 12 h
wall on 24 GPUs, so an arm lands in about a day once its plumbing exists.

---

## 1. The weakness this campaign closes

The manuscript's only headline comparator is compute-matched Dr.GRPO
(`paper/main.tex`, Table `tab:main-results`). Section `sec:limitations` already
concedes the gap:

> Baselines are limited to matched Dr.GRPO; wider decoding, matched token
> entropy, and distribution matching remain untested.

As written, the paper establishes **the full xGRPO package beats plain
Dr.GRPO**. It does not establish **executable-mode machinery beats simpler
diversity-preserving alternatives**. Two distinct objections follow:

- **O1 (weak comparator).** Any of: verified-response replay, a token-entropy
  bonus, a count-based rare-outcome bonus, a pass@K objective, or a
  distribution-matching objective might recover most of the reported breadth
  without banks, canonical keys, replay scheduling, or self-referenced
  controllers.
- **O2 (temperature confound).** The paper is motivated by repeated sampling
  but evaluates only at temperature one. If the baseline's collapse disappears
  at T = 1.4, the finding is "RL sharpened the policy" (a decoding-time
  phenomenon), not "RL destroyed support" (an irreversible one). These are
  different papers.

O2 is cheaper to answer than O1 and is more dangerous if left open, because it
threatens the *interpretation* of every existing number rather than only the
comparator set. This design therefore front-loads O2 (Tier 0, no training) and
stages O1 behind it.

### 1.1 Mapping from the review to this design

| Reviewer request | Arm(s) here | Code status |
| --- | --- | --- |
| Dr.GRPO + ordinary verified-response replay | B1, B1b | ready / small |
| Matched token-entropy regularizer | B2a, B2b, B2c | ready / small |
| Rare-outcome or count-based bonus without replay | B3a, B3b, B3c | ready / small |
| pass@K or set-level objective | B4 | **new** |
| Distribution-matching or mode-covering objective | B5a, B5b | **new** |
| Temperature / decoding adjustments to the baseline | Tier 0, B6 | small |

---

## 2. Estimands: what "better" means once temperature is free

The current headline compares two policies at one decoding setting. Once
decoding is a free variable, a single (accuracy, breadth) pair is no longer the
right object. We define three estimands and pre-commit to reporting all three.

Let `A` be an arm (a trained policy at its terminal pass-12 checkpoint), `d` a
domain, `T` a decoding temperature, `K` a sample budget.

**E1 — Decoding frontier (breadth at matched accuracy).**
Sweeping `T` traces a curve of points `(mean@K(A,d,T), distinct@K(A,d,T))`.
Define

```
D_A,d(alpha) = max { distinct@K(A,d,T) : mean@K(A,d,T) >= alpha }
```

the best breadth the arm can buy at accuracy at least `alpha`. The primary
claim becomes: **xGRPO's frontier dominates every baseline's frontier over the
accuracy range both attain.** This is the reviewer-proof form of the result: it
cannot be defeated by "just raise the temperature", because raising the
temperature is inside the estimand.

**E2 — Temperature-repair index.** For an arm A and a reference arm R,

```
rho(A -> R) = ( max_T distinct@K(A,d,T) ) / distinct@K(R,d,1.0)
```

with the max taken over the full sweep and *no accuracy constraint at all*.
`rho < 1` means the baseline cannot reach the treatment's temperature-one
breadth even when allowed to sacrifice arbitrary accuracy. That is the
operational meaning of "irreversible support loss" and is the single number
that settles O2. We additionally report the accuracy at that argmax `T`, so a
`rho` near 1 bought at ruinous accuracy is visible.

**E3 — Coverage-budget curve.** `distinct@K` and `pass@K` for
`K in {1,2,4,8,16,32,64}` at each arm's own best temperature. This is the curve
the Discussion section already promises ("the next test is a coverage-budget
curve") and it is what a repeated-sampling practitioner actually consumes. It
also exposes a case the current design cannot see: a baseline whose support is
intact but low-probability would show `distinct@64` recovering while
`distinct@8` stays pinned.

**Reference lines.** Every frontier plot carries the **pre-RL base model** at
the same temperature grid. A baseline whose frontier lies below the base model
at every temperature has lost support relative to initialization — the
`yue2025rlvrlimit` boundary claim, measured directly rather than argued.

**Note on what E1 changes about the paper.** The existing Table
`tab:main-results` stays as the temperature-one operating point. E1/E2/E3
become a new results subsection and an appendix. No existing number is
retracted or recomputed.

---

## 3. Arm roster

All arms share the E70/E71 common design unless stated: Qwen2.5-0.5B-Instruct at
pinned revision `7ae557604adf67be50417f59c2c2f167def9a775`, seeds 43–47,
G = 16 rollouts, `beta_KL = 0`, rollout temperature 1, 12 epochs over 384 train
prompts, 128 eval prompts, 4,608 optimizer updates, lr 2e-7, response length 192
(PantryPlan: 8 tokens, six-decision support mask).

Every arm keeps the **passive verifier bank bookkeeping** (as the matched
Dr.GRPO control already does) so discovery telemetry is comparable across the
whole roster without altering any arm's objective.

### B0 — matched Dr.GRPO (existing control)

Runtime variant `grpo_compute_matched`, `online_canonical_replay_compute_only=1`,
all auxiliary coefficients zero. Already run for all five domains × five seeds
(E70/E71). **No re-run needed.**

### B1 — Dr.GRPO + ordinary verified-response replay

*Claim under test:* rehearsal of correct responses alone preserves breadth; the
rare-mode advantage, balance KL, and controllers are unnecessary.

- **B1a (key-aware mass-only replay).** Bank retains one exemplar per verified
  key; **only** the verified-mass likelihood loss acts. No rare/new advantage,
  no balance KL.
  ```
  OAT_ZERO_SEMANTIC_SHANNON_COEF=0
  OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0
  OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=0          # balance KL off
  OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_ALPHA=<dose>
  OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=verified_likelihood
  OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=1
  ```
  Code status: **ready**, pending a smoke check that `replay_alpha=0` fully
  disables the balance term rather than falling back to a default
  (`src/oat_drgrpo/args.py:193-221`, `src/oat_drgrpo/canonical_replay.py`).
- **B1b (identity-blind replay).** The scientifically stronger control: replay
  buffers verified *responses* with **no canonical key at all** — retain the
  first N verified responses per prompt, dedup by raw string only. This removes
  every piece of ModeBench identity machinery from the baseline and is what a
  practitioner without a validator-key contract would build.
  Code status: **small** — add
  `online_canonical_replay_key_mode = {"canonical", "raw_response"}` and route
  exemplar admission accordingly.

*If B1 wins:* the paper's contribution narrows to "verified replay is the active
ingredient", and the rare/new advantage and balance KL must be demoted. Say so
in advance (Section 7).

### B2 — Matched token-entropy regularization

*Claim under test:* the treatment is a complicated way of keeping token entropy
up; a plain entropy bonus does the same job.

The frozen E70/E71 telemetry already contains the matching target. From seed 43
(`train/entropy`, mean over the last 200 logged updates):

| Domain | matched Dr.GRPO | xGRPO |
| --- | --- | --- |
| Graph coloring | 0.0005 | 0.5914 |
| PantryPlan | 0.0087 | 0.3726 |

The baseline's token entropy is effectively zero at the end of training while
the treatment's is two to three orders of magnitude higher. That makes a
*matched-entropy* arm well defined without new measurement runs.

- **B2a (fixed-dose entropy bonus).** `policy_entropy_coef in {1e-3, 3e-3, 1e-2}`
  (legacy token-entropy control; `src/oat_drgrpo/args.py:318`), or the direct
  MaxEnt objective `OAT_ZERO_MAXENT_ALPHA` with
  `OAT_ZERO_MAXENT_OBJECTIVE=conditional`. Code status: **ready**
  (`INCLUDE_TOKEN_ENTROPY_ARM`, `INCLUDE_MAXENT_ARM`).
- **B2b (target-matched dual controller).** The headline entropy comparator.
  Haarnoja-style dual ascent on the entropy coefficient with the target set to
  **xGRPO's own measured terminal token entropy in that domain**
  (`OAT_ZERO_MAXENT_DUAL_TARGET_ENTROPY=<per-domain value from the table
  above>`, `INCLUDE_MAXENT_DUAL_ARM=1`). This arm is, by construction, matched on
  token entropy and differs from xGRPO only in *what the entropy is over*
  (tokens vs executed keys). It is the cleanest possible statement of the
  paper's central distinction. Code status: **ready**; requires a per-domain
  target table extracted from frozen artifacts (script task T-6).
- **B2c (clip-higher / DAPO-style).** Asymmetric PPO clipping as the modern
  entropy-preserving alternative (`yu2025dapo`). Code status: **small** — a
  separate high clip bound must be added to the PPO loss
  (`src/oat_drgrpo/learner/grpo.py`, ratio-clipping block). Deferrable to
  Tier 3 if schedule is tight.

*Reporting requirement:* every B2 run logs `train/entropy` throughout, and the
results table reports realized terminal token entropy alongside breadth. A
"matched" claim that does not verify realized matching is not a matched claim.

### B3 — Rare-outcome / count-based bonuses without replay

*Claim under test:* the count-based bonus is doing the work; the bank memory and
replay scheduling are ceremony.

- **B3a (executed-key count bonus, no replay).** The E58 open-set rare-mode plus
  first-discovery advantage, replay entirely disabled.
  ```
  OAT_ZERO_SEMANTIC_SHANNON_COEF=0.10 (dose swept)
  OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.50
  OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=0
  OAT_ZERO_INCLUDE_OPEN_SET_SPLIT_CANONICAL_ARM=1   # replay coefficients zeroed
  ```
  Code status: **ready**. This is also the cleanest isolation of the
  *replay* component, which the paper currently supports only by analytic and
  developmental evidence (`tab:component-ablations`).
- **B3b (identity-blind count bonus).** Same bonus, but the count key is the
  **normalized final answer string** rather than the executed canonical key.
  This is the closest analogue of `song2025outcome`'s UCB-over-answers and is the
  direct test of the ModeBench contract's value. Code status: **small** — add
  `raw_answer_normalized` to `online_canonical_key_mode`
  (`src/oat_drgrpo/args.py:252-258`).
- **B3c (in-batch repeat penalty).** Penalize repeated outcomes within a group,
  no history, no bank. Code status: **ready**
  (`INCLUDE_OUTCOME_COLLISION_ARM`, `OAT_ZERO_OUTCOME_COLLISION_COEF`).

### B4 — pass@K / set-level objective

*Claim under test:* optimizing the sampling-set objective directly is the
principled fix; a mode-identity mechanism is unnecessary.

Implementation (pre-registered, faithful to `chen2025passktraining`): partition
the G = 16 on-policy rollouts into `G/k` disjoint subsets of size `k`; assign
every row in a subset the subset-level reward `r~_i = max_{j in subset(i)} r_j`;
then apply the *unchanged* Dr.GRPO centering to `r~`. Sweep `k in {2,4,8}`.
Partition is by a deterministic per-step permutation seeded from
`(prompt_id, global_step)` so it is checkpoint-reproducible.

Two properties make this the right form: it changes only the reward transform
(so the compute match with every other arm is exact), and at `k=1` it reduces
exactly to B0, giving a built-in sanity anchor.

Code status: **new** — pure function in a new
`src/oat_drgrpo/set_level_objectives.py`, applied immediately before
`compute_monte_carlo_advantages` (`src/oat_drgrpo/learner/grpo.py:2990,3296`),
plus unit tests. Estimated 120 lines + tests.

*Expected failure mode worth measuring:* pass@k training raises `pass@8` by
tolerating low-accuracy exploration, without necessarily raising `distinct@8`
(one mode found from more places). If so, that is a genuine, publishable
distinction and belongs in the paper.

### B5 — Distribution matching / mode covering

*Claim under test:* a mass-covering divergence recovers support without any
executable identity.

- **B5a (sequence-level uniform-over-correct, SNIS).** Target
  `p*(y|x) ∝ 1[V(y,s_x) != bot]` — uniform over *correct sequences*. Descending
  the forward KL `KL(p*||pi)` from on-policy samples gives a weighted
  log-likelihood of correct rows with self-normalized weights `w_i ∝ 1/pi(y_i|x)`.
  Numerically this must be tamed: use length-normalized sequence log-prob
  `s_i = log pi(y_i|x)/|y_i|`, weights `w_i ∝ exp(-(s_i - max_j s_j)/T_w)`
  restricted to correct rows and renormalized, with a hard ratio cap `C = 10`
  and `T_w` swept in `{0.5, 1.0}`. This is genuinely mode-covering — but at the
  *sequence* level, which is exactly the distinction the paper claims matters
  (surface strings overcount aliases; `sec:collapse`).
- **B5b (in-group key-uniform reweighting).** The most dangerous baseline in the
  roster and therefore mandatory: if a group holds `M` correct rows spanning `J`
  distinct executed keys with multiplicities `m_c`, weight each correct row by
  `1/(J * m_{c(i)})` instead of uniformly. This is mode-balancing *with* the
  canonical key but *without* bank, replay, controllers, or cross-batch memory.
  It isolates the paper's memory claim to a single number.

Code status: **new**, same module and insertion point as B4. B5b is ~40 lines
(the group's key vector is already computed for the semantic advantage). B5a is
~80 lines plus a numerical-stability smoke test.

*If B5b matches xGRPO:* the contribution is "in-group mode balancing", not
"persistent verified memory", and Section `sec:replay` plus
Theorem `thm:xdr-balance` must be re-scoped. Pre-committed in Section 7.

### B6 — Decoding and training-temperature adjustments

- **B6a (evaluation-time frontier).** Tier 0. No training. See Section 4.
- **B6b (hotter rollout temperature).** Retrain B0 with rollout temperature
  `OAT_ZERO_TEMPERATURE in {1.2, 1.5}` (evaluation grid unchanged). Tests
  whether collecting hotter during training preserves support. Code status:
  **ready**.
- **B6c (truncation-sampling changes).** Evaluation-side only: `top_p in
  {1.0, 0.95}` and `min_p in {0, 0.02}` crossed with the temperature grid, for
  both arms. Code status: **small** — add `eval_mode_coverage_top_p` (and
  optionally `min_p`) next to `eval_mode_coverage_temperature`
  (`src/oat_drgrpo/args.py:321-327`).

---

## 4. Tier 0 — the decoding frontier (run this first)

**Why first:** zero training cost, answers O2 outright, and its result changes
what Tier 1/2 must argue. If baseline collapse *were* temperature-repairable,
the whole framing changes and we should know before spending 1,600 GPU-hours.

**Inputs.** The frozen terminal exports already on disk, e.g.
`var/data/xdr_qwen25_0p5b_instruct_{arm}_{prefix}_s{seed}/debug_job*/saved_models/step_04609`
— present for all 5 domains × 2 arms × 5 seeds (50 checkpoints). Note that
`export_steps=0` means **terminal-only exports**; intermediate weights were not
retained, so the frontier is a pass-12 object. If a frontier *trajectory* is
wanted later, it needs re-runs with `export_steps>0` (Tier 3).

**Mechanism (decided).** Do **not** write a second evaluator. PantryPlan and
Graph use restricted canonical-action sampling inside the learner, and any
standalone re-implementation would silently diverge from the reported metric.
Instead run the ordinary training entry point in **eval-only mode**: same
templates, validators, mode-key extraction, and draw seeds as training, with the
checkpoint as `OAT_ZERO_PRETRAIN` and a zero-update budget, taking the step-0
evaluation. `ops/eval_exact_answer_mode_coverage.py` is retained only as a
cross-check on Graph, where it already works.

Implementation task T-1 must verify a zero-update run performs and logs its
step-0 evaluation and exits cleanly; the fallback (if a zero budget is rejected)
is a one-update budget with the post-update evaluation discarded by the
aggregator.

**Grid.**

| Factor | Values | Notes |
| --- | --- | --- |
| checkpoint | 50 terminal + 5 base-model | base = pre-RL reference per domain |
| temperature | 0.5, 0.7, 1.0, 1.3, 1.6, 2.0 | 1.0 reproduces the published cell |
| top_p | 1.0 (full grid), 0.95 (T ∈ {1.0,1.6} only) | B6c |
| K | 8 with 4 replicates (all cells); 32 and 64 with 1 replicate (T ∈ {1.0, best-T}) | E3 |
| eval seed | frozen per domain, as in E70/E71 | reuse `OAT_ZERO_EVAL_MODE_COVERAGE_SEED` |

**Validation gate (mandatory).** At `T = 1.0, top_p = 1.0, K = 8, 4 draws`, every
cell must reproduce the published Table `tab:main-results` value for that
domain/arm/seed to within Monte-Carlo tolerance (the recorded draw standard
error). Any cell that does not reproduce invalidates the sweep — it means the
eval-only path is not the training eval path. This gate is the reason to reuse
the learner rather than reimplement.

**Cost.** ~350 short jobs, ~10 min each including model load ≈ **60 GPU-hours**;
under 4 hours wall on 16 GPUs.

**Outputs.**
- `var/artifacts/e72_decoding_frontier_raw.jsonl` (one row per cell)
- `var/artifacts/e72_decoding_frontier_summary.json` (E1/E2/E3 per domain)
- `paper/figures/e72_decoding_frontier.pdf` — 5 panels, accuracy on x, breadth
  on y, one curve per arm with temperature annotated, base-model curve dashed
- `paper/figures/e72_budget_curve.pdf` — `distinct@K` vs `K`, log-x

---

## 5. Tier 1 — dose calibration screen

Every new baseline has a free coefficient. Running each at one guessed dose and
reporting that it lost is the single most common way a baseline comparison is
dismissed. Tier 1 exists purely to remove that objection.

**Rule (pre-registered, fixed before any Tier 1 result is read).** For each
baseline family, sweep three doses spanning an order of magnitude on the
**screening domain** at reduced budget, and select the dose maximizing
`distinct@8` **subject to** terminal `pass@1` no lower than matched Dr.GRPO's
terminal `pass@1` minus 0.05. If no dose satisfies the accuracy constraint,
select the one maximizing `distinct@8` outright and record that the family
cannot preserve accuracy — that is itself a finding, reported as such.

**Deliberate asymmetry, to be stated in the paper.** Baselines select their dose
using the evaluation metric. xGRPO does not: its coefficients are the frozen
E58 doses, chosen before this campaign existed. The comparison is therefore
biased *in favor of the baselines*, and we say so in the manuscript rather than
letting a reviewer discover it.

**Screening configuration.**
- Domains: **Graph coloring** (largest published collapse: baseline
  `distinct@8 = .325` vs treatment `2.406`) and **Countdown** (mid-range, and
  the domain where the baseline retains real accuracy at `.588`). Two screens
  guard against a dose that is optimal only where the effect is largest.
- Seeds: 43, 44 (two seeds; screening only, never reported as a result).
- Budget: 4 passes (1,536 updates) instead of 12 — ~3 hours per run.
- Doses per family: B1 mass alpha {0.03, 0.10, 0.30}; B2a coef {1e-3, 3e-3,
  1e-2}; B2b target = xGRPO entropy × {0.5, 1.0} (2 doses); B3a semantic coef
  {0.03, 0.10, 0.30}; B3b same; B3c collision coef {0.03, 0.10, 0.30}; B4
  k {2, 4, 8}; B5a T_w {0.5, 1.0}; B5b (parameter-free, dose sweep not needed —
  screened once to confirm stability); B6b temperature {1.2, 1.5}.

**Cost.** ~28 configurations × 2 seeds × 2 domains × 3 h ≈ **340 GPU-hours**,
about 15 hours wall on 24 GPUs.

**Screening also gates correctness.** Any new-code arm (B4, B5) must first pass
a 200-update smoke run with: finite losses throughout, no fail-closed integrity
violation, `k=1` pass@k reducing to B0 within numerical tolerance, and B5a
importance weights inside the declared cap. Smoke failures block Tier 2 entry
for that arm.

---

## 6. Tier 2 — headline baseline surface

**Design.** Selected dose per family, full common design: 5 domains × 5 seeds ×
12 passes, identical data, prompt order, rollout count, evaluation cadence, and
recovery rules as E70/E71. Arms: B1a, B1b, B2b, B3a, B3b, B4, B5a, B5b, B6b.
(B2a, B2c, B3c are Tier 3 unless their screen is unexpectedly strong.)

**Size.** 9 arms × 5 domains × 5 seeds = **225 runs** at ~11 h ≈ 2,500
GPU-hours. That is more than the campaign needs. Two staged waves:

- **Wave A (decisive subset).** 5 arms — B1a, B2b, B3a, B4, B5b — one per
  reviewer-named family, × 5 domains × 5 seeds = **125 runs ≈ 1,400 GPU-hours**,
  ~3 days wall on 24 concurrent GPUs. This is the minimum publishable answer to
  O1: every family the reviewer named, on every domain, at full seed count.
- **Wave B (identity and robustness).** B1b, B3b, B5a, B6b × 5 domains × 5 seeds
  = 100 runs. B1b/B3b are the identity-blind controls that defend the ModeBench
  contract; B5a/B6b are robustness. Launch after Wave A's first domain lands
  clean.

**Resource plan.** `mltheory` (5 nodes × 10 GPUs, currently idle) for Wave A to
avoid preemption; `pvl-lowprio`/`lowprio` (~40 nodes, preemptible, watchdog
requeue already implemented) for Wave B. Single GPU per run, 8 CPUs, 64 G,
1-day limit — as in `ops/exp_scaling/launch_e71_scale384_05b.sh`.

**Storage — a real constraint.** `/n/fs/similarity` has 2.4 T free and each
E71-style run leaves ~15 G (14 G DeepSpeed resume state + 1.3 G export). 225 runs
at that rate is ~3.4 T and would fill the filesystem mid-campaign. **Required
settings:** `OAT_ZERO_MAX_SAVE_NUM=1` and
`OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=1` (currently `0` in E71), which drops a
completed run to its ~1.3 G terminal export — ~300 G for the full 225. Verify
the prune path on the first completed run before Wave A saturates.

---

## 7. Analysis plan and pre-committed interpretations

**Primary comparison.** Per domain, at the terminal pass-12 checkpoint:
xGRPO versus the **best-of-baselines comparator** — the maximum over all
baseline arms of `distinct@8`, subject to that arm's `pass@1` being within 0.05
of matched Dr.GRPO's. Comparing against the per-domain best baseline is
deliberately conservative (it lets the baseline set be chosen after the fact, in
its own favor) and is the form a skeptical reader will apply anyway.

**Secondary.** Full frontier comparison (E1) at the terminal checkpoint, and E2
for every arm.

**Statistics.** Five paired seeds per domain. Report every seed row (as
Appendix `app:per-seed` already does), the paired mean delta, a
seed-level paired bootstrap interval (10,000 resamples), and the exact sign
test. n = 5 caps the one-sided sign-test p at 1/32; we therefore lead with
magnitude and direction-consistency across domains, as the existing evidence
policy already does, and do not pool domains into a single test.

**Pre-committed outcomes.** Written now, before any result:

- **P1.** If no baseline reaches within 0.15 `distinct@8` of xGRPO at matched
  accuracy in ≥4/5 domains → the claim strengthens to "executable-mode
  machinery outperforms every simpler diversity-preserving alternative tested",
  and the limitation paragraph is deleted.
- **P2.** If one baseline family matches xGRPO (within 0.15 `distinct@8` at
  matched accuracy) in ≥3/5 domains → the contribution is rewritten as
  "xGRPO and *<family>* both preserve breadth; xGRPO additionally
  <whatever it still does better, stated from data>". No selective-domain
  reporting.
- **P3.** If B5b (in-group key-uniform reweighting) matches xGRPO → the
  persistent-memory claim (`sec:replay`, `thm:xdr-balance`) is re-scoped to
  in-group balancing, and replay is demoted to an efficiency mechanism.
- **P4.** If the temperature sweep shows the baseline recovering xGRPO's
  temperature-one `distinct@8` at any temperature with accuracy loss below 0.05
  (E2 `rho >= 1` under the accuracy constraint) → the paper's framing changes
  from "support loss" to "sharpening", the abstract and introduction are
  rewritten accordingly, and this becomes the headline finding rather than a
  buried caveat.
- **P5.** If B4 (pass@k) raises `pass@8` but not `distinct@8` → report it as a
  distinct, favorable finding: set-level objectives buy retries, not
  alternatives.

**Reported alongside every arm:** realized terminal token entropy
(`train/entropy`), cumulative verified discoveries, invalid-response fraction,
mean response length, and wall clock — so a breadth gain bought by degeneracy
(longer, malformed, or lower-accuracy outputs) is visible rather than hidden.

---

## 8. Implementation work items

| # | Task | Files | Size |
| --- | --- | --- | --- |
| T-1 | Eval-only run mode (zero-update budget, step-0 eval, clean exit) + reproduction gate against published cells | `src/oat_drgrpo/learner/run.py:3484`, `ops/run_experiment.sh` | M |
| T-2 | `eval_mode_coverage_top_p` (and `min_p`) plumbing | `src/oat_drgrpo/args.py:321-327`, `learner/run.py:3672-3760`, `ops/train.sh` | S |
| T-3 | Frontier sweep launcher + aggregator (E1/E2/E3) | new `ops/exp_scaling/launch_e72_decoding_frontier.py`, `ops/exp_scaling/aggregate_e72_frontier.py` | M |
| T-4 | pass@k subset reward transform | new `src/oat_drgrpo/set_level_objectives.py`, hook at `learner/grpo.py:2990` | M |
| T-5 | Distribution-matching weights (B5a SNIS, B5b key-uniform) | same module, hook after task centering (`learner/grpo.py:3055-3120` pattern) | M |
| T-6 | Per-domain xGRPO token-entropy target table from frozen E70/E71 telemetry | new `ops/exp_scaling/extract_e72_entropy_targets.py` → `var/artifacts/e72_token_entropy_targets.json` | S |
| T-7 | `raw_answer_normalized` key mode (B3b) and `raw_response` replay key mode (B1b) | `src/oat_drgrpo/args.py:252-258`, `online_canonical_bank.py`, `canonical_replay.py` | M |
| T-8 | New arms in the comparative submitter | `ops/submit_countdown_comparative.sh:29-63, 870-1000` | M |
| T-9 | Unit tests: pass@k reduces to B0 at k=1; subset partition determinism; SNIS weight cap; key-uniform weights sum to 1 per group; zero-coefficient arms bit-identical to B0 | new `tests/test_e72_set_level_objectives.py`, `tests/test_e72_arm_configs.py` | M |
| T-10 | Campaign launcher, identity/hash freeze, audit, watcher (E71 pattern) | new `ops/exp_scaling/launch_e72_baseline_suite.sh`, `audit_e72_baseline_suite.py`, `watch_e72_baseline_suite.sh` | L |
| T-11 | Preregistration documents | `paper/preregistration/e72_decoding_frontier_20260731.md`, `paper/preregistration/e72_baseline_suite_05b.md` | M |
| T-12 | Paper integration: new results subsection, two figures, two tables, rewritten limitations | `paper/main.tex` | M |
| T-13 | DAPO clip-higher (B2c) — deferrable | `src/oat_drgrpo/learner/grpo.py` ratio clipping | M |

**Ordering.** T-1 → T-2 → T-3 unlocks Tier 0 and can ship independently of every
training change. T-4/T-5/T-7 → T-9 (tests) → T-8 → smoke runs → Tier 1. T-10/T-11
must land before any Tier 2 submission (the repo's freeze-before-submit
convention). T-12 last.

---

## 9. Provenance and audit requirements

Follows the existing campaign contract, not a new one:

- Preregistration frozen and hash-bound before submission; protocol SHA-256
  recorded in `var/artifacts/e72_*_identity.json` together with source-tree and
  ops-tree hashes, exactly as `launch_e71_scale384_05b.sh` does.
- Jobs submitted **held**; the launcher verifies exact job count, arms, seeds,
  doses, budget, evaluation cadence, and resource request before release; any
  mismatch cancels the held cohort.
- Fail-closed integrity: non-finite loss or coefficient, state mismatch, missing
  terminal marker, dose disagreement with the frozen manifest.
- No best-checkpoint selection, early stopping, seed substitution, carry-forward,
  or missing-cell averaging. Terminal pass 12 only, all five seeds, or the cell
  is not reported.
- Failed runs resume only from their own source-bound checkpoint with the same
  arm, seed, data, and protocol. Screening (Tier 1) results are never promoted
  into a reported table.

---

## 10. Budget summary

| Tier | Runs | GPU-hours | Wall (24 GPUs) | Prereq |
| --- | --- | --- | --- | --- |
| Tier 0 frontier | ~350 eval-only | ~60 | ~4 h | T-1..T-3 |
| Tier 1 screen | ~112 short | ~340 | ~15 h | T-4..T-9 |
| Tier 2 Wave A | 125 | ~1,400 | ~3 d | T-10, T-11 |
| Tier 2 Wave B | 100 | ~1,100 | ~2 d | Wave A clean |
| **Total** | **~690** | **~2,900** | **~6 d** | |

Storage after pruning: ~300 G. Without pruning: ~3.4 T (does not fit).

---

## 11. Open decisions

1. **Wave A arm selection.** Roster above picks B1a, B2b, B3a, B4, B5b — one per
   reviewer-named family, choosing the *strongest* member of each. Alternative:
   swap B5b → B5a if we would rather test sequence-level distribution matching
   first (weaker baseline, cleaner literature anchor).
2. **Screening domains.** Graph + Countdown proposed. PantryPlan is the paper's
   narrative centerpiece; screening on it instead would tune baselines where the
   story is told, which cuts both ways.
3. **Do we run Tier 1 at all for B5b?** It is parameter-free; skipping its screen
   saves little and costs a stability check. Currently: screen once, no sweep.
4. **B2c (DAPO clip-higher)** — Tier 3 as written. Reviewers who cite
   `yu2025dapo` may expect it in the main table.
5. **Whether Tier 0 gets its own short paper section** or folds into the results
   section. Given O2's severity, a dedicated subsection is probably right.

---

## 11a. Tier 0 implementation record (built 2026-07-31)

**Mechanism.** Eval-only mode was added to the learner rather than building a
second evaluator: `eval_only` (`src/oat_drgrpo/args.py`) makes `run()` evaluate
the loaded policy once through the ordinary training evaluation path and exit
before any rollout, optimizer step, export, or resume checkpoint
(`src/oat_drgrpo/learner/run.py`). A cell is therefore a pure function of the
checkpoint and its decoding settings, measured by the identical code that
produced every published number, including PantryPlan's restricted
canonical-action sampling. Completion is recorded in `EVAL_ONLY_COMPLETE.json`,
and the aggregator refuses any cell without that marker.

`eval_mode_coverage_top_p` was added alongside the existing temperature knob and
reaches the sampled draws only; the greedy trace keeps the untruncated surface
so it stays comparable across a sweep. Both flags are inert at their defaults
and fail closed against a frozen source snapshot that predates them
(`ops/train.sh`).

**Configuration inheritance.** `ops/exp_scaling/build_e72_frontier_manifest.py`
resolves all 50 terminal checkpoints and inherits each domain's evaluation
configuration from the learner's own recorded argument block, rather than
re-typing a table. It requires the two arms of a domain to agree on every
inherited field, and binds each cell to the published terminal metrics from the
frozen scaling-curve artifact. First run: 50/50 resolved, 0 problems.

**Reproduction gate.** Four pilot cells at the published decoding surface
(T = 1.0, top_p = 1.0, K = 8, 4 draws) reproduced their published terminal
values *bit-exactly* — Graph and PantryPlan, both arms, seed 43, all four
metrics — including the stochastic xGRPO cells (`distinct@8` 2.4062 against a
published 2.4062). The gate tolerance is
`max(3 x published draw SE, 0.02 for rates / 0.05 for mode counts)`.

**Hardware is part of the measurement.** The frozen pass-0 rows of a single
domain are not identical across its ten runs: they fall into exactly two groups,
matching the two GPU types the cohort ran on (31 runs on node302's a100, 19 on
node105's a5000). Identical weights and identical draw seeds still produce
different samples because GPU type changes floating-point reduction order, and
at a high-entropy policy that difference compounds — Graph pass-0 `pass@8` is
`.537` on one group and `.436` on the other. Every frontier cell is therefore
pinned to the node that trained its checkpoint. An unpinned sweep would have
mixed hardware into the accuracy/breadth trade-off the sweep exists to isolate,
and would have done so invisibly. This constraint applies to Tiers 1 and 2 as
well: **arms within a comparison must share a GPU type**, and the E70/E71 cohort
already violates that across seeds, which is worth stating in the manuscript.

**Cost, measured.** A stage-A cell takes about 2 minutes of evaluation plus
startup; the full 300-cell stage A runs in roughly 80 minutes of wall clock on
the 18 pinned mltheory GPUs. This is well under the 60 GPU-hour estimate in
section 10.

**Deferred within Tier 0.** Stage B (K=32 budget curve) and stage C (top_p
truncation) are implemented in the launcher but not yet submitted. Base-model
reference cells are not yet built: the base checkpoint is available as each
run's `pretrain` path, and the published step-0 rows give them a gate, but they
need an `arm = base` entry in the manifest and a gate exemption path.

## 11b. Tier 0 stage-A result (completed 2026-07-31)

300/300 cells measured, zero job failures, reproduction gate **PASS at
200/200 checks** — all 50 published terminal cells re-measured within tolerance.

**Temperature does not repair the collapse.** Matched Dr.GRPO's breadth is
essentially flat in temperature, and what little it gains is bought with
accuracy:

| domain | Dr.GRPO best `distinct@8` (at T) | accuracy there | xGRPO `distinct@8` at T=1 | rho |
| --- | --- | --- | --- | --- |
| Graph coloring | .327 (T=1.3) | .322 | 2.406 | **0.14** |
| Countdown | .650 (T=1.6) | .523 | 1.893 | **0.34** |
| Python factors | .172 (T=0.5) | .172 | 1.594 | **0.11** |
| MathIR | .675 (T=2.0) | .363 | 0.926 | **0.73** |
| PantryPlan | .719 (T=2.0) | .522 | 2.186 | **0.33** |

`rho` is computed with **no accuracy constraint at all** — the baseline is
allowed to sacrifice arbitrary correctness — and it still reaches at most 14%,
34%, 11%, 73%, and 33% of the treatment's temperature-one breadth. On Graph and
Python the baseline's breadth does not move at any temperature in
[0.5, 2.0]: it is one execution, and heating the sampler produces invalid text
rather than a second verified mode. Countdown is the clearest illustration of
the cost side: from T=1 to T=2 accuracy falls .59 to .21 while breadth falls
too (.63 to .58).

This settles objection O2 in the direction that supports the manuscript. The
observed failure is **support loss, not sharpening**: the modes are not sitting
at low probability waiting for a hotter sampler, they are gone. Pre-committed
outcome **P4 does not fire**.

A second, unplanned observation: xGRPO's own breadth peaks near T=1.0–1.3 and
collapses by T=2.0 (Graph 2.41 to 0.44), because at that temperature the policy
stops producing validator-positive output at all. Both arms are therefore
measured on the same bounded quantity, and neither is advantaged by the sweep's
upper end.

**What this licenses in the paper.** A claim that the temperature-one operating
point is not an artifact, stated per domain, with the frontier figure
(`paper/figures/e72_decoding_frontier.pdf`) and the `rho` table as evidence. It
does not license any claim about the other five baseline families, which remain
untested until Tier 1/2 run.

Artifacts: `var/artifacts/e72_frontier_source_runs.json` (inventory),
`e72_decoding_frontier_cells.jsonl` (300 raw cells),
`e72_decoding_frontier_summary.json` (E1/E2 + gate),
`e72_decoding_frontier_figure_provenance.json`,
`paper/results/e72_decoding_frontier_live.md`.

## 11c. B3a launched (2026-07-31)

**Reframed as a compute-matched remove-one.** B3a is implemented as the frozen
E58 treatment with `online_canonical_replay_compute_only=1` rather than as a
separately configured "bonus without replay" arm. The same banks are retained,
scheduled in global round robin, teacher-forced, and traversed backward; only
the replay score derivative is zeroed before the optimizer. That makes it
simultaneously the count-based rare-outcome baseline the review asks for and the
first *empirical* remove-one control for verified replay — the component
`tab:component-ablations` currently supports by analytic argument alone. It also
means generation, scoring, and optimizer work match the treatment exactly, so no
compute-matching argument is needed.

Runtime variant `verified_first_replay_gradient_ablation`
(`ops/run_experiment.sh`), launcher
`ops/exp_scaling/launch_e72_b3a_replay_ablation.py`, protocol
`paper/preregistration/e72_b3a_replay_gradient_ablation_05b_20260731.md`.

**No dose screening, by design.** Every coefficient is inherited from the paired
xGRPO run's own recorded arguments, and the launcher refuses to submit if any
disagrees with the frozen E58 values. Tuning here would convert "does this
component matter" into "can this baseline be made to work"; the second question
is only worth asking if the first answers yes. This also removes Tier 1's
~340 GPU-hour screen from the critical path for this arm.

**Placement inherited.** Each run is pinned to the GPU model that trained its
paired xGRPO seed, so the three-way per-seed comparison is hardware-matched.

**Verified at launch.** A pilot (Graph, seed 43) confirmed at the telemetry
level: `canonical_replay_compute_only = 1`, applied replay score gradient L2 and
sum both exactly `0.0`, replay banks still available and scored
(`replay_score_passes = 2`), while discovery credit is live
(`novelty_advantage_rms = .097`, `semantic_shannon_separate_advantage_active = 1`,
open-set coefficient `.10`, new outcomes accumulating). The prereg makes a
nonzero applied replay gradient a discard condition rather than a
reinterpretation.

Cohort: 25 runs (5 domains x seeds 43--47), 12 passes, ~280 GPU-hours.

**Also launched alongside:** frontier stage B (150 cells, K=32 at
T in {1.0, 1.3, 1.6}, for the coverage-budget curve by exact subsampling) and
base-model reference cells (60 cells). Base cells are registered as one arm per
(domain, GPU model) — `base_node105` / `base_node302` — because the frozen pass-0
rows differ between GPU models, and they are gated against those pass-0 rows
rather than against a terminal row. The per-GPU pass-0 split is itself
informative: Graph `distinct@8` is `.713` on one GPU model and `.523` on the
other from identical weights and draw seeds.

## 12. Changelog

- 2026-07-31 — initial design drafted against the E70/E71 completed surface.
  Facts verified at drafting time: terminal exports present for all 50 headline
  checkpoints (`saved_models/step_04609`); intermediate weights not retained
  (`export_steps=0`); `train/entropy` logged for every update in both arms
  (Graph .0005 vs .5914, PantryPlan .0087 vs .3726 at seed 43); ~8.7 h wall for a
  12-pass Graph run on one A5000; 2.4 T free on `/n/fs/similarity`; `mltheory`
  idle at 5 nodes × 10 GPUs.
