# E82: fixed semantic MaxEnt added to verified replay, Falcon3-1B

**Frozen before submission on 2026-08-07.**

## Question

Does the E81 result, whatever it turns out to be, hold in a second model
family? E82 is the Falcon3-1B replication of E81: the same fixed,
validator-gated semantic MaxEnt term, at the same fixed coefficient, added to
the same uniform verified-likelihood replay, evaluated against the same
paired-difference estimand.

E79 established the two reference arms on this cohort: compute-matched Dr.GRPO
(`control`) and uniform verified-likelihood replay (`replay`). E82 adds a third
arm that is the E79 `replay` arm plus one additional term and nothing else.
E82 does not re-run either E79 arm. Both are inherited unchanged as the frozen
comparators.

E82 is registered before E81 has a reportable endpoint. It is a replication by
construction, not a follow-up conditioned on E81's outcome: nothing in this
protocol may be revised once E81's results are known.

## Design

- Model: Falcon3-1B-Instruct at the pinned revision
  `28ba2251970a01dd1edc7ba7dad2eb71216ccfdf`, trained from the base model, not
  from any E79 checkpoint.
- Domains: Graph Coloring, Countdown, Python Factors, MathIR, and PantryPlan.
- Data: each domain's released 384-prompt training pool and fixed 128-prompt
  evaluation split, identical files to E79, presented through the same Falcon
  prompt-surface twins and per-domain generation budgets E79 used.
- Seeds: 55, 56, 57, 58, and 59, paired within domain against the E79 arms.
- Training: exactly eight passes, hence 3,072 optimizer updates per run;
  group size 16, one PPO epoch, AdamW at a constant 2e-7 with zero warmup,
  betas (0.9, 0.999), no weight decay, gradient clipping at 1, rollout
  temperature 1, top-p 1, and beta_KL = 0.
- Evaluation and resumable model checkpoints: every 192 updates, corresponding
  to passes 0, 0.5, 1.0, ..., 8.0. Pass 8 is the terminal endpoint.
- Placement: E79's placement is a deterministic function of domain and seed, so
  each E82 cell is pinned to the same physical node and GPU model as its E79
  pair and every paired difference is taken within one GPU model.

The new cohort has 5 domains x 1 arm x 5 seeds = 25 runs. Together with the
inherited E79 arms the analysis cohort is 75 runs.

PointMaze is excluded, as in E81. Its interactive trainer is a separate driver
with no semantic-advantage path. E82 makes no claim about PointMaze.

## Arms

### Inherited comparators, not re-run

`control` and `replay` are exactly the E79 runs recorded in
`var/artifacts/e79_falcon1b_aligned_verified_replay_jobs.json`.

### Verified replay plus fixed semantic MaxEnt (`semantic`)

Identical to the E79 `replay` arm — same passive verified bank, same capacity
of 16 observed modes per prompt, same one scheduled bank per optimizer update
in deterministic global round-robin order, same uniform verified-likelihood
replay objective at the same fixed weight 0.10, same live replay derivative —
plus one detached advantage term.

The term is bit-identical in definition to E81's. For row `i` of a candidate
group, let `a_i` be its canonical outcome key. A row is **eligible** if and
only if it is active, parseable, and validator-positive. Ineligible rows
receive exact zero and never enter the predictor's support, so an all-wrong
group is an exact no-op and no incorrect, unparseable, or failed output is ever
rewarded for being unusual.

For an eligible row, let the prompt-local open-set predictor be built from the
persistent per-prompt history of previously verified outcomes plus the
leave-one-out verified peers of the current group, with pseudocount 1 and one
structural unseen bucket. With `s_i` the surprisal of `a_i` under that
predictor, clipped at `C = 5`, the semantic advantage is

    A_sem_i = eta * (min(s_i, C) - E[min(s, C)]) / C,

where the expectation is taken under the same detached predictor. The centered
factor lies in [-1, 1], so `|A_sem_i| <= eta`. It is added to the Dr.GRPO task
advantage after that advantage's own centering, with no second centering step
and no outer clamp.

The coefficient is fixed at **eta = 0.10** for every domain, seed, and update:
the same reference dose E81 uses, carried over unchanged so the two families
test one intervention rather than two. E82 does not tune the coefficient, does
not re-select it against E81's outcomes, and does not identify an optimal
value. Because every task in this cohort uses a unit correct/incorrect reward
gap, the bound `|A_sem| <= 0.10` guarantees a verified response keeps at least
a 0.90 advantage margin over an incorrect one.

## Exact exclusions

As E81. The `semantic` arm hard-disables the one-time canonical novelty bonus,
the conditioned-bank balance KL, canonical-bank entropy shaping, the balance
actuator, adaptive or dual semantic coefficients, entropy controllers of every
kind, token entropy bonuses, sequence entropy, SEED, counterfactual proposals,
singleton escape, support-escape and proposal mechanisms, signed-surrogate xDr,
and reference KL. There is no reward for discovering an outcome for the first
time. No exhaustive support, evaluation outcome, desired mode count, or desired
entropy is available to training, scheduling, stopping, or checkpoint choice.

## Compute and code matching

The semantic term adds no rollouts, no additional generated tokens, no extra
forward or backward passes, and no additional loss term. Sampling budget,
optimizer-update count, replay traversal, and backward envelope are identical
to the E79 `replay` arm. The arms are not matched on the CPU-side cost of
maintaining the semantic predictor, which is not a gradient-carrying resource.

E82 runs from the E79 runtime snapshot with exactly two files replaced, the
same two E81 replaces: `src/oat_drgrpo/args.py`, which widens one validation
predicate so open-set semantic MaxEnt may compose with the uniform
verified-likelihood replay objective, and `ops/run_experiment.sh`, which adds
the `verified_replay_semantic_maxent` variant branch. Both changes are strictly
additive and unreachable from any E79 arm's configuration. The launcher walks
both trees and fails closed unless the divergence is exactly that patch set, so
E82 inherits E79's training code byte-for-byte on every path either E79 arm
executes.

E82 is a paired comparison against runs that were already executing when it was
registered, not a simultaneously randomized one. Model, data, splits, seeds,
schedule, decoding, placement, and training code are pinned identical; the
residual difference is the wall-clock window.

## Outcomes and estimands

At every registered half-pass checkpoint report, separately by domain:

- greedy pass@1;
- sampled mean correctness@8;
- sampled pass@8;
- mean distinct correct modes@8; and
- excess multiplicity, `distinct@8 - pass@8`.

The primary comparison is the paired seed difference `semantic - replay` at
pass 8 for `distinct@8` and `pass@8`. The secondary comparison is
`semantic - control` at pass 8. The trajectory summary is trapezoidal AUC over
the complete pass-0 through pass-8 half-pass grid.

Show all five paired seed differences and their mean and range.
Do not pool domains into one effect,
do not pool the two model families,
do not report a universal semantic-MaxEnt claim, and
do not select a best checkpoint. A breadth gain that arrives with a `pass@8` or
`mean@8` loss is reported as a trade, not as a win.

## Registered cross-family interpretation

The cross-family claim is registered now, before either cohort reports:

1. Both families show a `distinct@8` gain without correctness loss in most
   domains: semantic MaxEnt is reported as a model-family-general extension to
   verified replay.
2. The families disagree, or a family shows the effect only in some domains:
   reported as family- or domain-dependent, naming exactly where it held and
   where it did not, with the null cells at equal prominence. A Qwen-only
   effect is reported as a Qwen-only effect.
3. Neither family shows a gain: verified replay alone remains the method, and
   E81 and E82 are reported together as the registered negative result that
   retired the component.

Agreement in direction across two families is descriptive support, not a
significance test, and is reported as such.

## Mechanism telemetry

As E81. Every run logs, passively and without any effect on training: eligible
fraction; the effective semantic advantage's mean, min, max, and RMS; its
positive, negative, and zero fractions; the fraction with `|A_sem| > 0.05`; the
predictive probability quantiles; normalized predictive entropy; tracked prompt
and outcome counts; and the ratio of semantic advantage RMS to task advantage
RMS. Replay telemetry is unchanged from E79.

The `semantic` arm must show a finite nonzero applied semantic advantage on at
least one eligible update and exact zero on every ineligible row. These
measurements are descriptive and may not be used to re-scale, re-select, or
re-report E82's own coefficient.

## Integrity and failure policy

All 25 cells are submitted held from one hash-bound runtime snapshot and are
released only after their scheduler environments pass an exact audit. A
malformed environment, a snapshot diverging from the E79 snapshot outside the
two declared files, a missing or mismatched E79 pair, a wrong model revision, a
duplicate run directory, a non-finite loss, proposal or novelty leakage, a
traceback, or a missing pass-8 endpoint fails closed. Infrastructure
interruption may resume only from the same run's hash-bound checkpoint and
exact bank, semantic predictor, optimizer, data-cursor, and request-stream
state. No failed scientific run is silently replaced or excluded.
