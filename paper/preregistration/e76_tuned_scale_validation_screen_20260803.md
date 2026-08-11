# E76: validation-selected tuned-scale study

Frozen 2026-08-03, before any E76 outcomes were generated.

## Status and scope

E73 (Falcon3-1B) and E74 (Qwen2.5-3B) remain unchanged and are interpreted as
fixed-recipe transfer experiments. E76 is a separate, explicitly tuned study;
it must not be pooled with E73/E74 as though their hyperparameters had been
selected in the same way.

E76 uses Graph Coloring and PantryPlan as sentinel domains. They were chosen
before tuning because the frozen-recipe curves exhibit the two failure modes
the study must resolve: an early breadth peak followed by degradation, and a
sharp correctness collapse. No E76 choice is made using the existing reported
test splits.

## Sealed data split

For each domain, the existing 384-example training pool is deterministically
partitioned into 320 training examples and 64 validation examples. A row is
ranked by SHA-256 over a fixed salt, domain name, and canonical JSON. The first
64 hashes form validation and the other 320 form training. The materializer
records source fingerprints, row hashes, and prompt-identity disjointness in
`var/artifacts/e76_tuned_scale_splits.json`.

Stages A and B train only on the 320-example split and evaluate only on the
64-example validation split. The original ModeBench evaluation directories are
not passed to those jobs. Stage C is launched only after both selection files
are written; it trains on all 384 original training examples and evaluates the
unchanged reported test split.

## Stage A: optimizer and stopping

- Models: Falcon3-1B-Instruct and Qwen2.5-3B-Instruct.
- Domains: Graph Coloring and PantryPlan.
- Arms: compute-matched Dr.GRPO and full x-MODE.
- Learning rates: `5e-8`, `1e-7`, `2e-7`.
- KL coefficients: `0`, `0.01`.
- Seed: 53.
- Maximum depth: six passes over the 320-example tuning pool.
- Validation cadence: every 80 optimizer steps (one quarter pass).
- Eligible stopping points: steps 320, 640, 960, 1280, 1600, and 1920 only.

The selected learning-rate/KL pair is common to both arms and both domains
within a model. At each eligible checkpoint, define for each arm

`utility = (distinct@8 - initial distinct@8) + (pass@8 - initial pass@8)`.

A checkpoint is feasible only if both arms retain pass@8 within 0.02 absolute
of their own step-zero validation value. For each model/domain/configuration,
select the feasible checkpoint with largest mean utility across the two arms;
ties prefer larger mean pass@8 and then the earlier checkpoint. If no
checkpoint is feasible, select by mean pass@8, then mean distinct@8, then the
earlier checkpoint and mark the fallback.

For each model, select the configuration with the largest mean of its two
domain utilities. Ties prefer the larger worst-domain utility, then smaller KL,
then smaller learning rate. The two selected domain checkpoints are the
model-specific/domain-specific stopping horizons used in later stages, and are
common to all later arms.

## Stage B: replay form and strength

- Seed: 54.
- Data: the same sealed 320/64 tuning split.
- Optimizer and horizons: fixed by Stage A.
- Arms: compute-matched Dr.GRPO; full x-MODE at replay strengths 0.05, 0.10,
  and 0.20; rehearsal-only verified replay at the same strengths.

For full x-MODE, replay strength sets both replay-gradient alpha and replay-mass
alpha; discovery coefficients remain at the registered x-MODE values. For
rehearsal-only replay, discovery, rarity weighting, and uniform mode balancing
remain disabled and the same replay-gradient alpha is varied.

For each model/domain/replay variant, a dose is feasible if its terminal
validation pass@8 is no more than 0.02 below the terminal Dr.GRPO pass@8.
Among feasible doses, choose the largest distinct@8, then pass@8, then the
smaller dose. If none is feasible, choose pass@8, then distinct@8, then the
smaller dose and mark the fallback. Dose is selected separately for each
model/domain and replay variant.

## Stage C: untouched-test confirmation

- Seeds: 55, 56, 57.
- Training data: the original full 384-example training pool.
- Evaluation data: the original untouched reported test split.
- Arms: compute-matched Dr.GRPO, selected-dose full x-MODE, and selected-dose
  rehearsal-only replay.
- Learning rate, KL coefficient, and stopping horizon: fixed by Stage A.
- Replay doses: fixed by Stage B.
- Reported endpoint: the fixed terminal checkpoint only; no test-curve
  selection or test-dependent stopping is permitted.

The primary tuned-scale comparison is paired full x-MODE versus Dr.GRPO on
terminal distinct correct modes at 8 samples, with pass@8 reported alongside
it. Rehearsal-only is a mechanistic comparator motivated by B1a/B1b. The study
is evidence about tuned policies on the two sentinel domains, not a replacement
for the broader frozen-recipe transfer result.

## Gating and failure policy

Stage B is not submitted unless every Stage A job reaches its registered target
and the deterministic selector succeeds. Stage C is held to the analogous
Stage B gate. A missing or malformed evaluation, an ambiguous metric value, or
an incomplete job fails closed; it does not silently remove a cell from the
selection. Scheduler dependencies are `afterany` so the controller runs even
when a job fails, but the selector itself prevents advancement.

All E76 jobs are submitted behind the E73/E74 jobs active at initial launch so
the current fixed-recipe cohorts retain priority and placement. Falcon jobs use
the A5000/A6000 pool; Qwen-3B jobs use node302's A100s. Source and operation
scripts are snapshotted at submission and pinned in every E76 environment.
