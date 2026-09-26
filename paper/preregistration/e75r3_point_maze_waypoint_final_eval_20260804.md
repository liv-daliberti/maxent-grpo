# E75R3 PointMaze untouched final evaluation

**Frozen:** 2026-08-04, before any E75R3 checkpoint was evaluated on the
`eval` split.

## Scope and immutable inputs

This is the one-shot final evaluation of E75R3.  The development result is
already complete.  No further Maze training, checkpoint selection, threshold
selection, or exclusion selection is permitted before this evaluation.

The evaluated set is exactly the three terminal update-64 checkpoints from
seed `88504`:

1. `grpo`;
2. `verified_first_global_replay_canonical` (current);
3. `verified_first_delayed_singleton_replay_canonical` (delayed).

The machine-readable plan records the checkpoint tree hashes, training receipt
and trace hashes, E75R3 data-identity hash, source snapshot, evaluator snapshot,
and this protocol hash.  An evaluation job fails closed if any bound input has
changed.  There are no run, map, family, trajectory, or outcome exclusions.
An execution-integrity failure invalidates the affected evaluation; it does not
authorize silent replacement or a second draw.

## Evaluation

Each frozen checkpoint is evaluated exactly once on all 64 maps of the
previously untouched E75R3 `eval` split.  Each map receives eight trajectories
under the fixed 64-decision horizon.  All arms use seed `88504`; the evaluator
therefore uses the same map ordering and the same per-map, per-trajectory,
per-decision sampling seeds across arms.  The evaluator's internal
`learning_round=0` denotes evaluation-only loading of the already frozen
update-64 checkpoint, not an untrained checkpoint.

The four per-map endpoints are `mean8`, `pass8`, `distinct8`, and
`modes_per_success`, exactly as emitted by the frozen E75R3 runner.  The primary
diversity comparisons are current minus GRPO and delayed minus GRPO on
`distinct8`.  `mean8` and `pass8` are success guardrails;
`modes_per_success` and delayed minus current are descriptive secondary views.
All endpoints and all three contrasts are reported regardless of sign.

## Paired analysis

Analysis is paired by the exact 64 map IDs.  It reports each arm's mean and the
raw mean of the 64 within-map differences for every endpoint.  Descriptive 95%
paired map-bootstrap intervals use 10,000 resamples with replacement.  Seeds
are fixed as `756400 + 100 * comparison_index + metric_index`, with comparisons
ordered current-GRPO, delayed-GRPO, delayed-current and metrics ordered
`mean8`, `pass8`, `distinct8`, `modes_per_success`.  Percentiles use linear
interpolation at 0.025 and 0.975.  These single-training-seed intervals quantify
map uncertainty only and are not treated as population-level hypothesis tests;
there is no adaptive multiplicity-dependent winner rule.

The analysis job runs only after all three evaluation jobs complete and first
rechecks the frozen plan, checkpoint, receipt, data, and output identities.
