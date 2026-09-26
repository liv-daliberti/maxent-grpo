# PointMaze balanced-v5 five-seed online comparison

**Status: FROZEN BEFORE THE V5 WARM-START QUALIFICATION OUTCOME OR ANY V5
ONLINE JOB — 2026-07-30**

This cohort is a disclosed replacement for the mis-tuned PointMaze geometry
cohort and is excluded from the original 80-cell estimator until separately
reported. It launches only if the frozen orientation-balanced v5 warm-start
qualification passes.

## Cohort

- Shared checkpoint: exact `point_maze_interactive_warmstart_v5_balanced`.
- Train data: the eight-row, 2/2/2/2 orientation-balanced v3 train split.
- Evaluation data: the untouched four-row v3 evaluation split, one row per
  orientation. No evaluation row entered SFT, viability sampling, or online
  updates.
- Arms: compute-matched Dr.GRPO and verified-first global replay-canonical
  MaxEnt.
- Seeds: 76611, 76612, 76613, 76614, and 76615, shared by arms.
- Twelve prompt passes, 96 optimizer updates, 16 live trajectories per update.
- Learning rate `2e-7`, policy/evaluation microbatch 16, context cap 1536.
- Evaluate every two updates, including update zero, with four
  checkpoint-invariant common-random-number draws and K=8.

Both arms execute identical live-policy slots and replay-decision forward
slots. The control computes the MaxEnt/replay diagnostics but applies zero
derivative from them. The treatment applies the already frozen verified-first
global replay-canonical objective. Both use only the official terminal binary
reward and online-discovered canonical identities.

## Terminal audit

The audit requires all ten jobs to complete 96 updates and all 49 evaluation
coordinates, exact arm-paired compute traversal, finite metrics, no action
support escapes, zero treatment derivative in the control, applied treatment
mechanisms when their raw signals are nonzero, immutable hashes, official
state replay of every saved transition, and the exact untouched evaluation
family order. No efficacy threshold is imposed.
