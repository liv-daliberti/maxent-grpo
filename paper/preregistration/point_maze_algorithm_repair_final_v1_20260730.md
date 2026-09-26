# PointMaze algorithm-repair five-seed final v1

**Status:** FROZEN BEFORE PAIR OUTCOME AND FINAL TRAINING  
**Date:** 2026-07-30  
**Role:** Secondary post-outcome repair; excluded from the original 80-cell
estimator.

## Authorization

The final may launch only if
`point_maze_algorithm_repair_pair_v2r5_audit.json` passes with decision
`eligible_for_point_maze_algorithm_repair_v2_five_seed_final`. The audit is a
one-seed algorithmic qualification on the development split. A failed or
missing audit launches no final jobs.

## Frozen design

- Model: exact `point_maze_interactive_warmstart_v3`.
- Data: exact `point_maze_algorithm_repair_v2`.
- Train split: the eight train-only balanced maps.
- Evaluation split: `eval`, which is not loaded by viability, qualification,
  or the paired development run.
- Seeds: `76531, 76532, 76533, 76534, 76535`.
- Arms: compute-matched Dr.GRPO and online verified MaxEnt.
- Prompt passes: 12.
- Optimizer updates: 96.
- Rollouts per prompt: 16.
- Policy microbatch size: 16.
- Learning rate: `2e-7`.
- Evaluation: update zero and every two updates, four K=8 draws, using
  checkpoint-invariant common-random-number schedules within each seed.
- Resume and checkpoint selection: disabled.

The treatment retains semantic coefficient 0.10, novelty 0.50, and verified
replay 0.10. The control performs the same fixed-shape policy and replay
forward work with both derivatives disabled.

## Terminal audit

All ten jobs must complete successfully and produce exactly 96 training rows,
49 evaluation coordinates, 96 state-replay rows, and terminal receipts.
Every stored interactive transition is replayed through the official worker.
The audit also requires:

- exact source, execution, model, data, protocol, seed, arm, and scheduler
  identities;
- train/evaluation split binding to `train` and untouched `eval`;
- no evaluation feedback to training;
- finite metrics and exact action support;
- matched fixed policy/replay traversal between arms for every seed;
- exact-zero applied exploration and replay gradients in the control;
- nonzero applied exploration and replay gradients in every eligible treatment
  update;
- aggregate verified train rate in [2%, 50%] for every cell;
- at least 10 task-gradient updates in every cell; and
- no resume or post-outcome cell substitution.

Only a passing ten-cell audit makes this secondary repair result eligible for
reporting. It does not replace or alter the immutable original PointMaze
result.
