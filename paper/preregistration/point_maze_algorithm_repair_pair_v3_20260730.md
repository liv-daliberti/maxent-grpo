# PointMaze orientation-balanced v3 calibration pair

**Status:** FROZEN BEFORE THE V2 TERMINAL AUDIT AND V3 VIABILITY OUTCOME  
**Date:** 2026-07-30  
**Role:** One-seed development calibration for a disclosed engineering repair;
excluded from the original 80-cell estimator.

## Preconditions

The v2 pair must first be terminal, but its pass/fail result does not select or
change this design. The v3 executable data audit and K=16 viability
qualification must pass. No v3 evaluation row or v3 training outcome may be
loaded before launch.

## Pair

- Model: exact frozen Qwen2.5-0.5B PointMaze velocity-state warm start.
- Arms: compute-matched Dr.GRPO control and verified-first global
  replay-canonical MaxEnt treatment.
- Common seed: 76541.
- Data: v3 eight-row orientation-balanced training split.
- Evaluation: v3 four-row orientation-balanced development split only.
- Twelve prompt passes, 96 optimizer updates, 16 rollouts per update.
- Learning rate 2e-7; binary official terminal reward.
- The treatment mechanism and control derivative-zero contract are unchanged.
- Checkpoint evaluation every two updates uses checkpoint-invariant common
  random numbers, four draws, and K=8.

The two jobs must traverse identical policy and replay-decision forward slots.
No efficacy threshold is imposed.

## Pair audit

Both cells must finish all 96 updates and 49 evaluation coordinates, have
verified rollout rate in [0.02, 0.50], and have at least ten updates with
nonzero task advantage. The treatment must apply both exploration and replay
mechanisms; the control must apply neither. All hashes, finite metrics, state
replays, common-random-number seeds, and scheduler identities must match.

Passing authorizes a separately frozen five-seed v3 final on the untouched
evaluation split. Failure stops v3.
