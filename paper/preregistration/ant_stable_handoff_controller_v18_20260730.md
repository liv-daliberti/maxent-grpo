# AntMaze harder-task repair: stable-handoff controller v18

**Status:** FROZEN BEFORE V18 OPTIMIZATION  
**Date:** 2026-07-30  
**Role:** Secondary post-outcome engineering repair; excluded from the original
80-cell estimator.

## Sealed antecedent

Controller v17 completed its immutable 96-sequence gate and failed: overall
success was 21/96 (0.21875), minimum map success was 0.125, minimum pattern
success was zero, and unhealthy termination was 0.28125. Conditional segment
success was 0.885 on the first waypoint but 0.729 on the second. The immediate
position-only target switch therefore remains a concrete handoff-instability
mechanism rather than evidence that the admitted local locomotor cannot move.

V18 initializes from the exact sealed v17 checkpoint
`7a964daa7ebc02d52e4717e62ec7d02eed70d5b3155d4910728e24a5e10bbf0d`.
The v17 receipt and diagnosis are disclosed adaptive-engineering antecedents.

## Frozen intervention

- PPO optimizer reset; seed `73018`; eight CPU workers.
- 6,000,000 transitions at learning rate `2e-7`.
- Four generic 15x15 training maps; no v15 geometry.
- Waypoint distance 4.0, positional radius 0.45, segment budget 400.
- A waypoint switches only when planar speed is at most 1.0 in addition to
  satisfying the positional radius.
- Near-target planar speed is penalized to train braking before a handoff.
- The curriculum has 256 single-waypoint anchors, two copies of every one of
  the 64 ordered heading pairs, and every repeated-heading/next-heading triple.
- No result-dependent early stopping or checkpoint selection is allowed.

The v15 map, v15 route slate, language prompts or samples, MaxEnt outcomes,
and Dr.GRPO outcomes do not enter optimization.

## Fresh immutable gate

The gate has 24 new length-eight patterns on four new 19x19 generic maps, for
96 episodes. Patterns are disjoint from the length-one-to-three training
curriculum and from v17's development patterns. Reset noise is 0.1 and the
evaluation seed offset is 11,000,000. Training never accesses gate
trajectories.

All checks must pass:

- exactly 96 episodes;
- overall sequence success at least 0.90;
- success at least 0.75 for every pattern;
- success at least 0.75 on every map;
- unhealthy termination at most 0.10;
- median successful segment length at most 300;
- every recorded arrival has planar speed at most 1.0;
- all final distances and speeds finite.

Failure stops v18. Passing authorizes only a separately frozen worker binding
and executable admission of the unchanged v15 harder route slate.
