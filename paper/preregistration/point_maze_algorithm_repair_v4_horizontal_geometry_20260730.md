# PointMaze v4: horizontal-geometry trainability repair

**Status:** FROZEN AFTER THE V3 VIABILITY STOP AND BEFORE V4 ROUTE EXECUTION
OR MODEL SAMPLING  
**Date:** 2026-07-30  
**Role:** Disclosed tertiary engineering repair; excluded from the original
80-cell estimator.

## Sealed antecedent

V3 corrected the train/dev/eval orientation distribution and passed executable
admission. Its frozen development viability gate produced 139/256 verified
rollouts (0.54296875), narrowly above the unchanged 0.50 trainability ceiling.
The rotation-0 medium cross and rotation-2 hard block rows each verified
64/64, while the rotation-1 and rotation-3 rows verified 1/64 and 10/64.
Thus orientation balance alone is insufficient: horizontal tasks remain
saturated.

The v3 threshold is not amended and no algorithm coefficient changes.

## Frozen v4 intervention

V4 retains exact within-split orientation balance:

- train: eight rows, two at each rotation;
- development: four rows, one at each rotation;
- evaluation: four rows, one at each rotation.

Horizontal training/evaluation positions use new 13x13 obstacle families:
central block, wide block, bar, and diamond. Their reset/goal cells are
`(6,1)` and `(6,11)`, bounds are ±6.5, and certified upper/lower programs use
14 lateral-detour pulses, 40 forward pulses, and 17 return pulses before
rotation. Vertical positions retain previously defined 9x9 or 11x11
geometries. Exact family/rotation executable tasks remain disjoint across
splits.

The intervention targets geometry difficulty only. The model, public-state
interface, action repeat, action alphabet, horizon, verifier, optimizer
settings, viability threshold, and MaxEnt mechanism are unchanged.

## Gates

Before model sampling, all 32 routes across 16 maps must pass the same official
networkless executable admission, 3,200 perturbation replays, hash checks,
failure mutations, split-disjointness checks, and throughput floor.

If admission passes, the same development-only gate samples 64 trajectories
per prompt with seed 76550 and K=16. It requires exactly 256 attempts, overall
verified rate in [0.02,0.50], at least two K=16 prefix-success prompts, and at
least two multimode prompts. Failure stops v4; passing authorizes only one
new one-seed development pair.
