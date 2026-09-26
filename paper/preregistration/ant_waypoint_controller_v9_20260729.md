# Ant maze-local waypoint controller v9

**Status: FROZEN BEFORE V9 TRAINING OR EVALUATION — 2026-07-29**

## Antecedent

The maze-blind v8 controller passed its open-plane gate but failed the first
fixture of its separately frozen fresh-map route slate (job `30188228`,
`ant_v8_admission_train_00 upper`). No map or program was replaced, the failed
map was not rerun, and no trajectory was exposed. V8 remains ineligible.

This v9 attempt tests a specific mechanism suggested by that boundary: retain
the four-unit low-level waypoint policy, but continue it only on local free-cell
movements from training maps so it learns collision-adjacent locomotion before
another fresh route slate exists.

## Frozen training intervention

- initialization: exact v8 model
  `77e780dfff1147bc2f542c1761f6efeaa2305fd5ddb625244ebf8e82b3fa871d`;
- reset PPO optimizer;
- 5,000,000 `AntMaze_UMaze-v5` transitions, 8 workers, seed `73009`;
- learning rate `5e-6`, 400-step horizon, unchanged `[256,256]` policy,
  PPO objective, reward, four-unit relative-target observation, and 0.45
  waypoint success radius;
- balanced cyclic sampling across all eight compass headings; and
- each episode is exactly one adjacent free-cell target. Diagonals are allowed
  only when the target and both orthogonal corner cells are free.

The only training maps are the four 7x7 identities already designated as
training before the failed v8 route job: central wall `[3,3]` plus respectively
one wall at `[1,1]`, `[2,1]`, `[3,1]`, or `[4,1]`. Position noise is 0.1 during
training. No complete route fixture is used as a demonstration or reward.

Training may not load a v8 route trajectory, a v8 development/evaluation map,
any v9 development result, any future v9 route map, a language prompt, a
MaxEnt outcome, or a Dr.GRPO outcome.

## Frozen fresh local-waypoint evaluation

The four development maps are eight-by-eight and therefore fingerprint-
disjoint from all prior 7x7 route maps. Their interior walls are exactly:

1. `[3,3]` and `[1,6]`;
2. `[3,4]` and `[6,1]`;
3. `[4,3]` and `[1,2]`; and
4. `[4,4]` and `[6,5]`.

Evaluation uses zero position noise and seed base `4073009`. For each map it
runs three frozen free-cell edges per heading, for 96 episodes total and 12 per
heading. No evaluation episode enters training.

The gate passes only if:

- overall success is at least 0.90;
- every heading succeeds on at least 9/12 episodes (0.75);
- every map succeeds on at least 18/24 episodes (0.75);
- unhealthy termination is at most 0.10;
- median successful duration is at most 300 steps; and
- all metrics are finite and all hashes/counts match.

A pass authorizes exactly one separately preregistered, new-fingerprint v9
route gate. It does not authorize an AntMaze language-model sample or main
cohort. A failure stops v9. Thresholds, maps, seeds, and edge selection cannot
be repaired after evaluation.
