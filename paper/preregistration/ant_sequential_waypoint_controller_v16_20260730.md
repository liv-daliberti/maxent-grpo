# Ant sequential-waypoint controller v16

**Status: FROZEN AFTER V15 ROUTE FAILURE AND BEFORE V16 TRAINING OR EVALUATION — 2026-07-30**

## Antecedent and failure mechanism

The admitted v11 controller passes 96 fresh single-cell waypoint episodes, but
the unchanged v12 cumulative-target executor does not reliably turn after
multiple same-direction waypoints. On the source-only v15 route diagnostic,
the lower eight-command fixture succeeded while the upper fixture reached its
first two north waypoints and then failed every east segment. No language
model was sampled on v15.

V16 repairs the controller mechanism while leaving the v15 13x13 map, central
3x3 obstacle, reset, goal, action alphabet, four-unit waypoint distance,
0.45 waypoint radius, 400-step segment horizon, and two eight-command route
fixtures unchanged.

## Frozen training intervention

- initialize from exact admitted-v11 model SHA-256
  `e6d202bd525be5469135b35b63b2bf0884459cc630b71e76f8c49e86dcb8f913`;
- reset the PPO optimizer;
- 2,000,000 `AntMaze_UMaze-v5` transitions, 8 workers, seed 73016;
- learning rate 1e-6, unchanged `[256,256]` policy and observation/action API;
- four fixed 11x11 training maps whose only interior walls are paired
  peripheral corners;
- the complete 12-pattern cardinal curriculum consisting of every initial
  direction followed by the same, left-perpendicular, or right-perpendicular
  direction, with each direction repeated twice; and
- a hard 400-step limit for every constituent waypoint.

Training may load the exact v11 weights and aggregate v15 failure mechanism
(`multi-waypoint turns fail`). It may not load the v15 map, v15 trajectory
coordinates, either v15 route program, a language prompt, a language-model
sample, a MaxEnt outcome, or a Dr.GRPO outcome.

## Frozen fresh sequential evaluation

Evaluate deterministic policies on four new 15x15 maps with distinct paired
peripheral walls. The 16 evaluation patterns enumerate every cardinal first
direction, both perpendicular second directions, and both perpendicular third
directions. Segment lengths are 2, 3, and 2 commands, respectively, so the
seven-command evaluation trajectories are not members of the four-command
training curriculum. Use evaluation seed base 7,073,016, yielding 64 episodes.

The unchanged base checks require at least 90% overall success, at least 75%
for every sequence pattern, at least 75% on every map, at most 10% unhealthy
termination, finite metrics, and median successful duration at most 300
steps. For this sequential repair, the duration bound is prospectively
replaced by 1,800 total simulator steps because an episode contains seven
separately bounded waypoints.

Only a passing controller may be rebound to the already frozen v15 map and
rerun through its real-route and 2,400-perturbation admission. A failure stops
v16. Original v12 AntMaze results remain immutable and are reported
separately.
