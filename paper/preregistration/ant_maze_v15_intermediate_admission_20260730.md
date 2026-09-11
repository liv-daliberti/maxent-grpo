# AntMaze v15 intermediate-difficulty admission

**Status: FROZEN AFTER V14 ROUTE FAILURE AND BEFORE V15 ROUTE EXECUTION OR MODEL SAMPLING — 2026-07-30**

The original v12 AntMaze slate used four high-level decisions around one
blocked cell and was already solved at pass zero. The prospective v14-hard
slate used fourteen decisions around a 5x5 obstacle; its first official route
failed under the unchanged anchored controller, before any language model was
sampled.

V15 brackets those source-feasibility outcomes without selecting a map from
model behavior. It uses a 13x13 map, a fixed central 3x3 obstacle, the same
v12 controller and 400-step action repeat, and two eight-decision route
fixtures:

- `N N E E E E S S`
- `S S E E E E N N`

The reset is `(6, 4)`, the goal is `(6, 8)`, the route-length range is
8–20 actions, the reset-seed base is 108500, and all split rotations and
peripheral-wall variants remain those of the source-bound AntMaze generator.

Admission requires all 24 fixture routes to replay under the real official
worker, two distinct topology keys per map, all 2,400 topology perturbations
to preserve their keys, exact controller/runtime/source binding, mutation
rejection, split disjointness, and the unchanged throughput floor.

Only after admission may the frozen 0.5B warm start be sampled. Viability must
show nonzero verified completion and less than 90% verified completion. A
failure stops v15; it does not authorize another post-outcome map
substitution. The original v12 campaign remains immutable and is reported
separately.
