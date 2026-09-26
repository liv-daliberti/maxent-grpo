# AntMaze v9 fresh-map route admission

**Status: FROZEN DURING V9 TRAINING, BEFORE ITS CONTROLLER OUTCOME OR ANY V9 ROUTE EXECUTION — 2026-07-29**

## Antecedent and information boundary

The maze-local v9 controller is training under the separately frozen protocol
`ant_waypoint_controller_v9_20260729.md` as Slurm job `30192731`. At this
protocol freeze, no v9 controller receipt, v9 model, v9 route map, v9 route
trajectory, or language-model output exists. The prior v8 controller passed
its open-plane gate but failed the first fixture of its fixed 7x7 route slate;
that outcome and all v8 maps remain final development evidence.

This protocol fixes the only route gate v9 may enter if and only if its sealed
local-waypoint controller gate passes. A held launch identity must bind the
terminal v9 receipt and model hashes before release. No failed controller may
enter this gate.

## Frozen waypoint executor

Each language token creates one target exactly four world units from the Ant's
current x-y position in the token's compass direction. At every simulator
step, the v9 policy receives the ordinary AntMaze observation concatenated
with the clipped two-coordinate relative target divided by four. A token ends
when the target is within 0.45, the environment succeeds or terminates, or 400
simulator steps elapse. The controller receives no map geometry.

The executor may only load the exact v9 receipt and model bound in the held
route-job identity. Configuration checks may validate syntax, hashes, receipt
fields, and observation shapes but may not execute a route map.

## Frozen admission slate

Materialize 12 unique nine-by-nine maps: four train, four development, and
four evaluation. Every map has the central wall `[4,4]`, reset `[4,3]`, goal
`[4,5]`, and one of the first 12 nonempty lexicographically enumerated subsets
of these eight safe peripheral cells:

`(1,1), (2,1), (3,1), (4,1), (5,1), (6,1), (7,1), (1,2)`.

The 9x9 dimensions make every map fingerprint-disjoint from all prior 7x7 v5
and v8 route maps and all 8x8 v9 local-waypoint evaluation maps. Reset seeds
are exactly `97300..97311`. Map order fixes the split: first four train, next
four development, final four evaluation.

All specifications freeze:

- `AntMaze_UMaze-v5`, sparse reward, reset noise zero;
- compass tokens `N, NE, E, SE, S, SW, W, NW`, with no `STOP`;
- 4--16 language tokens and 400 simulator steps per token;
- environment goal threshold 0.5;
- upper and lower x=0 topology gates with spans `[1,10]` and `[-10,-1]`;
- upper fixture `N E E S`; and
- lower fixture `S E E N`.

Each fixture is attempted once during materialization and once during the
independent audit. Any failure stops the slate. No map, seed, route program,
action repeat, waypoint radius, goal threshold, or split may be replaced.

## Decision

The gate passes only if:

1. all 24 materialization executions and all 24 independent audit executions
   reach the goal and produce exactly two distinct topology keys per map;
2. 100 deterministic sub-cell perturbations per route preserve each key
   (2,400 replays, zero mismatch);
3. malformed programs, near-miss success claims, identity mutations, and
   cross-route collisions are rejected;
4. all train/development/evaluation fingerprints are disjoint;
5. persistent-worker throughput is at least 0.15 real routes per second; and
6. every environment, controller, model, receipt, runtime, data, protocol,
   execution source, and source-tree identity is hash-bound.

A pass authorizes only a separately frozen cross-node determinism replay of
this exact slate. That replay must pass before any Qwen2.5-0.5B AntMaze
viability sample. A failure stops v9; no post-outcome map or fixture
substitution is allowed.
