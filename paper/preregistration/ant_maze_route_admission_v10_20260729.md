# AntMaze v10 fresh-map route admission

**Status: FROZEN DURING V10 TRAINING, BEFORE ITS CONTROLLER OUTCOME OR ANY V10 ROUTE EXECUTION — 2026-07-29**

This gate may run only if the separately frozen v10 local-waypoint controller
gate passes. At this protocol freeze, v10 is training as job `30193111`; no
v10 controller receipt, route map, route trajectory, or language-model output
exists. The failed v9 controller remains failed, and its unexecuted 9x9 route
slate is not reused.

The held route-job identity must bind the exact terminal v10 receipt, model,
and training identity before release. The executor retains four-unit relative
compass targets, a 0.45 waypoint radius, 400 simulator steps per token, zero
reset noise, and no map input to the controller.

## Frozen route slate

Materialize 12 unique eleven-by-eleven maps: four train, four development,
and four evaluation. Every map has central wall `[5,5]`, reset `[5,4]`, goal
`[5,6]`, and one of the first 12 nonempty lexicographically enumerated subsets
of these eight safe peripheral cells:

`(1,1), (2,1), (3,1), (4,1), (5,1), (6,1), (7,1), (8,1)`.

The 11x11 dimension is fingerprint-disjoint from all prior 7x7, 8x8, 9x9,
and 10x10 controller/route maps. Reset seeds are exactly `107300..107311`.
Map order fixes the four/four/four train/development/evaluation split.

Every specification freezes `AntMaze_UMaze-v5`, sparse terminal reward,
compass tokens `N, NE, E, SE, S, SW, W, NW`, 4--16 tokens, environment goal
threshold 0.5, x=0 upper/lower topology gates with spans `[1,10]` and
`[-10,-1]`, upper witness `N E E S`, and lower witness `S E E N`.

Each witness is attempted once in materialization and once in the independent
audit. The gate passes only if all 48 real executions validate, both topology
keys remain distinct on every map, 2,400 deterministic trajectory
perturbations preserve identity, adversarial parser/identity/near-miss checks
reject, all split fingerprints are disjoint, persistent-worker throughput is
at least 0.15 routes/second, and every source/runtime/controller/data identity
is hash-bound.

Any failure stops v10. No map, seed, program, split, action repeat, radius, or
threshold may be replaced. A pass authorizes only a separately frozen
three-node exact-slate replay before any 0.5B language-model sample.
