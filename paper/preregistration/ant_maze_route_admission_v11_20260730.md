# AntMaze v11 fresh-map route admission

**Status: FROZEN DURING V11 TRAINING, BEFORE ITS CONTROLLER OUTCOME OR ANY V11 ROUTE EXECUTION — 2026-07-30**

This gate may run only if the separately frozen v11 local-waypoint controller
gate passes. At this freeze, v11 is training as job `30198291`; no v11
controller receipt, route trajectory, or language-model output exists.

The route slate is exactly the v10 11×11 slate frozen before v10's outcome.
V10 failed its local controller gate, so that route job was never submitted:
the v10 route identity, data root, audit, and language-model viability receipt
remain absent. Reusing this wholly unexecuted slate changes only the controller
binding and does not select a map or route from an observed outcome.

## Frozen route slate

Materialize 12 unique 11×11 maps: four train, four development, and four
evaluation. Every map has central wall `[5,5]`, reset `[5,4]`, goal `[5,6]`,
and one of the first 12 nonempty lexicographically enumerated subsets of
`(1,1), (2,1), (3,1), (4,1), (5,1), (6,1), (7,1), (8,1)`.
Reset seeds are exactly `107300..107311` and map order fixes the split.

Every specification freezes `AntMaze_UMaze-v5`, sparse terminal reward,
compass tokens `N, NE, E, SE, S, SW, W, NW`, 4–16 tokens, 400 simulator
steps per token, goal threshold 0.5, upper topology witness `N E E S`, and
lower topology witness `S E E N`.

Each witness is executed once in materialization and once in the independent
audit. The gate passes only if all 48 real executions validate, both topology
keys remain distinct on every map, 2,400 deterministic trajectory
perturbations preserve identity, adversarial parser/identity/near-miss checks
reject, split fingerprints are disjoint, persistent-worker throughput is at
least 0.15 routes/second, and every source/runtime/controller/data identity is
hash-bound.

Any failure stops v11. No map, seed, program, split, action repeat, radius, or
threshold may be replaced. A pass authorizes only a separately frozen
three-node exact-slate replay before any 0.5B language-model sample.
