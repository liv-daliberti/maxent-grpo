# AntMaze v8 fresh-map route admission

**Status: FROZEN BEFORE THE FIRST V8 MAZE EXECUTION — 2026-07-29**

## Antecedent and information boundary

The maze-blind v8 waypoint controller passed its preregistered 96-episode
open-plane gate: 91/96 successes, minimum heading success 10/12, unhealthy
termination 1/96, and median successful duration 147 steps. Its receipt and
model are immutable:

- receipt SHA-256:
  `fc2961d74a1fdfc8f3d2c1936ea61041cff65333ada0e6ebcd571b44919b18f1`;
- model SHA-256:
  `77e780dfff1147bc2f542c1761f6efeaa2305fd5ddb625244ebf8e82b3fa871d`.

No v8 maze map, maze trajectory, route outcome, or language-model output has
been observed. Controller development used only open-plane Ant-v5. The stopped
v5 route and cross-node outcomes are development evidence but their map shapes,
reset seeds, and long macro fixtures are excluded from this slate.

## Frozen waypoint executor

Each language token creates one target exactly four world units from the Ant's
current x-y position in the token's compass direction. At every simulator step,
the policy receives the ordinary Ant observation concatenated with the clipped
two-coordinate relative target divided by four, exactly as in v8 training. A
token ends when the target is within 0.45, the environment succeeds or
terminates, or 400 simulator steps elapse. No map geometry is exposed to the
controller.

One separate adjacent-cell development fixture may test only worker plumbing;
its map identity and seed are excluded from the admission slate. It cannot
alter the frozen executor, programs, maps, seeds, or thresholds below.

## Frozen admission slate

Materialize 12 unique seven-by-seven maps: four train, four development, and
four evaluation. Every map has the central wall `[3,3]`, reset `[3,2]`, goal
`[3,4]`, and one of the first 12 nonempty lexicographically enumerated subsets
of these eight safe peripheral cells:

`(1,1), (2,1), (3,1), (4,1), (5,1), (1,2), (1,3), (1,4)`.

These shapes are disjoint from the v5 slate, whose optional walls were only in
column 5. Reset seeds are exactly `87300..87311`. The split is determined by
map order: first four train, next four development, final four evaluation.

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
action repeat, target radius, or success threshold may be replaced or changed.

## Decision

The gate passes only if:

1. all 24 independent real-route replays reach the goal and produce exactly
   two distinct topology keys per map;
2. 100 deterministic sub-cell perturbations per route preserve each key
   (2,400 replays, zero mismatch);
3. malformed programs, near-miss success claims, identity mutations, and
   cross-route collisions are rejected;
4. all train/development/evaluation fingerprints are disjoint;
5. persistent-worker throughput is at least 0.15 real routes per second; and
6. every environment, controller, model, receipt, runtime, data, and source
   identity is hash-bound.

A pass authorizes only the separately frozen v8 cross-node determinism gate.
That later gate must pass before any Qwen2.5-0.5B AntMaze viability sample. A
failure stops v8; no post-outcome map or fixture substitution is allowed.
