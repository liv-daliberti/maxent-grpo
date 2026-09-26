# Ant maze-local waypoint controller v11

**Status: FROZEN AFTER THE TERMINAL V10 RECEIPT AND BEFORE V11 TRAINING OR EVALUATION — 2026-07-30**

## Antecedent

V10 completed its frozen two-million-transition budget and failed its unchanged
local-waypoint gate. Overall success was 89/96 (0.9270833), every map was at
least 21/24, unhealthy termination was zero, and median successful duration
was 160 steps. The sole failed criterion was heading balance: heading index 3,
grid delta `(+1,-1)` (southeast), succeeded on 8/12 trials. Heading index 1,
the northeast stratum remediated by v10, improved from 4/12 to 10/12. No v10
route map was executed and no language model was sampled.

V10 remains failed. V11 is a new development controller, not a reinterpretation
of that outcome.

## Frozen training intervention

- initialize from exact failed-v10 model SHA-256
  `fffb66696aa4435c83b8bed854834ce82ccb01a9a1ce9dfe22e813c7e94a7b94`;
- reset the PPO optimizer;
- 2,000,000 `AntMaze_UMaze-v5` transitions, 8 workers, seed `73011`;
- learning rate `1e-6`, 400-step horizon, unchanged `[256,256]` policy,
  observation, reward, four-unit target distance, and 0.45 success radius;
- fixed heading schedule
  `SE,NE,SE,N,SE,NE,SE,E,SE,NE,SE,S,NE,SW,W,NW`, giving southeast 6/16,
  northeast 4/16, and each other heading 1/16; and
- the unchanged four preregistered 7×7 training maps and all legal local
  free-cell edges.

Training may load only the exact v10 weights and its aggregate terminal and
per-heading counts. It may not load a v10 development trajectory, route map,
language prompt, MaxEnt outcome, or Dr.GRPO outcome.

## Frozen fresh local-waypoint evaluation

Evaluate on four new 12×12 maps, dimension-disjoint from all previous 7×7
training maps and 8×8, 9×9, and 10×10 controller evaluations. Interior walls
are exactly:

1. `[5,5]` and `[1,10]`;
2. `[5,6]` and `[10,1]`;
3. `[6,5]` and `[1,2]`; and
4. `[6,6]` and `[10,9]`.

For every map and heading, select the legal edges at lexicographic indices
`0`, `floor((n-1)/2)`, and `n-1`. Use zero position noise and evaluation seed
base `6073011`, for 96 episodes and 12 per heading. No evaluation episode may
enter training.

The unchanged gate passes only if overall success is at least 0.90, every
heading succeeds on at least 9/12, every map succeeds on at least 18/24,
unhealthy termination is at most 0.10, median successful duration is at most
300 steps, and all identities/counts/metrics are valid. A pass authorizes only
the separately frozen, still-unexecuted 11×11 route slate. A failure stops
v11; no map, edge, seed, mixture, budget, or threshold may be substituted.
