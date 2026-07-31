# Ant maze-local waypoint controller v10

**Status: FROZEN AFTER THE TERMINAL V9 RECEIPT AND BEFORE V10 TRAINING OR EVALUATION — 2026-07-29**

## Antecedent

V9 completed its full frozen budget and failed its unchanged local-waypoint
gate. Overall success was 87/96 (0.90625), every map was at least 21/24,
unhealthy termination was zero, and median successful duration was 184 steps.
The sole failed criterion was heading balance: grid delta `(-1,+1)` (the
northeast compass target) succeeded on 4/12 trials, while all other headings
were at least 11/12. The eight failures were 400-step timeouts, not unhealthy
terminations. No v9 route map was executed and no language model was sampled.

V9 remains failed. V10 is a new development controller, not a reinterpretation
or rerun of that receipt.

## Frozen training intervention

- initialize from exact failed-v9 model SHA-256
  `c49cdb1b8c1bf0d89027766d062492e84538c827d6aa88a2079067a6867b19b7`;
- reset the PPO optimizer;
- 2,000,000 `AntMaze_UMaze-v5` transitions, 8 workers, seed `73010`;
- learning rate `2e-6`, 400-step horizon, unchanged `[256,256]` policy,
  reward, four-unit relative-target observation, and 0.45 success radius;
- heading schedule
  `NE,N,NE,E,NE,SE,NE,S,NE,SW,NE,W,NE,NW`, giving northeast 50% of
  episodes and each other heading 1/14; and
- the same four preregistered 7x7 training maps and all their legal local
  free-cell edges. Diagonals remain legal only when both corner cells are
  free.

Training may load the aggregate terminal v9 receipt and exact v9 weights. It
may not load a v9 development trajectory, any 8x8 development outcome beyond
the registered aggregate/heading counts above, any route map, language prompt,
MaxEnt outcome, or Dr.GRPO outcome.

## Frozen fresh local-waypoint evaluation

Evaluation uses four new 10x10 maps, dimension-disjoint from all prior 7x7
training/route maps, 8x8 v9 development maps, and the unexecuted 9x9 v9 route
slate. Interior walls are exactly:

1. `[4,4]` and `[1,8]`;
2. `[4,5]` and `[8,1]`;
3. `[5,4]` and `[1,2]`; and
4. `[5,5]` and `[8,7]`.

For every map and heading, three legal edges are selected at fixed indices
`0`, `floor((n-1)/2)`, and `n-1` from the lexicographic legal-edge list. This
stratifies boundary and interior positions rather than reusing v9's first
three. Evaluation uses zero position noise, seed base `5073010`, 96 episodes
total, and 12 per heading. No evaluation episode enters training.

The gate passes only if:

- overall success is at least 0.90;
- every heading succeeds on at least 9/12 episodes (0.75);
- every map succeeds on at least 18/24 episodes (0.75);
- unhealthy termination is at most 0.10;
- median successful duration is at most 300 steps; and
- all metrics are finite and all identities/counts match.

A pass authorizes one separately preregistered, new-fingerprint v10 route
gate. A failure stops v10. Maps, edge indices, seeds, thresholds, training
mixture, and budget cannot be repaired after evaluation.
