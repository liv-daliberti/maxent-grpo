# PointMaze algorithm repair v1

**Status: FROZEN BEFORE ROUTE EXECUTION AND BEFORE MODEL SAMPLING — 2026-07-30**

The original PointMaze RL train slate was saturated: 15.98 of 16 rollouts
verified on average and fewer than 3% of updates carried any task advantage.
This secondary repair uses four unseen 11x11 geometry families with longer
certified routes. The PointMaze warm start has never trained on these rows.

Admission requires 16 unique train maps, eight held-out maps, two real
checker-accepted topology modes per map, 100 trajectory perturbations per
route, exact runtime/source binding, failure-mutation rejection, and the
existing throughput floor.

The first development pair uses seed 76501, three passes, 16 rollouts per
prompt, binary official task reward, within-prompt Dr.GRPO centering, and
compute-matched verified MaxEnt. Evaluation sampling uses common random
numbers: row, draw, sample, and decision round determine request seeds, while
checkpoint update does not.

Qualification requires:

1. both arms have verified rollout rate between 10% and 90%;
2. both arms have nonzero task advantage on at least 25% of updates;
3. treatment applies nonzero verified exploration and replay gradients;
4. control exploration/replay derivatives remain exact zero; and
5. all route, fixed-shape, source, compute, and evaluation identities match.

A failure stops the pair. Original PointMaze results remain immutable and are
reported separately.

