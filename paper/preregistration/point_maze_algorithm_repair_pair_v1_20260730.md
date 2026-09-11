# PointMaze algorithm-repair paired qualification v1

**Status: TEMPLATE FROZEN BEFORE THE FROZEN-MODEL VIABILITY OUTCOME — 2026-07-30**

This pair may launch only if development-only viability job 30204570 returns
at least one verified completion, at least one multimode prompt, and a
non-saturated verified completion rate. The qualification adapter must bind
the viability receipt, admission receipt, data identity, and cardinality
clarification by hash.

The pair uses seed 76501, eight train-only 11x11 maps, 12 complete prompt
passes, exactly 96 updates, and 16 rollouts per prompt. Arms are
compute-matched Dr.GRPO and verified-first global replay MaxEnt with the
existing semantic 0.10, novelty 0.50, and replay 0.10 recipe. Optimizer,
sampling, verifier, public action support, fixed-shape policy/replay work, and
initial model are matched.

The four development maps are evaluated at pass 0 and every two updates with
four K=8 draws. Evaluation request seeds depend on arm seed, row, draw,
sample, and decision round, but not checkpoint update. Evaluation feedback is
never used for training or checkpoint selection.

Qualification requires each arm's aggregate train verified rate to lie in
[0.10, 0.90], nonzero task advantage on at least 24 of 96 updates, exact-zero
control exploration/replay derivatives, nonzero treatment
exploration/replay derivatives, matched fixed-shape traversal, all 49
evaluation coordinates, finite metrics, exact receipts, and no resume. Failure
stops the final five-seed repair cohort.

