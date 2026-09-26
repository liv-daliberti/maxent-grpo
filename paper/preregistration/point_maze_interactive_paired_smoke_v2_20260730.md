# PointMaze interactive paired smoke v2

**Status: FROZEN BEFORE ANY v2 MODEL SAMPLE — 2026-07-30**

This is the prospective scoring-repair gate defined by
`maze_microbatch4_scoring_repair_20260730.md`.  PointMaze v1 remains a failed
development attempt and is neither overwritten nor relabelled.

The two arms are plain Dr.GRPO and verified online MaxEnt.  Both start from
the same immutable PointMaze warm-start v3 checkpoint, use development seed
`75302`, and consume train rows `1,3,5,7` in that order: one previously unused
map from each family `bar7`, `block9`, `bar9`, and `asymmetric_block9`.
Evaluation rows are not loaded.

Each arm performs exactly four optimizer updates, 16 rollouts per prompt,
6,144 fixed policy slots, 64 replay-mode slots, 6,144 replay-decision slots,
and the same replay/canonicalization traversal.  Learning rate is `2e-7`,
policy microbatch size is exactly 4, and the action horizon remains 96.  The
control executes all passive MaxEnt computations with exact-zero applied
derivatives; the treatment applies the frozen verified exploration and replay
terms.

The independent audit retains every v1 scientific criterion and additionally
requires `policy_microbatch_size=4` on every update and maximum behavior/live
log-probability discrepancy at most `0.0001`.  It replays every stored public
state/action transition against the frozen simulator.  Only a pass with
decision `eligible_for_ten_point_maze_stage_b_jobs` may authorize the five-seed
PointMaze cohort.

There is no resume, v1 receipt reuse, final-seed use, map replacement,
threshold change, extra update, endpoint substitution, or evaluation-row
access.
