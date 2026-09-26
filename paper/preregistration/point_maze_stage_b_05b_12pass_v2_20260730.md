# PointMaze Stage-B 0.5B, 12-pass cohort — v2 qualification

**Status: FROZEN BEFORE ANY v2-QUALIFIED FINAL CELL — 2026-07-30**

This protocol preserves the complete scientific design, final seeds, train and
evaluation splits, update schedule, evaluation schedule, mechanisms, and
fail-closed audit contract in
`point_maze_stage_b_05b_12pass_20260730.md`.  The only operational repair is
the prospectively frozen microbatch-4 scorer defined in
`maze_microbatch4_scoring_repair_20260730.md`.

The ten cells are the Cartesian product of arms Dr.GRPO and verified online
MaxEnt with final seeds 43, 44, 45, 46, and 47.  Each cell performs 96 optimizer
updates (12 passes over eight frozen train maps), evaluates all 49 quarter-pass
coordinates from 0 through 12 passes, and executes all 132 frozen evaluation
trajectories at every coordinate.  Evaluation never feeds training.

The cohort may launch only if the independent
`point_maze_interactive_paired_smoke_v2_audit.json` has schema v2,
`status=pass`, no errors, and decision
`eligible_for_ten_point_maze_stage_b_jobs`.  The v2 smoke must bind seed
`75302`, unused row indices `1,3,5,7`, and policy microbatch 4.  PointMaze v1's
failed audit is preserved and cannot qualify these cells.

Every final update must record `policy_microbatch_size=4` and behavior/live
log-probability discrepancy at most `0.0001`.  The independent final audit
reexecutes every stored simulator transition, verifies both arms' exact
compute traversal per seed, and rejects missing, resumed, substituted, or
nonterminal cells.
