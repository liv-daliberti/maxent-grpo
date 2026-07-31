# AntMaze Stage-B 0.5B, 12-pass cohort — scorer repair r1

**Status: FROZEN BEFORE ANY R1-QUALIFIED FINAL CELL — 2026-07-30**

This protocol preserves the complete scientific design, final seeds, map
splits, online objective, 48-update schedule, 49 evaluation coordinates, and
132-trajectory evaluation slate in
`ant_maze_stage_b_05b_12pass_20260730.md`.  Its only operational repair is the
microbatch-4 scorer frozen in
`maze_microbatch4_scoring_repair_20260730.md`.

The ten paper cells remain the Cartesian product of Dr.GRPO and verified
online MaxEnt with final seeds 43, 44, 45, 46, and 47.  The model, v12 train and
evaluation maps, v11 low-level controller, action repeat 400, horizons,
rollouts, fixed policy/replay slots, learning rate, and mechanism coefficients
are unchanged.  Evaluation never feeds training.

Launch requires a passing independent v13r1 paired audit whose identity binds
development seed 76313, `artifact_cohort=v13r1`, policy microbatch 4, and the
zero-runtime cancelled predecessor jobs 30202665 and 30202666.  Every final
update must record microbatch 4 and behavior/live discrepancy at most 0.0001.
The final audit reexecutes every controller transition and rejects any missing,
resumed, substituted, nonterminal, or compute-mismatched cell.
