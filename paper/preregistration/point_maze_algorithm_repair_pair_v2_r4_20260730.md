# PointMaze repair-v2 pair r4 qualification-name adapter

**Status: FROZEN AFTER JOBS 30204896–30204897 STOPPED AT THE GENERIC GUARD AND BEFORE R4 SUBMISSION — 2026-07-30**

R3 loaded the flattened trainer successfully and reached the generic Stage-B
qualification guard. It stopped before model loading because that reusable
trainer accepts the legacy decision name
`eligible_for_ten_point_maze_stage_b_jobs`, while the passing v2
qualification uses the more specific
`eligible_for_point_maze_algorithm_repair_v2_pair`.

R4 maps only that exact passing v2 schema/decision pair to the legacy internal
decision string during guard evaluation. The receipt and identity retain the
v2 qualification hash and original external decision. No data, model, seed,
arm, optimizer, rollout, evaluator, threshold, or algorithm setting changes.
