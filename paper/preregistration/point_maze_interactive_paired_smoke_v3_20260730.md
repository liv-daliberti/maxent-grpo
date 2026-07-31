# PointMaze interactive paired smoke v3

Frozen on 2026-07-30 before any v3 model request. This is the operational
replacement authorized by `maze_microbatch16_sampling_geometry_repair_20260730.md`.

V3 keeps the v2 model, public prompts, rows [1, 3, 5, 7], two arms, four
ordered updates, 16 rollouts per row, 96-decision horizon, optimizer,
objectives, replay traversal, worker, and all pass/fail thresholds. It uses
fresh development seed 75303 and policy/replay microbatch 16. The identity
must bind the failed v2 audit and immutable v2 batch diagnostic.

The independent audit requires 64 terminal verified episodes and at least one
verifier-distinct multimode update per arm, exact fixed traversal and worker
replay, byte-identical initial checkpoints, zero control treatment derivative,
eligible treatment derivatives, finite telemetry, and behavior/live maximum
absolute log-probability error at most 1e-4. A pass authorizes only the already
frozen five-seed PointMaze Stage B replacement. These weights are discarded.
