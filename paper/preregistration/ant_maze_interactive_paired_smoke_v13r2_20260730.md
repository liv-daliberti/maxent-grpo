# AntMaze interactive paired smoke v13r2

Frozen on 2026-07-30 before any v13r2 model request. This is the operational
replacement authorized by `maze_microbatch16_sampling_geometry_repair_20260730.md`.

V13r2 keeps the v13r1 model, public prompts, train rows [0, 1, 2, 3], seed
76313, two arms, four ordered updates, 16 rollouts per row, 16-decision
horizon, controller, optimizer, objectives, replay traversal, worker, and all
pass/fail thresholds. Only policy/replay microbatch changes from 4 to 16. The
identity must bind the failed v13r1 audit and immutable batch diagnostic.

The independent audit requires 64 terminal verified episodes and at least one
verifier-distinct multimode update per arm, exact fixed traversal and worker
replay, byte-identical initial checkpoints, zero control treatment derivative,
eligible treatment derivatives, finite telemetry, and behavior/live maximum
absolute log-probability error at most 1e-4. A pass authorizes only the already
frozen five-seed AntMaze Stage B replacement. These weights are discarded.
