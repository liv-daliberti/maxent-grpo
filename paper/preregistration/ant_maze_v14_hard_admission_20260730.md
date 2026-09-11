# AntMaze v14-hard prospective admission

**Status: FROZEN BEFORE ROUTE EXECUTION AND BEFORE MODEL SAMPLING — 2026-07-30**

The original AntMaze cohort was saturated at pass 0. This secondary benchmark
therefore replaces the one-cell, four-decision geometry with a 15x15 map, a
5x5 central obstacle, start/goal separation of eight cells, and two
fourteen-decision upper/lower routes. It retains the admitted v12 Ant
controller, official AntMaze-v5 simulator, public action alphabet,
action-repeat 400, exact execution-bound canonicalization, and four
train/development/evaluation maps per split.

Before any language model is sampled, all 24 registered routes must execute
successfully, produce two distinct topology keys per map, survive 100 stored
trajectory perturbations per route, and pass the existing identity,
near-miss, source-hash, and throughput checks. A failed route stops this map
slate; no map or route may be substituted after observing a model outcome.

After admission, a frozen-model development gate must show at least one
verified completion and must not exceed 90% mean@8 success. Only then may a
paired repair smoke be registered. The original AntMaze results remain
unchanged and are never pooled with v14-hard.

