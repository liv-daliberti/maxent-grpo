# PointMaze orientation-balanced v3 viability gate

**Status:** FROZEN BEFORE V3 ROUTE OUTCOMES OR MODEL SAMPLING  
**Date:** 2026-07-30  
**Role:** Development-only trainability qualification for the disclosed v3
engineering repair; excluded from the original 80-cell estimator.

The gate may start only after executable admission of the v3 data. It samples
the exact frozen Qwen2.5-0.5B PointMaze velocity-state warm start on the four
development rows. Those rows contain exactly one task at each rotation
0/1/2/3. Evaluation rows are not loaded.

Sampling is fixed at 64 closed-loop rollouts per prompt, temperature 1,
top-p 1, seed 76530, and the official terminal checker. The online-training
alignment prefix is the first 16 rollouts per prompt.

All qualification checks must pass:

- exactly 256 terminal rollouts;
- overall verified rate in [0.02, 0.50];
- at least two of four prompts have a verified rollout in the K=16 prefix;
- at least two prompts expose both certified topology modes across 64 draws;
- exact balanced-orientation data and passing admission identities;
- development-only information boundary and no evaluation-row loading.

Failure stops v3 before paired optimization. Passing authorizes one new
development calibration pair only; it does not authorize the five-seed final.
