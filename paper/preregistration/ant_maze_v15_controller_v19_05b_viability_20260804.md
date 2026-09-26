# AntMaze v15 / controller-v19 0.5B viability

**Status: FROZEN BEFORE THE V19 CONTROLLER OUTCOME, V15/V19 EXECUTABLE
ADMISSION, OR ANY V19 LANGUAGE-MODEL SAMPLE — 2026-08-04**

This development gate tests whether the existing train-only
Qwen2.5-0.5B AntMaze public-Markov-state checkpoint can interact with the
prospectively frozen v19 controller on the unchanged harder v15 maps.

The gate runs only if controller v19 and v15/v19 executable admission both
pass. It uses only the four v15 development rows. It does not load evaluation
rows, certified route programs, planner outputs, controller feedback,
canonical identities, intermediate verifier feedback, earlier v15 LM samples,
or any MaxEnt/Dr.GRPO outcome.

Use the exact `ant_maze_interactive_warmstart_v13` checkpoint, the fixed
eight-token compass support, 64 trajectories per prompt, prefix 16, fresh seed
`76702`, temperature/top-p 1.0, at most 20 decisions, and 400 simulator steps
per decision. Each decision is accepted as a handoff only when the v19
controller reaches the cumulative waypoint within 0.45 at planar speed at
most 1.0.

Require exactly 256 terminal attempts, at least two prompts with prefix
success, at least two prompts with two verified canonical modes, and an
aggregate verified rate in the inclusive interval [0.02, 0.50]. The upper
bound retains the prospective anti-saturation criterion motivated by the
original AntMaze cohort's 255/256 development success. No threshold, prompt,
seed, checkpoint, or sampling parameter may change after sampling.

Failure stops before harder-route online training. Passing authorizes only a
separately frozen five-seed online comparison. This seed and every resulting
trajectory are development-only and may not be reused in that comparison.
