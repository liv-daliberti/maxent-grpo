# AntMaze v15 / controller-v18 0.5B viability

**Status: FROZEN BEFORE THE V18 CONTROLLER OUTCOME, V15 EXECUTABLE
ADMISSION, OR ANY V15 LANGUAGE-MODEL SAMPLE — 2026-07-30**

This gate tests whether the existing train-only Qwen2.5-0.5B AntMaze
public-Markov-state checkpoint can interact with the prospectively frozen v18
stable-handoff controller on the harder v15 maps.

The gate runs only if controller v18 and v15 executable admission both pass.
It uses only the four v15 development rows. It does not load evaluation rows,
certified route programs, planner outputs, controller feedback, canonical
identities, or intermediate verifier feedback.

Use the exact `ant_maze_interactive_warmstart_v13` checkpoint, the fixed
eight-token compass support, 64 trajectories per prompt, prefix 16, seed
`76701`, temperature/top-p 1.0, at most 20 decisions, and 400 simulator steps
per decision. Each decision is accepted as a handoff only when the v18
controller reaches the cumulative waypoint within 0.45 at planar speed at
most 1.0.

Require exactly 256 terminal attempts, at least two prompts with prefix
success, at least two prompts with two verified canonical modes, and an
aggregate verified rate in the inclusive interval [0.02, 0.50]. The upper
bound is the prospective anti-saturation criterion responding to the prior
AntMaze cohort’s 255/256 development success. No threshold may be changed
after sampling. A failure stops before online training; a pass authorizes only
a separately frozen five-seed online comparison.
