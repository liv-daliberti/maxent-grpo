# AntMaze v15/v17 constrained 0.5B development viability

Frozen on 2026-07-30 before the v17 controller outcome, before v15/v17 route
admission, and before any 0.5B completion on a v15 map.

## Purpose and model stratum

The endpoint-only v12 base-model format failed at 0/256, while the explicitly
constrained closed-loop v13 language policy passed at 255/256. This gate
therefore uses the language interface that is demonstrably viable rather
than repeating a known-dead interface.

The model is the exact unchanged
`ant_maze_interactive_warmstart_v13` Qwen2.5-0.5B checkpoint with tree hash
`0f18039887e77b2e37eb15161dd8bb44ed6103f188755db9029bf0a61fe22090`.
It was trained only on 32 v12 training examples; its receipt records that no
v12 development/evaluation data or online reward was loaded. No v15 example,
route, development row, evaluation row, or outcome is used to update it.
This is reported as a transferred task-specific warm-start stratum, not as
base-checkpoint performance.

## Frozen development gate

Only the four-row v15 `dev/multi_answer` split is loaded. The maze geometry,
split, reset seeds, action repeat 400, horizon 20, controller receipt, and
topology verifier are exactly those admitted by the preceding deterministic
v15/v17 gate.

At each decision, the 0.5B model sees the public maze, public Ant position and
velocity, public goal, and remaining horizon. Its next-token logits are
restricted to the fixed public one-token alphabet
`N NE E SE S SW W NW`; the model selects the action. It receives no planner
output, canonical key, reward, answer, future state, controller feedback, or
endpoint-verifier feedback before termination.

For each map, sample 64 closed-loop trajectories at temperature 1 and top-p
1 with base seed 108317. The first 16 form the frozen training-group viability
prefix. The pinned v13 checkpoint is evaluated unchanged with max model
length 1536 and batch size 256.

## Decision

The gate passes only if at least two of four maps have a verified trajectory
in the first 16 and at least one map exposes both canonical route keys among
all 64. All 256 trajectories must terminate with complete source, simulator,
controller, and topology records.

A pass authorizes one separately frozen paired online mechanism smoke. A
failure stops this AntMaze cohort before online language-model training. No
prompt, threshold, route, map, seed, sample count, checkpoint, or controller
may change after the outcome.
