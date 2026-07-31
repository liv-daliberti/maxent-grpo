# PointMaze-GeometryShift 0.5B 12-pass replacement Stage B v1

Status: prospective; frozen on 2026-07-30 while the paired smoke was running,
before its outcome and before any final replacement cell was sampled or
optimized.

Launch requires the independent v4 paired-smoke audit to pass with decision
`eligible_for_ten_point_maze_geometry_shift_replacement_jobs`. The final
cohort is compute-matched Dr.GRPO versus verified online MaxEnt for seeds 43,
44, 45, 46, and 47: ten fresh cells. This is the registered
configuration-level replacement for the failed ConstructiveCode row; it is
not an independent semantic domain and does not erase the reported
ConstructiveCode qualification failures.

Each cell starts from the unchanged Qwen2.5-0.5B PointMaze warm-start, uses all
eight shifted training prompts for 12 ordered passes (96 updates), 16
rollouts per prompt, a 96-decision horizon, learning rate 2e-7, policy/replay
microbatch 16, and evaluation batch 16. The control computes but zeros all
MaxEnt and verified replay derivatives; the treatment applies both. Fixed
policy/replay traversal is identical between arms.

The four held-out rotation-3 evaluation maps are evaluated at round 0 and
every two updates through 96 using one greedy trajectory and four deterministic
temperature-one draws of K=8 per map (132 trajectories per coordinate).
Evaluation outcomes never affect training, replay, stopping, or selection.

The independent terminal audit requires the exact ten scheduler-complete
cells; pinned source, execution, model, data, and qualification identities;
all 96 training and 49 evaluation coordinates; finite metrics; exact
compute-matched traversal; behavior/live log-probability error at most 1e-4;
the control/treatment derivative boundary; and re-execution of every stored
simulator transition and canonical route key. No resume, row substitution,
seed substitution, map substitution, or carry-forward is allowed.
