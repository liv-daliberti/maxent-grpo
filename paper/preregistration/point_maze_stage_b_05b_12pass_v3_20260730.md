# PointMaze 0.5B 12-pass Stage B v3

Frozen on 2026-07-30 before any v3-qualified final cell. This replaces only
the failed qualification path; no prior final PointMaze paper cell ran.

Launch requires a passing PointMaze v3 paired audit bound to development seed
75303, rows [1, 3, 5, 7], policy/replay microbatch 16, and the immutable v2
failure diagnostic. The final cohort is two arms (compute-matched Dr.GRPO and
verified-first global canonical replay MaxEnt) by seeds 43, 44, 45, 46, and
47: ten cells total.

Each cell keeps the frozen Qwen2.5-0.5B warm-start checkpoint, eight training
prompts, 12 ordered passes (96 updates), 16 rollouts per prompt, the fixed
96-decision budget, learning rate 2e-7, exact online objectives and replay
schedule, microbatch 16, and evaluation batch 16. Evaluation is at round 0
and every two updates through 96 with four draws per coordinate. The
independent terminal audit continues to require exact scheduler/artifact
identity, fixed traversal and simulator replay, finite metrics, behavior/live
log-probability error at most 1e-4, and the frozen control/treatment derivative
boundary. No resume, row substitution, seed substitution, or carry-forward is
allowed.
