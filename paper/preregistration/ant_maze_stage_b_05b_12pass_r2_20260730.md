# AntMaze 0.5B 12-pass Stage B r2

Frozen on 2026-07-30 before any r2-qualified final cell. This replaces only
the failed qualification path; no prior final AntMaze paper cell ran.

Launch requires a passing AntMaze v13r2 paired audit bound to development seed
76313, rows [0, 1, 2, 3], policy/replay microbatch 16, and the immutable v13r1
failure diagnostic. The final cohort is two arms (compute-matched Dr.GRPO and
verified-first global canonical replay MaxEnt) by seeds 43, 44, 45, 46, and
47: ten cells total.

Each cell keeps the frozen Qwen2.5-0.5B warm-start checkpoint, four training
prompts, 12 ordered passes (48 updates), 16 rollouts per prompt, the fixed
16-decision horizon and v11 low-level controller, learning rate 2e-7, exact
online objectives and replay schedule, microbatch 16, and evaluation batch 16.
Evaluation is at round 0 and every update through 48 with four draws per
coordinate. The independent terminal audit continues to require exact
scheduler/artifact identity, fixed traversal and simulator replay, finite
metrics, behavior/live log-probability error at most 1e-4, and the frozen
control/treatment derivative boundary. No resume, map substitution, seed
substitution, or carry-forward is allowed.
