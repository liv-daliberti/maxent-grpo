# PointMaze closed-loop viability v1 runtime repair r1

Frozen after infrastructure-only job 30185346 and before any repaired sample.

Job 30185346 was broadened from A5000-specific to generic `gpu:1` to use idle
capacity. Slurm placed it on an RTX 2080 Ti. vLLM rejected BF16 because the
device has compute capability 7.5. The job stopped during device
initialization, before model weights completed loading, before the worker reset
any development episode, and before any completion or endpoint reward.

The sole repair is to require one A100 GPU and write a fresh r1 identity and
receipt. The model snapshot, BF16 dtype, source snapshot, closed-loop
finite-action interface, four development maps, 64 rollouts, first-16 prefix,
seed 75101, information firewall, thresholds, and endpoint verifier remain
unchanged from
`point_maze_interactive_05b_viability_v1_20260729.md`.

Job 30185346 is retained as an infrastructure failure and has no scientific
status.
