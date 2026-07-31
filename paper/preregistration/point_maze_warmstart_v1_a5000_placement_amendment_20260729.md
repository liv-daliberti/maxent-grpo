# PointMaze warm-start v1 A5000 placement-only amendment

**Status: FROZEN BEFORE JOB 30185642 STARTED — 2026-07-29**

PointMaze warm-start job `30185642` is identity-bound, has executed zero
seconds, and remains pending under `mltheory` even though idle A5000 nodes are
available through the cluster-wide `all` partition. The scheduler currently
estimates its start on 2026-08-05.

This amendment changes only the eligible partition of the same held-and-
released Slurm job from `mltheory` to `all`. It preserves job ID, Qwen2.5-0.5B
base snapshot, exact source and execution snapshots, train-only examples,
SFT seed 75201, 69 optimizer steps, development sampling seed 75103, A5000
accelerator requirement, CPU/memory/time requests, output paths, and every
scientific gate. No development or evaluation output exists at freeze time.

The amendment script must fail closed unless the job is still pending with
zero elapsed time, its identity hashes and command match the frozen receipt,
and no model or terminal receipt exists. It records complete before/after
Slurm records and hashes of this protocol and amendment source. A scheduler
start immediately after the partition update is permitted; any other command,
resource, source, data, seed, or objective change is forbidden.
