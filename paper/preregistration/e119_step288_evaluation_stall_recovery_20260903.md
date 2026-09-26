# E119 step-288 evaluation-stall recovery — 2026-09-03

This operational amendment was written before manually recovering the affected
allocations and without inspecting E119 treatment outcomes.

Three E119 continuations stopped making filesystem progress after optimizer
step 287, during the scheduled step-288 sampled mode-coverage evaluation:

- original job 31014408 / continuation 31037848: Countdown, MaxRL, seed 45;
- original job 31014410 / continuation 31037849: Countdown, Dr.GRPO, seed 46;
- original job 31014450 / continuation 31037850: MathIR, Dr.GRPO, seed 46.

Each allocation completed its step-288 greedy evaluation, entered sampled
mode-coverage evaluation, and retained checkpoint `step_00192`. The recovery is
a scheduler-only requeue of the same continuation job IDs. It preserves the
frozen source snapshot, run directory, seed, treatment, optimizer, evaluation
configuration, and original CS placement constraints. Auto-resume may therefore
redo optimizer steps 193 onward but does not change a scientific cell.

The recovery must remain on the non-PVL E119 route: account `allcs`, partition
`cs`, CS GPUs, and the existing `node[203-205,207]` allowlist. Successful
installation requires all three jobs to acquire a new Slurm attempt with an
incremented restart count and for their logs to confirm checkpoint-192 resume.
No endpoint values are used to select, alter, or stop a recovery.
