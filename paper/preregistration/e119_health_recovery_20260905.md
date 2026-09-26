# E119 health recovery — 2026-09-05

The user requested verification that all E119 cells are healthy and progressing.
The live audit found Countdown Dr.GRPO seed 44 job 31048378 FAILED with exit 75
after seven restarts and a 2,826-second watchdog timeout. Its complete step-2112
model and optimizer checkpoint has matching saved counters, although the latest
pointer still names step 1920. The existing validated auto-resume selector chooses
the highest structurally complete checkpoint independently of that pointer.

Pantry Dr.GRPO seed 43 job 31037827 and Pantry MaxRL seed 43 job 31048178 remained
RUNNING after their learners raised CUDA out-of-memory exceptions on node204's
24-GB A5000 GPUs. Their last metric steps were 120 and 159. Both have structurally
complete model and optimizer checkpoints at step 96 with matching saved counters.

Resume these exact scientific cells. Preserve datasets, seed, objective,
optimizer settings, evaluation, horizon, frozen learner source, and run directory.
Replace the terminal Countdown scheduler allocation using its exact recorded
submission command, adding the established two-hour watchdog, one-hour startup
grace and twelve-restart cap. Submit held, audit, update the continuation ledger,
then release. Preserve its node105/mltheory placement and 64-GB host-memory request.

Requeue and hold only the two proven dead Pantry allocations, preserve their job
IDs and submitted environments, then place them on existing permitted A6000
nodes: 31037827 on node205 and 31048178 on node207. Preserve their 40-GB host-memory
requests, account, partition, exclusions, checkpoint cadence and walltime.
Release after auditing placement and resource preservation.

Extend only the E119 frozen runtime guard's fatal-error pattern to recognize the
observed `torch.OutOfMemoryError: CUDA out of memory`, so future dead learners
are detected without waiting for the two-hour stale timer. Preserve prior bytes,
hashes, causal logs, checkpoint validation, scheduler before/after records and
guard checks in `var/artifacts/e119_health_recovery_20260905/`. Other running cells
are not restarted by this amendment. Selection uses operational failures only.

Execution followed the user's explicit request to fix E119 after reviewing the
health audit. The three-cell recovery was applied at approximately 13:45 EDT.
Countdown replacement job 31074759 is recorded in the canonical continuation
ledger; the two Pantry recoveries retain their original job IDs. Startup and
follow-up evidence are recorded under the same recovery audit directory.

The follow-up found two MathIR cells naturally requeued by the inherited
45-minute inactivity allowance and five recorded Pantry OOM cells on 24-GB
A5000 GPUs. Seven additional still-pending Pantry jobs were restricted to their
previously allowed A6000 nodes205/207; newly running jobs were preserved. The
separate placement protocol and pending_pantry_a6000.json record each change.

Future E119 allocations now have a minimum two-hour inactivity allowance and
twelve-restart budget, preserving longer explicit allowances. Their watchdog
counts writes to current-attempt training metrics, evaluation sidecars, result
files and checkpoint archives. Timestamps predating allocation start and other
attempt directories cannot extend the timer. Existing fatal-error detection
runs before activity checks. The broader legacy metric scanner is bypassed only
for E119 with this artifact-progress mode enabled. Scientific evaluation cadence,
checkpoint cadence, batches, objectives and optimizer settings are unchanged;
current training processes are not restarted merely to pick up this amendment.

Bash syntax, guarded configuration cases, current-versus-old artifact behavior,
and the composed watchdog loop passed validation. Independent review passed.
Source backups, before/after hashes and validation results are in
watchdog-progress-amendment.json and watchdog-review.json under the recovery
directory. These source hashes supersede the earlier E119 OOM-only runtime
amendment while retaining its behavior.
