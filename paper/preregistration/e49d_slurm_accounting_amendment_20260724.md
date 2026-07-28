# E49D Slurm-accounting amendment — 2026-07-24

**Status: FROZEN BEFORE ANY E49D TRAINING LAUNCH**

Slurm purged failed menu job 30073606 from `squeue`/`scontrol` shortly after
completion. The durable babysitter conservatively treated the resulting
nonzero `squeue` query as unknown and therefore would not archive the job
record or submit its pending-only successor. `sacct` independently recorded
the exact terminal state `FAILED|1:0|2026-07-24T10:54:56`.

When `squeue` no longer recognizes a recorded job, the babysitter now queries
`sacct` and archives the record only for an explicit terminal Slurm state.
Unknown, pending, or running states still fail closed. No job is cancelled or
duplicated by this change.

This is orchestration recovery only. It does not alter data, proposal or audit
behavior, policy inputs, runtime validation, reward, E46 controller, cohorts,
schedule, or scientific gates.
