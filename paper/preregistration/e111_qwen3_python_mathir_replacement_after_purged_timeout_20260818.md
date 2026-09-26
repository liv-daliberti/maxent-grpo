# E111 Qwen-3B Python/MathIR replacement after purged timeout

Date frozen: 2026-08-18, before submitting any replacement job.  No endpoint
evaluation result is inspected; the recorded post-freeze Pantry training-reward
exposure is not used.

## Trigger

Original jobs `30674760` (Python) and `30674761` (MathIR) ended in `TIMEOUT`.
An exact same-ID `scontrol requeue` attempt was then rejected with
`Invalid job id specified`; Slurm had already purged both from its active job
table.  The failed command changed no scheduler or run state and is preserved
in `e111_qwen3_python_mathir_timeout_requeue.json`.

## Recovery

Submit exactly two continuation jobs, one for each frozen cell, using:

- the same model, seed 70, domain data, prompt order, verifier, treatment,
  target 64-step horizon, and original run directory;
- the same frozen E111 source and runtime-ops snapshot;
- the effective A6000 low-priority placement and 45-minute backfill limit;
- the effective two-step checkpoint save/resume cadence; and
- the new highest-valid-checkpoint selector.

Submit both jobs held.  Fail closed unless their held scheduler records contain
the exact treatment, source/ops snapshot, run directory, seed, 64-step target,
two-step checkpoint cadence, and A6000 placement.  Record the mapping from each
original job ID to its continuation ID, then release both atomically.  On any
partial failure, cancel only the newly submitted held jobs and remove no data.

## Invariants

This is continuation after scheduler-record expiry, not a new scientific cell.
Optimizer updates, RNG restoration, MaxEnt estimator/coefficient, ReplayDr
objective/weight, proposal policy, evaluation, and stopping rule are unchanged.
Existing checkpoints and metrics remain authoritative.  No original artifact
is overwritten or deleted.  PointMaze remains excluded.

## Audit rule

The terminal E111 auditor must use the continuation job state/stdout for these
two cells while retaining the original ledger identity and run directories.
It must validate the failed same-ID attempt, held records, one-to-one mapping,
unchanged treatment/environment except recorded storage/placement recovery,
and absence of new failure markers.
