# E111 Qwen-3B restart-durability amendment

Date frozen: 2026-08-18, before applying the runtime amendment and without
inspecting any E111 endpoint evaluation result.

## Scope and motivation

This scheduler-recovery amendment applies only to E111 Qwen-3B jobs
`30674758`, `30674759`, `30674760`, `30674761`, and `30674762`. The jobs run
on the preemptible `lowprio` partition. Scheduler records and training-step
telemetry show repeated restarts before the preregistered step-32 recovery
boundary. The amendment is motivated solely by restart counts and checkpoint
progress, not by reward, accuracy, coverage, or any other endpoint outcome.

## Frozen change

For a future allocation/restart of one of the five exact job IDs above, and
only when its source snapshot and variant exactly match E111, the Slurm wrapper
will override these storage-only settings:

- `OAT_ZERO_SAVE_STEPS`: 32 -> 8
- `OAT_ZERO_SAVE_FROM`: 32 -> 8
- `OAT_ZERO_RESUME_STEPS`: 32 -> 8

The existing atomic checkpoint implementation and two-checkpoint retention
limit remain unchanged. An attempt already running when this amendment is
installed continues with its original cadence; the override takes effect only
if Slurm starts a later attempt from the batch wrapper.

## Invariants

- Model, seed, data, prompt order, number of samples, optimizer, learning-rate
  schedule, MaxEnt coefficient and estimator, replay objective and weight,
  proposal policy, verifier, evaluation settings, and target step count remain
  byte-for-byte those in the submitted E111 environment.
- The proposal stream remains excluded from PPO and on-policy counts.
- No run directory is reset or duplicated. A later attempt auto-resumes the
  newest valid checkpoint written by the same job.
- The five original Slurm job IDs remain authoritative.
- PointMaze remains excluded.
- This is a durability amendment, not a treatment or outcome amendment.

## Audit rule

The live wrapper must contain a fail-closed block naming all five job IDs,
the exact E111 source and ops snapshots, and the exact E111 variant before it
overrides the three cadence variables. The amendment record must preserve the
wrapper block digest, pre-amendment scheduler records, restart counts, and the
E111 ledger digest. The final E111 auditor must validate this record.
