# E113-R4-R2-S1: completed-smoke dependency-expiry amendment

Date frozen: 2026-08-24 15:30 EDT, after both R4-R2 operational smokes passed
and after the first authorized science-release attempt was rejected before any
science job ID was assigned.

## Trigger

R4-R2 smoke jobs `30865563` and `30865564` are terminal `COMPLETED` with exit
`0:0`. Each has nonzero runtime, a matching `TRAINING_COMPLETE.json`, a
`global_step_1` actor checkpoint, a latest-checkpoint pointer equal to one, and
the frozen trainer markers in its log. The outcome-blind release auditor passes
both jobs without violations.

The first science submission attempted to retain the frozen redundant Slurm
dependency `afterok:30865563:30865564`. Slurm rejected the first held `sbatch`
before assigning a job ID because this cluster sets `MinJobAge=300` seconds and
both successful smoke records had already expired from the active controller.
They remain immutable and verifiable through Slurm accounting and their output
receipts. Zero science jobs were submitted by the rejected transaction.

## Scheduler-only amendment

The release transaction must continue to run the complete outcome-blind smoke
gate before any science submission. Only after both smoke records, receipts,
checkpoints, pointers, and trainer markers pass may it:

1. submit the exact frozen 50-cell R4-R2 science matrix in held state;
2. audit every held job's resources, environment, runtime snapshot, and cell
   identity;
3. atomically record the full replacement graph and this amendment; and
4. release all 50 jobs together.

Because the already successful smoke IDs are no longer resolvable by Slurm's
active controller, the held science jobs omit the redundant scheduler
`afterok` expression. Their ledger rows retain both smoke IDs as gate
provenance. If the pre-submission auditor does not pass both smokes, no science
job is created.

This changes only scheduler dependency representation after the gate has
already passed. Models, data, prompts, seeds, 50 cell identities, sampling,
reward, optimizer, DAPO loss, stopping rule, step budget, runtime snapshot,
resources, and scientific analysis remain unchanged. No scientific endpoint
was inspected before this amendment.
