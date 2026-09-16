# E113-R4-R2-S3 user-hold release

Frozen: 2026-08-29T18:50:58-04:00 before releasing the affected jobs.

Status: scheduler-only continuity amendment. This does not change any E113-R4
scientific cell, official-verl source, model/data/seed assignment, DAPO recipe,
sampling distribution, reward, optimizer, checkpoint, evaluation, or stopping
rule.

## Trigger

The four Qwen-0.5B Graph jobs `30869111`--`30869114` are training-terminal.
The other 46 authoritative R4-R2 science jobs `30869115`--`30869160` remain in
Slurm as `PENDING`, `Reason=JobHeldUser`, with zero runtime and zero restarts.
Their two R4-R2 operational smoke gates passed before science submission, and
the complete 50-job ledger was released after that outcome-blind gate.

The S2 continuity amendment already shortened the 46 pending jobs from seven
days to 12 hours so they can fit the non-preempting `all` partition. A current
user hold, rather than a dependency or scientific failure, prevents Slurm from
considering any of them for the available A6000 capacity. The user explicitly
requested on 2026-08-29 that official-verl DAPO resume wherever possible.

## Authorized action

Verify exactly jobs `30869115`--`30869160` against the authoritative ledger and
require all 46 to remain pending, user-held, zero-runtime, zero-restart,
dependency-free, `Partition=all`, `Account=allcs`, `QOS=long`, requeue-enabled,
one A6000 GPU, 16 CPUs, 128 GiB, and a 12-hour limit. Require their scientific
exports to match the held R4-R2 ledger records and require every affected run
directory to be absent.

Release exactly those 46 jobs together with `scontrol release`. Leave completed
jobs `30869111`--`30869114`, all E117 jobs, and every unrelated campaign
untouched. Afterward verify that no affected job remains user-held and record
the complete before/after scheduler provenance. Scheduling order and how many
jobs start immediately remain Slurm capacity decisions.

No incomplete-cell endpoint is inspected or used to justify this action.
