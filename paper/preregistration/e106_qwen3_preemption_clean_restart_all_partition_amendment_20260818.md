# E106 amendment: Qwen-3B preemption clean restart on `all`

Frozen on 2026-08-18 at 01:22 EDT, after scheduler preemption of E106
Qwen2.5-3B Python job `30640331` and before its restart. The decision uses
only scheduler state, optimizer-step progress, file names/sizes, hardware
inventory, and previously frozen mechanism/resume proofs. No post-update
evaluation outcome was inspected. PointMaze is excluded.

## Trigger and checkpoint diagnosis

Job `30640331` ran on one A6000 on node208, reached logged optimizer step 22,
and was requeued once with exit code `0:0`. It then returned to `PENDING` on
`lowprio`; Slurm moved its next estimate to 04:07 EDT. The logs contain no
traceback, OOM, CUDA error, or NCCL error.

The frozen job saves model/optimizer state first at step 32
(`OAT_ZERO_SAVE_FROM=32`, `OAT_ZERO_SAVE_STEPS=32`). At the amendment time,
the run directory contains only the step-0 and step-16 evaluation records,
the mode-coverage draw log, and `train_metrics.jsonl`; it contains no model,
optimizer, replay-bank, semantic-history, or joint-resume checkpoint. Thus
calling the next allocation a step-22 resume would be false. The correct
scientific unit is a fresh 64-step run from the same frozen initialization.

## Authorized execution-only change

While the job is pending, the application script must hold it and verify the
exact frozen environment, restart count, step-22 prefix, and absence of a
resumable checkpoint. It may then:

1. rename the interrupted run directory to an immutable, explicitly labeled
   archive ending in `_preempted_no_checkpoint_step22_restart1`;
2. retain the original `SAVE_PATH`, causing the released job to create a clean
   run directory and restart from the frozen initialization; and
3. change only partition `lowprio` to partition `all`, retaining account
   `mltheory`, node pool `node[103-104,205-208,805]`, one A6000, 16 CPUs,
   128 GiB, and the two-hour limit.

The model, source/ops snapshot, seed, data, prompts, parser, optimizer,
learning-rate schedule, group size, replay objective, v6 semantic estimator,
evaluation cadence, stopping rule, and output path do not change. The archived
22-step prefix is diagnostic only and must never be combined with the clean
64-step trace in the gate or any outcome analysis.

The operation is transactional while the job is held: if validation, archive
rename, scheduler update, or artifact creation fails, restore the old
partition and original run directory before releasing the job. A durable
artifact must record the preemption, file manifest, frozen evidence hashes,
hardware inventory, and before/held/amended/released scheduler records.

## Gate consequence

Only the clean replacement trace may satisfy the Qwen2.5-3B Python cell. It
must independently reach step 64 and pass every existing verified-admission,
replay-actuation, v6-estimator, centering, legacy-off, and controller-off
criterion. This amendment cannot relax the combined 15-cell gate and does not
authorize E105 or E109 before that gate passes.
