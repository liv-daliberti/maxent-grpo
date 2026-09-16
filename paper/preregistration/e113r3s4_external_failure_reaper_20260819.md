# E113-R3-S4: external terminal-failure reaper

**Frozen:** 2026-08-19 16:54 EDT, after observing that Slurm retained the
submission-time batch script for newly allocated R3 job 30790929. No endpoint
efficacy result was inspected.

## Reason for the amendment

All 50 E113-R3 jobs were submitted before the S3 mutable-wrapper amendment.
Slurm spooled that submission-time batch script, so pending jobs do not inherit
the later wrapper edit when they allocate. Job 30790929 confirmed this by
reporting the old `2700/3600` watchdog thresholds after allocation.

## Cleanup-only external monitor

One CPU-only, non-requeueing Slurm job may poll the exact 50 job IDs frozen in
`var/artifacts/e113r3_dapo_full_relaunch_jobs.json`. It may invoke the S3 cleanup
utility only when all of the following hold for a science job:

- the job ID belongs to that exact released ledger and is still `RUNNING`;
- no `TRAINING_COMPLETE.json` receipt exists;
- the unique job log contains the registered DAPO ten-generation-batch
  exhaustion signature; and
- that log has been unchanged for at least 300 seconds.

Before cancelling the orphaned allocation, the cleanup utility records the
maximum accepted policy-update index from `train_metrics.jsonl`. It then writes
an immutable per-job evidence record. It must never requeue, resume, resample,
raise the ten-batch cap, or modify any scientific input or output.

This amendment repairs process lifetime and monitoring only. Every exhausted
cell remains a failed DAPO feasibility outcome at its originally observed
accepted-update count; it is not an endpoint efficacy estimate.
