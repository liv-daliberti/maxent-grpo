# E111 small-scale node026 health widening

Date frozen: 2026-08-18, before changing any scheduler field and without
inspecting an endpoint or training-reward field.

## Trigger

The three unfinished Qwen-0.5B/Falcon-1B E111 jobs are pinned to `node026` by
the prior backfill amendment.  Slurm now reports `node026` as `MIXED+DRAIN`
with the NHC reason `check_nv_smi_temp: 5 GPUs are overheated`.  The jobs are
pending and cannot make progress on that required node.

## Recovery

For the same jobs `30674729`, `30674733`, and `30674754`, change only
`ReqNodeList` from `node026` to `node[202-204,403]`.  These are currently
healthy low-priority nodes with generic GPU allocation (A5000 or L40).  Keep
the `lowprio` partition, `gpu:1`, 64 GB memory, CPU request, 45-minute limit,
job IDs, run directories, checkpoints, source/runtime snapshot, data, seed,
optimizer, treatment, evaluation, and stopping rule unchanged.  Do not
requeue, reset, cancel, or replace a job.

Fail closed unless all three jobs are pending and their scheduler records
match the frozen E111 environment before the update.  Record before/after
scheduler and node-health evidence.  If a partial update fails, restore any
changed job to `node026` and remove no data.

This scheduler-only mechanism-gate recovery does not change the E112 paired
hardware plan.  PointMaze remains excluded.
