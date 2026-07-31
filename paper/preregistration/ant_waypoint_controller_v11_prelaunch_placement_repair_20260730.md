# Ant controller v11 prelaunch placement-only repair

**Status: RECORDED BEFORE ANY V11 CONTROLLER STEP OR EVALUATION — 2026-07-30**

The first held v11 submission, job `30198258`, was reported in partition `cs`
despite an explicit `--partition=all` submission flag. The fail-closed launcher
canceled it at runtime `00:00:00`, with no node allocation, controller step,
model artifact, or evaluation receipt.

The retry preserves the exact protocol, v10 initialization hash, training
schedule, seed, two-million-step budget, fresh 12×12 evaluation maps, resource
request, and thresholds. While the new job remains user-held, the launcher
verifies all scientific resources, applies `scontrol update Partition=all`,
verifies the resulting partition, disables requeue, and only then releases it.
The canceled identity is retained under its job-specific archival name.
