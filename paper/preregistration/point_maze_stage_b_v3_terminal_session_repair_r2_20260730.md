# PointMaze Stage-B v3 terminal-session repair r2

Frozen on 2026-07-30 after the complete initial cohort outcome and before any
r2 replacement was submitted.  All ten initial jobs 30203021--30203030 failed
with the same `interactive PointMaze worker request timed out` transport
error.  Seeds 46--47 stopped after 60/96 updates, seed 43 after 62/96, and
seeds 44--45 after 64/96.  Every job has an exact matching partial state replay
and no terminal receipt.  Those artifacts are quarantined and ineligible.

Source inspection identified a lifecycle leak: completed sessions closed
their simulator but remained in the persistent worker's `sessions` mapping.
Repeated 132-trajectory evaluations accumulated thousands of closed
environment objects until a later reset exceeded the frozen 60-second worker
alarm.  The r2 source constructs the exact same terminal public response,
closes the environment exactly as before, then removes that completed session
from the private mapping.  No observation, action, reward, route, canonical
key, action horizon, model input, optimizer update, evaluation schedule, or
verifier changes.  An integration regression test requires a terminal session
ID to be reusable and all targeted maze tests pass.

The placement-only r1 jobs 30203550--30203553 were canceled after 37--38
seconds when the remaining six original failures became visible.  They reached
model initialization only, emitted no update/evaluation/replay artifact, and
have no scientific status.  R2 restarts both arms and all five seeds from the
original immutable warm-start model; there is no resume, carry-forward, seed
or row substitution, added pass, or result-selected retry.  Data, model,
protocol, trainer, auditor, Slurm resources, sampling, objective, and compute
traversal remain unchanged.  The patched source is independently snapshotted,
and the terminal audit uses a ten-clause AND barrier plus the explicit
repository-root runtime binding.
