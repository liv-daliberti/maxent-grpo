# PointMaze repair-v2 pair r3 flattened trainer wrapper

**Status: FROZEN AFTER JOBS 30204893–30204894 FAILED ON A SECOND IMPORT AND BEFORE R3 SUBMISSION — 2026-07-30**

R2 added the first missing wrapper, then exited before model or data loading
because that wrapper imported another repository-local Stage-B module absent
from the snapshot.

R3 flattens the repair trainer: it imports the Stage-B trainer directly,
applies the same seed, family list, checkpoint-invariant evaluation wrapper,
and receipt metadata, and snapshots both that Stage-B module and its existing
interactive base. This removes the nested repair-wrapper import chain.

No model, data, seed, arm, optimizer, rollout, evaluation, threshold, or
algorithm setting changes.
