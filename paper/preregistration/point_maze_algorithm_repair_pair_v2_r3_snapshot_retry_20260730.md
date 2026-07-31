# PointMaze repair-v2 r3 snapshot retry

**Status: FROZEN AFTER R3 STOPPED BEFORE `sbatch` AND BEFORE RETRY — 2026-07-30**

The r3 dry import successfully loaded the flattened trainer and auditor from
their exact execution bundle, but Python wrote `__pycache__` into that
content-addressed directory. The subsequent run phase detected the hash drift
and stopped before `sbatch`; no r3 job was created.

The retry uses a fresh snapshot namespace. Dry-import validation sets
`PYTHONDONTWRITEBYTECODE=1`. No scientific or scheduler setting changes.
