# PointMaze repair-v2 pair r1

**Status: FROZEN DURING CONFIGURATION AND BEFORE SUBMISSION — 2026-07-30**

The first v2 launcher configuration stopped before tests or `sbatch` because
it attempted to override a nonexistent generic `snapshot_files` helper. R1
uses the actual `snapshot_execution` hook and adds the imported v1 base
auditor to the immutable execution snapshot required by the v2 audit wrapper.

No model, data, seed, arm, optimizer, rollout, evaluation, threshold, or
scientific setting changes from the frozen v2 pair protocol.
