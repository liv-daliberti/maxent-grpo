# PointMaze repair-v2 pair r5 JSON adapter signature

**Status: FROZEN AFTER JOBS 30204899–30204900 STOPPED DURING DATASET LOAD AND BEFORE R5 SUBMISSION — 2026-07-30**

R4 passed the qualification guard and reached `datasets.load_from_disk`.
Hugging Face then invoked the process-wide `json.loads` adapter with the
standard `cls=` keyword, but the narrow adapter accepted only one positional
argument. Both jobs stopped before model loading or any rollout.

R5 accepts and forwards arbitrary standard `json.loads` positional and keyword
arguments, and applies the decision-name mapping only when the parsed value is
the exact v2 qualification object. No scientific setting changes.
