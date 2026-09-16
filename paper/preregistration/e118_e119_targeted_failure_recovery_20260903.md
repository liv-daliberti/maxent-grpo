# E118/E119 targeted failure recovery — 2026-09-03

This operational amendment was written before replacement submission and
without inspecting treatment endpoints.

Two registered cells require checkpoint continuations:

- E118 job 31010906: Qwen2.5-3B, Python, MaxRL, seed 72. The allocation reached
  the fixed 12-hour Slurm limit and retained coherent checkpoint `step_01920`.
- E119 job 31014398: Qwen2.5-0.5B, Countdown, Dr.GRPO, seed 43. The allocation
  exhausted its evaluation watchdog attempts after saving coherent checkpoint
  `step_01344` and entering sampled mode-coverage evaluation.

Each replacement preserves its model, data, seed, arm, run directory, frozen
source snapshot, optimizer, and evaluation settings. E118 retains its original
12-hour limit. E119 uses the already registered evaluation-safe watchdog
settings (7,200-second stale threshold, 3,600-second startup grace, and at most
12 restarts). No endpoint is used for recovery or stopping decisions.

Both replacements must be submitted held, must request automatic checkpoint
resume, and must carry the full PVL node exclusion. They may be released only
after their scheduler records and run-directory identities pass audit. E118's
Qwen-3B lineage is installed in its scale ledger and aggregate ledger; E119's
lineage is installed in the existing continuation ledger.
