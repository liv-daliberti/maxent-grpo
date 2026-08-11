# E75F1 held-audit dependency validation amendment

**Frozen:** 2026-08-04, after the first held submission transaction failed
validation and before any E75F1 job was released or executed.

The first E75F1 launcher transaction submitted held jobs 30258089--30258104.
The audit job requested the correct conjunction of all ten online dependencies.
Slurm represented that request in `scontrol show job` as ten comma-separated
`afterok:<job>(unfulfilled)` clauses rather than the compact
`Dependency=afterok:<job>:...` string expected by the launcher validator.

The launcher failed closed and canceled the entire transaction. Scheduler
records show runtime 00:00:00 and canceled state; no E75F1 data directory,
identity, submission, model, receipt, metric, or replay artifact was created.
No model or benchmark outcome was observed.

This amendment changes only held-job read-back validation. A single dependency
continues to require the exact compact string. Multiple dependencies require
every individual `afterok:<job>` clause to appear in the scheduler record.
The requested dependencies, model, data seed and exclusions, development gate,
arms, seeds, training horizon, evaluation schedule, audit, and all scientific
rules in the original E75F1 preregistration are unchanged. The replacement
transaction receives fresh scheduler job IDs and a new execution snapshot.
