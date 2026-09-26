# E120-R2 amendment: smoke-auditor path repair

Date: 2026-09-03

This amendment is recorded after the E120-R1 operational smoke finished its
preregistered 32 training updates and before any of the 45 scientific
frequency-weighted runs received an allocation. No scientific endpoint outcome
has been produced or inspected.

## Failure and evidence boundary

The trainer completed successfully and wrote `TRAINING_COMPLETE.json`. The
post-run smoke auditor then exited nonzero because it searched for
`train_metrics.jsonl` directly under the stable run root. The launcher's frozen
training wrapper uses an attempt-specific directory (`debug_job<job_id>`) and
records that directory in the completion receipt as `terminal_attempt`.

This is an operational path-resolution failure. It does not alter training,
the replay objective, key counts, target weights, prompts, seeds, evaluation,
or any scientific estimand.

## Prospective repair

The auditor will first accept the historical direct metrics path. Otherwise it
will read `TRAINING_COMPLETE.json`, require `terminal_attempt` to resolve inside
the submitted stable run root, and audit
`<terminal_attempt>/train_metrics.jsonl`. All existing fail-closed telemetry
checks remain unchanged.

If and only if the repaired auditor passes the already completed smoke metrics,
the failed `afterok` dependency will be removed from the 45 still-unallocated
science jobs. Their commands and immutable source snapshot remain unchanged.
The smoke is operational only and is excluded from efficacy analysis.

The non-PVL resource restriction remains absolute. Before dependency repair,
