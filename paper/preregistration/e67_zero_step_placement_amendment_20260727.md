# E67 zero-step compatible-capacity placement amendment

Recorded 2026-07-27 after E67 submission and before any affected job
materialized a run directory or optimizer metric.

## Reason

The initial frozen placements used the correct accelerator families, but
Slurm reservations appeared between the read-only capacity check and release.
Predicted start times serialized Graph through July 31 and most
Countdown/Python jobs through July 29. Independent `srun --test-only` probes
found earlier compatible A6000 and RTX 3090 capacity under the MLTheory
account.

This is a scheduler-only, zero-step amendment. It is not based on model
quality, mechanism telemetry, or evaluation outcomes.

## Exact scope

Only these still-pending jobs may change:

- Graph: `30128500`, `30128501`, `30128502`;
- Countdown: `30128503`, `30128504`, `30128505`; and
- Python: `30128506`, `30128507`, `30128508`.

The three running MathIR jobs `30128509--30128511` are untouched.

Before mutation, every affected job must be held and must have:

- zero runtime;
- no matching `debug_job<id>` run directory;
- no `train_metrics.jsonl`; and
- the identity-bound E67 SubmitLine.

## Allowed mutation

- Graph retains `gpu:a6000:1`; its account becomes `mltheory` and its allowed
  nodes become `node104,node205,node206,node207,node805`.
- Countdown and Python retain `gpu:rtx_3090:1`; their account becomes
  `mltheory` and their allowed nodes become
  `node020,node021,node022,node024,node026`.

No source, execution snapshot, variant, model, data, seed, optimizer,
coefficient, controller, proposal, replay, evaluation, checkpoint, recovery,
or information-firewall setting may change. The amendment script and this
document are SHA-256-bound in a machine-readable artifact before release.
