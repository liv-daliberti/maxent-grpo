# E68 zero-step lowprio placement amendment

Recorded 2026-07-27 after E68 submission and before any affected job
materialized a run directory or optimizer metric.

## Reason

The initial same-family `pvl-lowprio` placement passed prelaunch capacity
probes, but the released jobs received predicted starts between July 28 01:00
and July 29 01:00. Fresh scheduler-only probes found earlier same-family
capacity under the existing `mltheory` account on `lowprio`: RTX 3090 at
approximately 18:47 and A6000 at approximately 19:10 on July 27.

This is a scheduler-only, zero-step amendment. It is not based on model
quality, mechanism telemetry, or evaluation outcomes.

## Exact scope

Only these still-pending jobs may change:

- Graph: `30130469`, `30130470`, `30130471`;
- Countdown: `30130472`, `30130473`, `30130474`; and
- Python: `30130475`, `30130476`, `30130477`.

The three running MathIR jobs `30130478--30130480` are untouched.

Before mutation, every affected job must be held, pending, and have no
matching run directory or `train_metrics.jsonl`.

## Allowed mutation

- Graph retains `gpu:a6000:1` and account `mltheory`; its partition becomes
  `lowprio` and its allowed nodes become
  `node103,node104,node205,node206,node207,node805`.
- Countdown and Python retain `gpu:rtx_3090:1` and account `mltheory`; their
  partition becomes `lowprio` and their allowed nodes remain
  `node020,node021,node022,node024,node026`.

No source, execution snapshot, protocol identity, variant, model, data, seed,
optimizer, coefficient, controller, proposal, replay, evaluation, checkpoint,
recovery, or information-firewall setting may change. The amendment script
and this document are SHA-256-bound in a machine-readable artifact before
release.
