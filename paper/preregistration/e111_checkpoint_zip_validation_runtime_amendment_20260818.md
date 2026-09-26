# E111 checkpoint ZIP-validation runtime amendment

Date frozen: 2026-08-18, before changing the live E111 runtime-ops snapshot
and without inspecting any E111 task outcome.

## Motivation

The Qwen-3B Pantry recovery proved that the existing auto-resume selector can
choose the numerically highest DeepSpeed checkpoint even when preemption left
one of its `.pt` ZIP archives without a central directory.  The resulting
`PytorchStreamReader` failure is deterministic and cannot make progress until
the incomplete directory is ignored or quarantined.

## Change

Before considering any `step_*` directory, the auto-resume selector invokes a
small standard-library validator.  A candidate is eligible only if it has at
least one model-state archive, at least one optimizer-state archive, and every
`.pt` file exposes a readable ZIP central directory.  Invalid candidates are
reported and skipped; selection among valid candidates remains the existing
highest-step/newest-tie rule.  If none is valid, training starts from model
initialization as before.

Install the same helper and selection check in the current checkout and the
frozen E111 runtime-ops snapshot.  The live snapshot change applies only on a
future natural/requeued allocation; no running job is signaled or reset.

## Invariants

- Python training source and all optimizer-update mathematics are unchanged.
- Model, seed, data order, MaxEnt estimator/coefficient, ReplayDr objective and
  weight, proposal policy, verifier, evaluation, and target steps are
  unchanged.
- Valid checkpoints retain exact optimizer/RNG restoration.  Only candidates
  that DeepSpeed cannot deserialize are excluded.
- Original job IDs, run directories, and prior metrics remain authoritative.
- No task outcome is inspected.  PointMaze remains excluded.

## Audit rule

Unit tests must show acceptance of a complete model+optimizer checkpoint,
rejection of a truncated archive, and fallback to the highest valid candidate.
The amendment record must hash the protocol, helper, root selector, and live
E111 selector and declare no treatment or optimizer-update change.
