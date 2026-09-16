# E111 Qwen-3B Pantry partial-checkpoint recovery amendment

Date frozen: 2026-08-18, before changing checkpoint storage and without
inspecting any E111 task outcome.

## Trigger

Job `30674762` selected `step_00004`.  Its model-state archive has a valid ZIP
central directory, while its optimizer-state archive fails the read-only
`zipfile.ZipFile` directory check with `BadZipFile`.  The previous complete
checkpoint, `step_00002`, passes the same check for both archives.  The failed
allocation emitted `PytorchStreamReader failed reading zip archive` and then
remained stale until its ordinary scheduler timeout.

## Recovery

Move the complete `step_00004` directory, without deleting it, to a sibling
quarantine directory outside `checkpoints/`; point the small `latest` marker
at `step_00002`.  Do not signal, reset, duplicate, or otherwise alter job
`30674762`.  Its next natural allocation must resume from `step_00002`.

## Invariants

- This is checkpoint-storage recovery only.  Python source, treatment,
  optimizer configuration, seed, data order, evaluation, and target steps do
  not change.
- No reward, accuracy, coverage, or other task outcome is inspected.
- The original job ID, run directory, and all prior metrics remain
  authoritative.  The quarantined bytes remain recoverable.
- PointMaze remains excluded.

## Audit rule

The recovery record must identify the exact paths, pre-action sizes and ZIP
directory checks, old traceback count, quarantine time, unchanged scheduler
job ID, and post-requeue evidence that `step_00002` was selected and training
advanced without another partial-checkpoint traceback.
