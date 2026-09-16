# E111 Qwen-3B Python second continuation after purged timeout

Date frozen: 2026-08-18, before submitting the continuation and without
inspecting an endpoint or training-reward field.

## Trigger

The first exact continuation `30739797` of original E111 Python cell
`30674760` ended in `TIMEOUT` after reaching optimizer step 54.  Slurm has
already purged that job from the active table, so same-ID requeue is
impossible.  The highest complete model+optimizer ZIP checkpoint is
`debug_job30739797/checkpoints/step_00054`.

## Recovery

Submit exactly one held continuation for the same scientific cell.  Preserve
the Qwen2.5-3B model, Python domain, seed 70, run directory, prompt traversal,
RNG/optimizer state, 64-step target, verifier, v7 MaxEnt estimator and
coefficient, uniform ReplayDr objective and weight, proposal policy, frozen
E111 source/runtime snapshot, A6000 low-priority placement, 45-minute limit,
and two-step save/resume cadence.  Select the highest valid checkpoint using
the installed ZIP validator.

Fail closed unless the held scheduler record has the exact frozen treatment,
source/runtime roots, run directory, checkpoint cadence, and placement.  On
failure, cancel only the newly submitted held job and remove no data.  Record
the chain `30674760 -> 30739797 -> new continuation`, then release it.

The E111 terminal auditor and campaign status must use the newest continuation
state/stdout while retaining original cell identity and the common run
directory.  PointMaze remains excluded.
