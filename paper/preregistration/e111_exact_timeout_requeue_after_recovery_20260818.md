# E111 exact timeout requeue after recovery

Date frozen: 2026-08-18, after the 17:27 allocation attempts reached their
ordinary 45-minute wall-time and before requeueing any job or inspecting any
E111 task outcome.

## Trigger and exact scope

Jobs `30674729`, `30674733`, `30674754`, and `30674762` ended their allocation
attempts at 2026-08-18 18:12:40 EDT with scheduler state `TIMEOUT`.  This
cluster did not automatically place them back into the queue despite their
submitted `Requeue=1` setting.

- `30674729`, `30674733`, and `30674754` require the already recorded
  proposal-retention checkpoint-deserialization repair.
- `30674762` requires the already recorded recoverable quarantine of its
  partial `step_00004` optimizer archive, leaving validated `step_00002` as
  its highest selectable checkpoint.

Issue exactly one `scontrol requeue` for each of those four original job IDs.
Do not reset job state, submit replacements, change run directories, or alter
any job environment.

## Invariants

- This is scheduler-only recovery.  Model, optimizer, RNG restoration, seed,
  data and prompt order, MaxEnt estimator/coefficient, ReplayDr objective and
  weight, proposal policy, verifier, evaluation settings, and target step
  count remain unchanged.
- Existing checkpoints and prior metrics remain authoritative.
- No task reward, accuracy, coverage, or endpoint outcome is inspected.
- PointMaze remains excluded.

## Audit rule

Record pre-action scheduler/accounting evidence, the exact successful requeue
commands, post-action scheduler records, unchanged job IDs and environments,
and subsequent checkpoint-load/training progress.  Any replacement job ID,
state reset, or treatment/environment drift fails the E111 gate.
