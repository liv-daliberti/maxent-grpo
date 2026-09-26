# E91 adaptive-semantic checkpoint resume repair

Recorded 2026-08-14 after the failure was observed and before any repaired
resume was submitted. This is an operational repair, not a new preregistration.

## Observed failure

All 15 unfinished E91 Graph, Countdown, and MathIR cells completed roughly
6.7--7.0 of 8 passes and wrote valid checkpoints. On every later allocation,
checkpoint restore raised:

'semantic Shannon resume mismatch for coefficient: saved=<adapted> configured=0.1'

The adaptive RMS controller had correctly changed the coefficient during
training. The tracker checkpoint therefore correctly stored the active
coefficient, while a fresh process correctly initialized at the registered
base coefficient 0.1. The restore implementation incorrectly validated the
tracker before restoring the controller's active coefficient.

## Repair

The repaired runtime changes only 'src/oat_drgrpo/learner/run.py': it restores
the RMS controller, assigns its restored 'current_coefficient' to the semantic
tracker, and then invokes the tracker's existing strict 'load_state_dict'.
Consequently:

- a valid adaptive checkpoint resumes with its evolved coefficient;
- disagreement between the controller and tracker checkpoint states still
  fails closed;
- fixed-coefficient resume behavior remains unchanged;
- no objective, data, seed, placement, decoding, optimizer, or schedule setting
  changes.

The original frozen snapshot is retained byte-for-byte. A derived snapshot is
accepted only if every file is identical except the declared learner file.

## Replacement jobs

Each unfinished cell is resubmitted on its original node and GPU model with its
original 'SAVE_PATH'. 'OAT_ZERO_FIXED_EXP_SUFFIX=job<old_job_id>' points the new
allocation to the original checkpoint and metrics directory. The original job
is canceled only after its held replacement passes a Slurm-record audit.

Replacement jobs request eight hours because only 384--576 optimizer steps
remain, compared with 2,496--2,688 already checkpointed. This changes scheduler
fit only. Terminal evaluation remains pass 8 and no earlier checkpoint is used
as an outcome.
