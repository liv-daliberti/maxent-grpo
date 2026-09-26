# E95 completion placement amendment

Date: 2026-08-18.

## Trigger

At the August 17 snapshot, nine untouched Qwen2.5-0.5B E95 cells were still
pinned to `node105` in `cs` at `Nice=4000`, even though `node105` is currently
advertised through `all` rather than `cs`. Four Falcon3-1B PantryPlan cells
were also at `Nice=4000` and user-held. Falcon seeds 55 and 58 retained valid
step-192 rolling checkpoints after reaching optimizer steps 335 and 344;
seeds 56 and 59 had never started. The two partial cells encountered GPU
memory contamination on their original A6000 placements, not a scientific
gate failure.

## Frozen amendment

This amendment changes scheduler placement and priority only.

- The nine zero-step Qwen jobs keep their existing job IDs, runtime snapshot,
  run directories, model, seeds, data, plain-GRPO objective, optimizer,
  evaluation cadence, and 3,072-update horizon. Their eligible pool is widened
  to the `all` partition on `node105,node202,node203,node204`, all with one
  A5000, and `Nice` is restored from 4000 to 0.
- The four Falcon Pantry jobs likewise keep their job IDs and every scientific
  setting. Their eligible pool is widened to the `all` partition on
  `node103,node104,node205,node206,node207,node208,node805`, all with one
  A6000, and `Nice` is restored to 0. Seeds 55 and 58 resume their registered
  rolling checkpoint; seeds 56 and 59 start from initialization.
- No completed cell is touched. No optimizer prefix is deleted or pooled
  across seeds. A5000 and A6000 jobs remain within their original accelerator
  family.

The amendment is applied fail-closed: all jobs are held, their immutable
scientific exports and resource class are audited, the scheduler changes are
recorded, and only then are they released.
