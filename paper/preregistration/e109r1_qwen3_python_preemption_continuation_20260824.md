# E109-R1 Qwen-3B Python preemption continuation

Frozen: 2026-08-24 before continuation submission and without reading an E109
or E112 evaluation endpoint. PointMaze remains excluded.

E109 has 13/15 terminal repaired Python ReplayDr comparators. The missing cells
are Qwen-3B seeds 73 and 74, original jobs 30659554 and 30659555. Scheduler
accounting records both as `PREEMPTED` after 11 and 9 restarts. Neither job is
active. Training-only progress reaches steps 2,391 and 2,197; the latest
complete, independently validated model+optimizer checkpoints are step 2,304
and step 2,112. No terminal receipt exists.

Submit exactly two held continuations into the existing registered run
directories. Reuse each original ledger record's full `--export` block byte for
byte, including model, data, parser, seed, optimizer, replay, evaluation,
checkpoint, source snapshot, run stamp, and save path. Auto-resume must select
the latest complete checkpoint. No checkpoint or partial artifact is deleted.

Both cells were prospectively assigned to A6000 hardware. Keep one A6000,
16 CPUs, 128 GiB, the three-day limit, and `Nice=100`. To avoid another
low-priority requeue cycle, use non-preempting partition `all`, account `allcs`,
and the contemporaneously healthy A6000 pool `node[104,205-207,805]`. This is a
scheduler-only continuation; model family, cell identity, hardware class,
scientific environment, horizon, and estimand do not change.

The launcher must validate both checkpoints, confirm the originals are
inactive and preempted, audit both new jobs while user-held at zero runtime,
record full scheduler rows and environment hashes, then release both together.
On any pre-release failure it cancels the new jobs. The continuation artifact
is the provenance bridge from the original E109 ledger to the shared run
directories consumed by the complete E112-R1 analysis.

