# E61-R1 second same-family resume placement amendment

Recorded 2026-07-27 after the E61-R1 scientific protocol and trajectories
were frozen, after new scheduler preemptions, and before this placement
change.

The first same-family amendment covered three jobs that were pending at its
freeze time. Eight additional RTX 3090 jobs have since been preempted and are
now pending checkpoint resumes:

- Countdown: `30126339`, `30126340`, `30126341`, `30126343`, `30126344`;
- Python factors: `30126345`, `30126346`, `30126347`.

Their latest durable metric steps before this amendment are recorded in the
machine-readable amendment artifact. All eight are already materialized and
past the halfway point of their 12-pass trajectories. Under their original
`allcs/lowprio` placement on `node021,node022,node023,node024,node026`, their
scheduler estimates extend from 2026-07-30 through 2026-08-02 or are unknown.
A scheduler-only probe made before this amendment found a substantially
earlier same-family backfill window under `mltheory/pvl-lowprio`.

This infrastructure-only amendment moves only those eight pending jobs to
`pvl-lowprio` under account `mltheory` and broadens their placement within
the original RTX 3090 accelerator family to:

`node020,node021,node022,node023,node024,node026`.

The prospective E68 jobs retain higher scheduler priority than these older
E61-R1 recoveries. Run stamps, job IDs, frozen source and execution snapshots,
model, data, arm, seeds, objective, sampling, evaluation cadence, checkpoint
cadence, optimizer state, and every scientific setting remain unchanged.

All affected jobs must be held and pending before mutation. Existing
checkpoint state and metrics are retained. The E61-R1 integrity audit must
verify this amendment's document and executable hashes, exact affected-job
set, same-family mutation, and reject any trace that regresses below its
recorded pre-amendment step.
