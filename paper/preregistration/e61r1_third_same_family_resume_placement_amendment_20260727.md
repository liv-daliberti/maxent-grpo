# E61-R1 third same-family resume placement amendment

Recorded 2026-07-27 after the E61-R1 scientific protocol and trajectories
were frozen, after new scheduler preemptions, and before this placement
change.

Four A6000 MathIR jobs are now pending checkpoint resumes:

- Dr.GRPO seed 43: `30126351`;
- E58 seed 43: `30126352`;
- Dr.GRPO seed 44: `30126353`; and
- E58 seed 44: `30126354`.

All four are materialized and retain durable optimizer, model, prompt cursor,
controller, canonical-bank, request-stream, metric, and evaluation state.
Their latest metric steps before this amendment are recorded in the
machine-readable amendment artifact. The original placement is
`mltheory/lowprio` on `node103,node104,node208`. Node availability and
reservations have repeatedly preempted or delayed this narrow pool.

This infrastructure-only amendment moves only these four jobs to
`mltheory/pvl-lowprio` and changes their eligible node list to
`node103,node104,node805`. Every eligible node advertises one A6000 per job.
E66/E68 Graph jobs use the same registered A6000 family and remain independently
audited; this amendment changes no E66 or E68 job.

Run stamps, job IDs, frozen source and execution snapshots, model, data, arm,
seed, objective, sampling, evaluation cadence, checkpoint cadence, optimizer
state, and every scientific setting remain unchanged. All affected jobs must
be held and pending before mutation. The E61-R1 integrity audit must verify
this amendment's document and executable hashes, exact affected-job set,
same-family mutation, and reject any trace that regresses below its recorded
pre-amendment step.
