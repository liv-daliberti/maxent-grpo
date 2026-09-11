# E61-R1 same-family resume placement amendment

Recorded 2026-07-27 after the E61-R1 scientific protocol and trajectories
were frozen, after scheduler preemptions of the affected jobs, and before this
placement change.

Twenty-one of the 24 E61-R1 jobs are terminal or running. The remaining jobs
`30126334`, `30126336`, and `30126342` are pending checkpoint-resume jobs.
The two Graph E58 jobs are at steps 1930 and 1953 of 2304 with no scheduler
start estimate. The Countdown E58 job is at step 2450 of 4608 with a July 31
estimate and substantially lower queue priority.

This infrastructure-only amendment moves those three jobs to
`pvl-lowprio` under account `mltheory` and broadens placement only within the
original accelerator family:

- Graph remains A6000 and may use `node103,node104`.
- Countdown remains RTX 3090 and may use
  `node020,node021,node022,node023,node024,node026`.

Run stamps, job IDs, frozen source and execution snapshots, model, data,
seeds, objective, sampling, evaluation cadence, checkpoint cadence, and every
scientific setting remain unchanged. The other 21 E61-R1 jobs are untouched.

All affected jobs must be held and pending before mutation. Existing
checkpoint state and metrics are retained. The E61-R1 integrity audit remains
fail-closed and must reject a trace that regresses below its recorded
pre-amendment step. The amendment document and executable script are
SHA-256-bound in a machine-readable artifact before release.
