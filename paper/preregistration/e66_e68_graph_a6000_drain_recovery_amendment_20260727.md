# E66/E68 Graph A6000 drain-recovery placement amendment

Recorded 2026-07-27 after the E66 and E68 protocols, trajectories, source
snapshots, and job identities were frozen, and before this placement change.

## Non-outcome reason

All six prospective Graph jobs are pending on A6000 nodes `node103,node104`.
At the time of this amendment, Slurm reports both nodes draining after node
health checks detected overheated GPUs:

- node103: 9 overheated GPUs;
- node104: 7 overheated GPUs.

The E68 Graph jobs therefore have no scheduler start estimate. The three E66
Graph controls have partial checkpoint-resumable trajectories but are also
pending on the same drained pool.

A scheduler-only `srun --test-only` probe under account `allcs` and partition
`lowprio` found a schedulable same-accelerator-family placement on
`node205,node206,node207`, all of which advertise A6000 GPUs. The probe
returned an estimated start rather than allocating a GPU.

## Mutation

The following six jobs move together:

- E66 controls: `30128394`, `30128395`, `30128396`;
- E68 treatments: `30130469`, `30130470`, `30130471`.

Their placement changes from account `mltheory`, partition `pvl-lowprio`, and
nodes `node103,node104` to account `allcs`, partition `lowprio`, and nodes
`node205,node206,node207`.

Every job continues to request exactly one `gpu:a6000`. Run stamps, job IDs,
frozen source and execution snapshots, model, data, seeds, objective,
controller, proposal settings, sampling, evaluation cadence, checkpoint
cadence, accumulated metrics, and all other scientific settings are
unchanged. E66 resumes its durable checkpoints; E68 remains at zero optimizer
updates at amendment time.

Moving both causal arms to the same accelerator family preserves the
registered paired comparison. Actual node identity remains reported and
cannot select an exclusion.

## Fail-closed conditions

All six jobs must be held and pending before mutation. The script verifies
the frozen arm, novelty beta, separated-support flag for E68, and A6000
request before release. The machine record binds this document and executable
by SHA-256. E66 and E68 audits must reject a missing/mutated record or any
trace that regresses below its recorded pre-amendment step.
