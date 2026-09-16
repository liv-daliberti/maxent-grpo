# E117-S1 zero-runtime lowprio partition repair

Frozen: 2026-08-24 before scheduler mutation and without reading any E117 run
artifact or endpoint. PointMaze remains excluded.

## Trigger

E117 jobs 30873543--30873554 passed their held scientific-configuration audit
and were released. The submit-side router then resolved the requested
`--partition=all --account=mltheory --time=08:00:00` surface to partition
`mltheory`. All 12 jobs are `PENDING`, `Priority=0`,
`Reason=BadConstraints`, `RunTime=00:00:00`, `Restarts=0`, with no allocated
node and no run directory.

Current partition membership explains the deterministic failure: frozen E117
nodes 021 and 022 are members of `lowprio` (and the general partitions) but not
of `mltheory`. The job's resolved partition and required node therefore have an
empty intersection. Waiting cannot repair it.

## Authorized change

Transactionally hold all 12 jobs, verify their full exported scientific
environment is byte-identical to the held record in the E117 release ledger,
and change only:

- partition `mltheory` to `lowprio`.

Keep account `mltheory`, each exact node021/node022 binding, generic one-GPU
request, 8 CPUs, 64 GiB, eight-hour limit, Nice=0, job ID, dependency graph,
source/ops snapshot, model, data, seed, optimizer, sampler, proposal controls,
semantic coefficient, checkpoint/evaluation schedule, and output path
unchanged. `lowprio` accepts the `mltheory` account and contains both requested
nodes.

The application must hold every job before the first update, audit the complete
set before release, record before/held/after scheduler rows plus environment
hashes, and roll every changed job back to `mltheory` if any update or audit
fails. This amendment changes placement eligibility only and reads no efficacy
or mechanism outcome.

The existing after-any audit job 30873569 remains bound to the same 12 job IDs;
it is not replaced or relaxed.
