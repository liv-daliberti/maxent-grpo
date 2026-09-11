# E51 Countdown/Python RTX 3090 placement amendment

**Status: FROZEN BEFORE SCHEDULER AMENDMENT — 2026-07-24**

## Scope and reason

This is an operational placement-only amendment to E51. At inspection time,
the six Countdown jobs `30074931--30074936` and six Python-factor jobs
`30074937--30074942` were all pending, with zero runtime, zero restarts, no
allocated node, and no training trajectory. The original node302 A100 queue
could not admit another 64-GiB job, while the original Python jobs were
priority-blocked in the `cs` partition despite substantial idle RTX 3090
capacity.

The user directed that the pending experiments start on available GPUs.

## Frozen amended placement

All twelve never-started jobs move together to:

- account `allcs`;
- partition `lowprio`;
- one `gpu:rtx_3090` per job;
- eligible nodes `node020,node022,node023,node024,node026`;
- eight CPUs, 64 GiB host memory, and the original seven-day limit.

The jobs remain preemptible and retain E51's exact rolling optimizer
checkpoint, automatic resume, and watchdog-requeue policy.

## Scientific invariants

The amendment changes only scheduler placement. It does not change:

- E51 protocol or identity;
- frozen source or execution snapshots;
- model revision, dataset, prompt pool, verifier, or canonicalizer;
- arm, seed, run stamp, group size, optimizer, budget, or evaluation cadence;
- policy-entropy sensor, 64-observation warmup, EMA decay, alpha formula, or
  projection-free controller contract;
- canonical-bank estimator, novelty coefficient, or checkpoint semantics.

Countdown and Python each remain hardware-homogeneous across all six jobs.
Graph-coloring jobs are not amended and remain on their original node302 A100
placement.

## Admission procedure

The twelve jobs must be held before mutation. Each resolved held record must
show `PENDING`, `JobHeldUser`, zero runtime, zero restarts, account `allcs`,
partition `lowprio`, the frozen RTX 3090 node list, one RTX 3090, eight CPUs,
64 GiB, the original E51 source and execution snapshots, and its original run
stamp. Release is all-or-nothing after this audit. Any job that allocated
before the hold or cannot be amended in place is excluded and requires a new
explicit recovery decision.

## Execution record

The twelve-job held audit passed before release. All jobs retained zero runtime
and zero restarts through the mutation. After release, Slurm allocated:

- Countdown `30074931--30074936`: node020;
- Python `30074937`: node026;
- Python `30074938--30074939`: node024;
- Python `30074940--30074941`: node026;
- Python `30074942`: node022.

Every allocation resolved to `allcs/lowprio`, one RTX 3090, eight CPUs, and
64 GiB. Startup logs contained no traceback, runtime error, CUDA OOM, or NCCL
failure. Live treatment telemetry crossed optimizer step 6 for Countdown and
step 2 for Python, with finite model entropy, observation counters, and the
registered warmup alpha of 0.10.
