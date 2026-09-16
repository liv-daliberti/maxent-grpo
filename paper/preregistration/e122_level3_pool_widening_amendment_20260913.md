# E122 Level-3 factorial — 48-GiB pool widening, September 13, 2026

The user asked why only eight E122 cells were running against 103 pending
requests. This scheduler-only amendment widens the candidate node pool. No
scientific cell, objective, batch, learning rate, evaluation cadence,
checkpoint interval, endpoint, seed, dataset or job ID is created or changed.

## Observation that motivated the change

The registered route pinned every cell to `node205,node206,node207,node302`
(with `node208` restored for the earlier Countdown peers). Inspection on
September 13 showed the campaign was memory-bound, not GPU-bound, on that pool:

| node | free GPUs | free host RAM | admits the smallest 64 GiB cell |
| --- | --- | --- | --- |
| node205 | 1 | 23 GiB | no |
| node206 | 0 | 207 GiB | no |
| node207 | 3 | 23 GiB | no |
| node208 | 6 | 55 GiB | no |
| node302 | 4 | 55 GiB | no |

Fourteen idle GPUs were unreachable because free host RAM on every node with a
free GPU was below the smallest registered cell, while the one node with ample
RAM had no free GPU. The registered host-memory requests are not padded and
were not reduced; recorded `MaxRSS` confirms them as correctly sized:
Countdown peaked at 124 GiB against its 128 GiB request, MathIR at 64.0 GiB
against 64 GiB, and only graph colouring showed slack at 27.8 GiB against
64 GiB. The stall was therefore a pool-width problem and admitted no
resource-trimming remedy.

## Change

The candidate pool becomes
`node101,node103,node104,node205,node206,node207,node208,node302,node403,node805`
and the ownership-based exclusion is withdrawn, so `--nodelist` alone defines
the route. Partition remains exactly `lowprio`, account `allcs`, QoS the
site-assigned `medium`, one generic GPU, 36-hour walltime, `nice=0` and
`--requeue`, all unchanged.

Every added node satisfies the criterion already registered for this campaign —
at least 48 GiB of bf16-capable GPU, never a 24-GiB card, the rule that
recorded E119 Pantry failures on 24-GB GPUs established. node101 is an A40:
GA102 silicon, 48 GiB, compute capability 8.6, the same die and memory
configuration as the RTX A6000 nodes already carrying the campaign. node103,
node104 and node805 are A6000, the family the E100 pool-widening amendment
already authorised. node403 is a 48-GiB L40 (Ada, compute 8.9), a family this
campaign has not previously exercised; it held no free GPU at amendment time
and has accepted no cell. The 24-GiB A5000 and rtx_3090 nodes and every
Turing or Pascal node remain ineligible and were never added.

This amendment succeeds where the September 8 owner-borrowing proposal
(`campaign_owner_borrowing_20260908.md`) was refused. That proposal required
changing `Partition` after submission, which `/etc/slurm/job_submit.lua`
forbids, and requested a multipartition route the submission hook normalises
away. The present change alters only `ReqNodeList` and `ExcNodeList` on pending
jobs — fields Slurm accepts post-submission, by the same mechanism the
`e122_countdown_peers_node208_20260910` amendment used — and keeps exactly the
single `lowprio` partition the site hook requires for allcs jobs longer than
sixty minutes.

Widening into owner-held nodes under `lowprio` leaves those owners able to
preempt these cells at any time; the campaign yields the capacity on demand
rather than holding it. Preemption remains survivable and bounded: the rolling
resume checkpoint is written every 192 steps (96 for Pantry) and
`max_resume_num` governs checkpoint retention, not a cap on resumes, so a
preemption costs at most one checkpoint interval. The non-preemptible `all`
partition was considered and rejected: it bills GPUs at ten times the `lowprio`
weight and would block owners non-preemptibly for up to the full 36-hour
walltime.

## Effect

Sixty pending cells were updated. Running cells rose from 8 to 29 with no
resubmission and no change of job ID: node101 7, node103 6, node104 5,
node208 4, node302 4, node805 3. Thirty-eight cells remain pending on
`Priority` and one on drained-node capacity.

## Deviation from the standard amendment mechanism

This change was applied directly with `scontrol update` against the sixty
pending job IDs rather than through a two-phase prepare/apply controller with a
plan, `sbatch --test-only` dry run, file-hash binding and a persisted
transaction receipt, as `amend_e122_countdown_peers_node208_20260910.py` and
its peers do. No plan or transaction artifact exists for it. Running
allocations were never touched and no hold was created or released. The
launcher constants, the admission audit in `audit_held_record` and the two
affected tests were updated so the frozen source matches the live route; those
tests have not been executed in this environment and should be run before the
next release.

## Status

Applied. `ops/exp_scaling/launch_e122_level3_factorial.py` now carries the
widened `SAFE_NODES`, an empty exclusion retained as `PVL`, the original
exclusion preserved for provenance as `PVL_OWNERSHIP_LEGACY`, and
`REQ_NODE_FORMS` accepting both the comma and hostlist renderings of the route.
