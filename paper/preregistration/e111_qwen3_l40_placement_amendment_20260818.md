# E111 Qwen-3B L40 placement amendment

Date frozen: 2026-08-18, before changing any E111 Qwen-3B placement and
without inspecting an E111 endpoint evaluation result.

## Motivation and scope

This amendment applies only to the five pending Qwen-3B mechanism-gate jobs:
`30674758`, `30674759`, `30674760`, `30674761`, and `30674762`. Repeated
preemption on A6000 nodes 103–104 prevented the jobs from reaching their
recovery boundary, and Slurm subsequently projected starts on 2026-08-19.
Node403 is presently idle, has eight 48-GB L40 GPUs, and is in the same
`lowprio` partition.

E111 is a mechanism gate. It does not gate on reward, accuracy, coverage, or a
paired endpoint effect. The full E112 efficacy evaluation remains frozen to
paired A6000 placement. Therefore this amendment may change E111 device model
for completion latency while leaving all endpoint-grade comparisons untouched.

## Frozen scheduler-only change

For the five exact job IDs, while pending:

- required node list: `node[103-104,205-208]` -> `node403`
- generic resource: `gpu:a6000:1` -> `gpu:l40:1`

Partition `lowprio`, account `mltheory`, one GPU, 16 CPUs, 128 GB host-memory
request, two-hour limit, job IDs, run directories, submitted environment, and
stored batch scripts remain unchanged.

## Invariants

- No job is canceled, requeued, reset, duplicated, or edited while running.
- Model, seed, data, prompt order, optimizer, learning-rate schedule, MaxEnt
  estimator/coefficient, ReplayDr objective/weight, proposal policy, verifier,
  evaluation settings, checkpoint state, and target step count are unchanged.
- The runtime-ops durability amendment remains active on future allocations.
- E111 remains outcome-blind with respect to endpoint metrics and PointMaze
  remains excluded.
- E112 retains its paired A6000 hardware rule.

## Audit rule

The amendment record must contain exact before/after `scontrol show job -o`
records for all five jobs, unchanged submit lines, the E111 ledger and protocol
digests, node403 capacity evidence, and exact old/new node and GRES fields. The
final E111 auditor must validate it.
