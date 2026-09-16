# E106 Falcon A6000 all-partition pool amendment

Frozen before application on 2026-08-17 EDT while both target jobs were
`PENDING` at runtime `00:00:00`. This amendment used scheduler state,
hardware inventory, and previously frozen capacity records only. No E104 or
E106 post-update evaluation outcome was inspected. PointMaze is excluded.

## Motivation and capacity evidence

Falcon E104 Pantry job `30637794` and repaired E106 Python job `30640330`
both request one 48-GB A6000, 8 CPUs, and 64 GiB, but are restricted to
`cs` and `node[205-207]`. The cluster exposes the same A6000 GPU type in
partition `all` on `node[103-104,205-208,805]`.

Three full 3,072-step Falcon Python runs already completed on nodes 205, 206,
and 207. In addition, the exact larger Qwen-3B v6 loss stack completed its
registered update-only A6000 capacity job `30638185` on node208 with twice
the target memory request. These records establish both Falcon capacity and
the repaired loss stack on this GPU family. The application script also
records the current `sinfo` inventory for every authorized node.

## Authorized scheduler-only change

For jobs `30637794` and `30640330`, and only while each remains pending
with zero runtime and its exact frozen scientific environment, change:

- partition `cs` to partition `all`; and
- node list `node[205-207]` to `node[103-104,205-208,805]`.

Retain account `allcs`, one A6000, 8 CPUs, 64 GiB, the one-hour limit,
priority, model, source and ops snapshots, domain, seed, data, prompt,
optimizer, group size, replay and semantic settings, stopping rule,
checkpoint and evaluation cadence, output path, auto-resume, and requeue
behavior.

The application is transactional: if either update or the post-update audit
fails, every job changed by that invocation is restored to `cs` and
`node[205-207]`. A content-addressed artifact records the protocol, script,
ledgers, prior pool amendment, capacity evidence, hardware inventory, and
before/after scheduler records.

## Gate consequence

This amendment changes placement eligibility only. Both cells must still
reach optimizer step 64 and satisfy every registered E104/E106 mechanism
criterion. It cannot relax the combined gate or authorize outcome inspection.
E105 and E109 remain locked until all fifteen effective cells pass.
