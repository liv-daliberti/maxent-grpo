# E106 Falcon Python A6000-pool amendment (2026-08-17)

## Scope

This scheduler-only amendment applies to E106 Falcon3-1B Python job
`30640330`. At registration the job was `PENDING`, runtime `00:00:00`, with no
training metrics. PointMaze is excluded. No other E104 or E106 job is changed.

## Capacity evidence available before the change

The exact Falcon3-1B Python family has already completed full 3,072-step runs
on each proposed node:

- seed 56, job `30516431`, node 205;
- seed 57, job `30516432`, node 206;
- seed 55, job `30516430`, node 207.

Nodes 205--207 are all in partition `cs` and expose the same A6000 GPU type.
The proposed E106 job retains one A6000, 8 CPUs, and 64 GiB. Scheduler state,
hardware inventory, and training telemetry were inspected; no post-update E104
or E106 evaluation outcome was inspected.

## Frozen change

Change only:

```text
ReqNodeList: node207 -> node[205-207]
```

Do not change the job ID, partition, account, GPU type or count, CPU count,
memory, time limit, model, seed, domain, data, prompt template, source or ops
snapshot, optimizer, objective, estimator flags, replay settings, output path,
or outcome-blinding contract.

## Gate consequence

The amendment changes placement eligibility only. Job `30640330` must still
reach optimizer step 64 and satisfy every preregistered E106 mechanism check.
E105 remains locked until the complete combined 15-cell gate passes.
