# E69 A5000 placement-pool amendment

Date frozen: 2026-07-28, while Gate 2 Python successor job `30160205`
remained pending with `RunTime=00:00:00`, before it created a run directory or
any optimizer/evaluation record, and before any Gate 2 terminal result.

## Scheduler-only reason

The first executable-domain placement repair assigned all Countdown and Python
cells to `node105`, which has ten A5000 GPUs. Seven jobs started there, but the
eighth (`30160205`, Python `verified_route_successor`, seed 43) remained pending
with a scheduler forecast of `2026-07-31T13:11:51`.

At the amendment decision, `node202` was `IDLE` with ten unallocated
`gpu:a5000` devices. Both nodes advertise the same accelerator type
`gpu:a5000:10` and approximately 515 GB host memory. This is therefore a
placement-capacity repair, not a scientific change.

## Gate 2 action

Update only the never-started job's requested node from `node105` to `node202`
in place:

```text
scontrol update JobId=30160205 ReqNodeList=node202
```

The job ID, run stamp, frozen source and execution snapshots, data, seed,
variant, model, optimizer, rollout budget, fixed compute controls, replay
settings, six-pass stopping rule, evaluation surface, requested A5000 GPU,
CPUs, memory, account, partition, and time limit remain unchanged. The job is
not cancelled or duplicated.

## Gate 3 prospective placement

No Gate 3 job has been submitted. Future Countdown and Python Gate 3 jobs may
use the same-model A5000 pool `node105,node202,node203,node204` instead of only
`node105`. Every cell still requests exactly one `gpu:a5000`; accelerator class
and scientific compute matching remain fixed.

This amendment is based only on queue state and node inventory. It does not
inspect the pending Python successor outcome (none exists), select an arm or
seed, change any gate, or use MATH-500.
