# E104 static-smoke placement amendment

**Recorded before the affected jobs started and before any post-update
evaluation result was inspected on 2026-08-17.**

Node105 entered a thermal-drain state and the originally selected node202--207
and node302 queues would delay the mechanism-only smoke by days, while nodes
300 and 301 were idle.  The following still-pending E104 jobs may therefore run
on `node[300-301]` with one A6000 GPU instead of their initially recorded
placement:

- Qwen2.5-0.5B: jobs 30637785, 30637787, and 30637789;
- Falcon3-1B: jobs 30637790--30637794.

This is an execution-only amendment for the 64-update E104 mechanism gate.
The model checkpoint, immutable source/ops snapshot, data, prompt surface,
seed, optimizer, decoding, group size, objective, coefficients, telemetry
criteria, and stopping rule remain byte-identical.  E104 evaluation outcomes
remain excluded from the gate.  The full E105 paired runs retain each
registered ReplayDr.GRPO cell's original hardware placement; this amendment
does not authorize a full-run placement change.

Qwen2.5-3B remains on its originally registered A100 placement unless a
separate pre-outcome amendment is recorded after a capacity-only preflight.

## Execution record

The pending requests were briefly changed to `node[300-301]`, but scheduler
inspection showed that those nodes belong to partition `pci`, whose
`AllowAccounts=pci` policy excludes the submitting `mltheory` account.  No job
allocated or ran under the alternate request.  All eight jobs were restored to
the exact account, partition, node, and GPU type in the immutable E104 ledger
before continuing.  This failed capacity check has no effect on either E104 or
E105.

## Backfill time limit

Training-only telemetry from the already-running Qwen2.5-3B replay cohort
showed 16.2 seconds for its latest optimizer step.  Sixty-four updates therefore
take about 17 minutes; three registered evaluations still leave a large margin
inside two hours.  To permit backfill on the original nodes, the still-pending
Qwen2.5-0.5B jobs 30637785 and 30637787, Falcon3-1B jobs
30637790--30637794, and Qwen2.5-3B jobs 30637795--30637799 may have their
scheduler time limit reduced from eight hours to two hours.  Falcon is smaller
than Qwen2.5-3B and uses the same 64-update/evaluation schedule, so the same
bound is conservative for those cells as well.

This does not change the target of 64 updates, checkpoint/evaluation cadence,
watchdog settings, or any scientific stopping rule.  A job that fails to reach
step 64 still fails the original gate.  The full E105 time limits are
unchanged.

## Falcon A6000 backfill after live scale validation

**Recorded on 2026-08-17 after Falcon Countdown completed the mechanism gate,
before the four affected jobs started, and before any post-update E104
evaluation outcome was inspected.**

Falcon Countdown job 30637791 completed all 64 updates on its registered A5000
with v6 group centering and verified replay active, establishing that the
Falcon3-1B runtime fits the smaller GPU.  The remaining pending Falcon jobs
30637790 (Graph), 30637792 (Python), 30637793 (MathIR), and 30637794 (Pantry)
may therefore move from partition `cs` and their single registered A5000 or
A6000 node to partition `lowprio` and one A6000 on
`node[103-104,205-208,805]`.  Their account remains `allcs`; CPU, memory, and
the already amended two-hour limit remain unchanged.

This is an execution-only amendment to the outcome-blind E104 mechanism
smoke.  It changes no model, source/ops snapshot, data, static domain, seed,
optimizer, decoding setting, objective, coefficient, step target, evaluation
cadence, checkpoint cadence, or gate criterion.  The jobs retain requeue and
watchdog recovery.  E105 keeps its original matched placements.  No PointMaze
job is included.

### Falcon execution record

All four scheduler updates succeeded.  Immediate post-change inspection found
jobs 30637790, 30637792, 30637793, and 30637794 still `PENDING` with
`RunTime=00:00:00`, account `allcs`, partition `lowprio`, requested node list
`node[103-104,205-208,805]`, `gres/gpu:a6000:1`, and `TimeLimit=02:00:00`.
No affected process had allocated or executed before the amended request was
verified.

The resulting scheduler estimates were not materially better and exposed the
jobs to preemption: Graph/MathIR moved only a few hours earlier while
Python/Pantry moved later.  Before any affected job allocated, all four were
therefore restored to partition `cs`, account `allcs`, and their exact original
nodes/GPU types: jobs 30637790 and 30637793 to `node202`/A5000, job 30637792 to
`node207`/A6000, and job 30637794 to `node206`/A6000.  Immediate inspection
confirmed all four remained `PENDING` with `RunTime=00:00:00`; the two-hour
backfill limit remains the only active execution amendment.
