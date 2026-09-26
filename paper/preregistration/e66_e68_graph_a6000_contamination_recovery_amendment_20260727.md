# E66/E68 Graph A6000 contamination recovery amendment

Recorded 2026-07-27 after the paired Graph drain-recovery amendment, after
E68 job `30130469` was assigned node206, and before this placement change.

## Observed infrastructure failure

E68 Graph seed 43 (job `30130469`) started on node206 with Slurm assigning
physical A6000 GPU 6. Before the actor could load the 0.5B model, vLLM
repeatedly reported CUDA out-of-memory and terminated. While Slurm was slowly
tearing down that allocation, seed 44 (job `30130470`) was assigned node206
and emitted the same pre-model-load OOM sequence. Both runs produced zero
optimizer metric records and no evaluation or training outcome.

A read-only diagnostic inside the still-active allocation showed the assigned
GPU using approximately 48.3 GiB. Every listed compute process belonged to
other users (`mi9937` or `rj5498`), not to the E68 job owner. The five foreign
processes reported 3,016, 2,372, 1,708, 8,088, and 33,156 MiB. This is
cross-allocation GPU contamination, not model memory demand or an algorithm
failure.

## Paired recovery

Fail closed on the entire `node205,node206,node207` recovery pool rather than
risk an unverified sibling GPU. Hold all six paired Graph jobs:

- E66 controls: `30128394`, `30128395`, `30128396`;
- E68 actuator: `30130469`, `30130470`, `30130471`.

Requeue-hold the running zero-step jobs `30130469` and `30130470`, then move
all six together to `mltheory/pvl-lowprio` on the same A6000 accelerator
family using eligible nodes `node103,node104,node805`. At amendment time
node103 was schedulable, node104 was idle, and neither carried a drain state;
node805 advertises the same A6000 family. Scheduler-only probes allocate no
GPU.

Job IDs, run stamps, source snapshots, data, arms, seeds, objectives,
controllers, proposal settings, sampling, optimizer state, evaluation,
checkpoint cadence, and every scientific setting remain unchanged. Existing
E66 checkpoints are retained. E68 Graph remains pre-optimizer.

The amendment artifact binds:

- exact affected jobs and same-family mutation;
- every job's pre-amendment step;
- jobs `30130469` and `30130470`'s zero-metric states;
- the exact byte length of each pre-amendment stdout/stderr log.

The auditors may classify uncaught signatures for jobs `30130469` and
`30130470` as the registered infrastructure interruption only before their
respective byte offsets. Any uncaught exception appended after the amendment,
including any later OOM, remains a hard failure.
