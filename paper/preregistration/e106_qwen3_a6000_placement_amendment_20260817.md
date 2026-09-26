# E106 amendment: Qwen-3B Python A6000 placement

Frozen on 2026-08-17 after E106 submission but before job `30640331` was
allocated, before any E106 post-update outcome was inspected, and while that
job remained `PENDING` with runtime `00:00:00`.

The inherited A100 placement on node302 received a scheduler estimate of
2026-08-20 17:48 EDT because the node's memory was reserved by long-running
jobs.  This is an execution constraint, not a scientific treatment.

The preregistered E104 capacity preflight job `30638185` used the identical
Qwen2.5-3B model, optimizer/memory recipe, rollout group size, v6 estimator,
and 128 GiB/16-CPU request on the low-priority A6000 pool
`node[103-104,205-208,805]`.  It reached optimizer step 1 on node208 with
finite centered-estimator telemetry and no OOM, then exited cleanly.  That
preflight did not satisfy its separate replay-application gate because the old
Python-unrelated sample supplied no replay group; it nonetheless establishes
the required device-memory capacity.  E106 differs from that runtime only in
the formatting-only `math_grader.py` parser repair, which cannot increase model
or optimizer memory.

Therefore job `30640331` may be changed, while still pending at zero runtime,
to:

- partition `lowprio`, account `mltheory`;
- node pool `node[103-104,205-208,805]`;
- one `gpu:a6000`, 16 CPUs, 128 GiB memory; and
- the existing eight-hour limit, checkpointing, requeue, and auto-resume rules.

Every environment variable, seed, data path, prompt surface, objective,
checkpoint, and output directory remains unchanged.  Preemption is an
execution event and may trigger the already-frozen resume path; it does not
relax the 64-step, verified-admission, replay-gradient, or estimator-invariant
gate.  If allocation begins before the update can be verified, the amendment
must not be applied.  PointMaze remains excluded.
