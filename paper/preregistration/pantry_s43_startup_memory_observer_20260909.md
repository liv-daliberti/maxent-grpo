# Read-only observer for repaired Pantry seed43 — September9,2026

Observe only E119 Pantry MaxRL seed43, job31048178, after the completed same-ID128GiB memory repair. Preparation requires the applied and released transaction, validated checkpoint960 and zero repeated optimizer updates. Preserve the existing observer31158674 and all its frozen files. This separate observer never changes scientific jobs, queues, priorities, resources or node state.

Wait for allocation for at most48hours; observe startup for at most6hours from actual allocation start, with an absolute54-hour bound persisted across observer restarts. Errors do not reset either bound. Pin the repair receipt, launcher, full exports and effective resource fields. Report drift and stop. Initial progress floor960 is replaced by the actual selected resume checkpoint if it is newer. Historical metrics are not evidence of resumed progress.

Verify fresh current-attempt optimizer updates, a newer structurally valid current-attempt model/optimizer checkpoint, and a successful read-only cgroup observation after that checkpoint. Report current/high/peak memory, noncache usage and OOM/high events; record whether current noncache remains below128GiB. Refresh memory periodically during startup and again after the checkpoint. An optional probe failure does not mutate training; retry observation within the finite budget. A validated terminal completion also satisfies the observation. Log activity, fatal markers and pending state remain explicit.

The separate CPU-only supervisor uses node915 with256MiB, noGPU and54h10m limit. No scientific allocation is created or repaired by this observer.
