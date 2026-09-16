# E119 remaining Pantry jobs gain restored node208 — September 10, 2026

The user requested wrapping up E118/E119/E120 while starting E122 today. Expand
only the requested node pool of nine currently inactive, pending E119 Pantry jobs:
31163339,31163560,31163584,31163595,31163624,31163665,31163676,31163683,31163710.
Add208 to their existing205/207/302 generic-GPU pool. Preserve IDs,116/128GiB RAM,
8CPUs,36/72hour limits,priority,each job's frozen current restart count,all exports,frozen source and runtime,checkpoints,
scientific identity,100cells,and75continuation rows. Running MaxRL seed44 job
31163361 and all running GPU allocations remain unchanged. No reward or evaluation
outcome selects this operational expansion.

The currently registered ten-cell guard CPU31170078 is replaced with one held,
registered CPU successor using2CPUs,2GiB,noGPUs,and the existing25h10min renewal
policy. One handoff freezes all final retry state, resume floors and the original
absolute deadline September17 00:18:59.779878Z. Preserve the original singleton
lock, three-retry caps, bounded scheduler calls, source validation, exact dormant
fallbacks and all same-ID TIMEOUT recovery rules. Existing guard files remain
immutable. Only the nine ReqNodeList entries differ in the new guard plan.

Preparation and read-only verification never submit or mutate jobs. Record one
CPU submission with durable uncertainty receipt, audit the exact returned heldID,
and never repeat an uncertain submission. At the continuation-ledger boundary,
hold only still-pending unchanged target jobs with persisted intents. Stop only
the exact old CPU after verifying no unresolved retry, then acquire the shared
singleton and copy its final state. Amend each held target node list and its
continuation-row route provenance, verify all other fields, and register/release
the successor CPU. No GPU job is canceled or requeued by this amendment.

Release the nine owned GPU holds only after the new CPU has fresh monitoring for
all ten cells, node208 is healthy with a fresh physical A6000/48GiB/temperature
proof, and at least two of the three E122 peers31158679/680/681 are already RUNNING
on node208. This gives E122 the two currently free slots first while allowing E119
to backfill later releases automatically. The physical memory snapshot is recorded
but is not an admission promise; Slurm enforces each unchanged memory request.
Two jobs have one prior startup preemption; these current restart counts are frozen unchanged.
A target that starts before the initial hold is left untouched and requires new
preparation; uncertain scheduler actions retain durable receipts for reconciliation.
