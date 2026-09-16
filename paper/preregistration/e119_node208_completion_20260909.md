# E119 node208 completion routing amendment — September 9, 2026

The user requested additional E119 progress and use of other eligible capacity.
At approximately 19:00 EDT, node208 returned to an idle, schedulable state with
ten A6000 GPUs and 503 GiB of schedulable host RAM. Exact lowprio/mltheory
requests for existing E119 Pantry recipes forecast immediate starts.

This is an operational placement amendment, not a new scientific cohort.
Preserve each cell's training seed, method, frozen source and shell entry point,
all exported training/evaluation settings, model/data paths, run stamp, output
directory, full-state automatic resume policy, target 3,072 updates, and existing
36- or 72-hour walltime. No partial outcome or score selects a configuration.

Route pending, unguarded E119 Pantry cells to node208 through lowprio/mltheory.
The initial four are MaxRL seeds44,45,46,47, with authoritative predecessor jobs
31048181,31124279,31048187,31048191. Seed45 retains 128 GiB host RAM and its
72-hour walltime. The other three receive 116 GiB and retain 36 hours. This
reserves 476 GiB total and four GPUs; the scheduler enforces node capacity.
Additional unguarded queued cells may use the same 116 GiB route as capacity
becomes available: Dr.GRPO seed44 (31048180), ReplayDr.GRPO seed45 (31037836),
ReplayMaxRL seed45 (31048185), ReplayMaxRL seed46 (31048188), Dr.GRPO seed47
(31037843), and ReplayMaxRL seed47 (31037846).

The 116 GiB request increases host headroom relative to the former 96 GiB
allocation. Preserve the existing physical microbatch, offload settings,
generation settings and A6000-compatible vLLM ratio0.25. Existing 128 GiB
requests are never reduced. Inspect actual memory availability and GPU health
on the first allocation before filling the route. If unexpected faults recur,
preserve checkpoint state and stop the affected operational expansion for review.

Site policy requires new jobs for account/partition changes. Recheck each
predecessor is still pending and authoritative, then hold it before cloning.
Persist every mutation intent before execution. Submit replacements held,
audit exact scientific exports and source/script identities, and promote one
continuation row under the existing ledger lock. Preserve the 100-cell
scientific denominator and 75-row continuation ledger; keep each predecessor
as an explicitly owned dormant hold until the successor is established.
Never duplicate an active writer. An ambiguous submission is reconciled from
its recorded identity rather than blindly retried. If a predecessor starts
during the hold race, preserve that allocation and skip its replacement.

Use the existing complete checkpoint when present. A pending cell without a
committed checkpoint retains its existing automatic fresh-start behavior;
partial files never count as a validated checkpoint. Do not interrupt any
running E119 cell to create these allocations. In particular, leave the
slowly progressing Dr.GRPO seed43 writer intact. Guarded MaxRL seed43 and
ReplayMaxRL seed44 are excluded from this initial routing transaction.
