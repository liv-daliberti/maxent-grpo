# E119 healthy GPU completion guard — September 9, 2026

The user requested faster completion of E119 and recovery of broken work. This
operational guard covers only the ten existing E119 Pantry cells moved through
the node208 transaction/healthy-node amendment and the eight-cell healthy A6000
transaction. Preparation freezes their final authoritative successor IDs after
all ten promotions and releases have completed. No new scientific cell, seed,
method, optimizer setting, evaluation setting, model, data, run directory,
checkpoint cadence, source snapshot, resource request, or node pool is introduced.

One separately registered CPU supervisor polls at 60-second intervals. Its
absolute deadline is seven days after preparation; restarting the supervisor
never extends that deadline. It may issue at most three same-ID requeuehold
operations per cell. It never creates successor jobs or edits continuation maps.

An automatic retry requires an inactive, accounting-confirmed TIMEOUT and a
complete model-plus-optimizer checkpoint with matching global/prompt counters,
strictly newer than the previous resume floor (initially the route's frozen
checkpoint). Terminal completion suppresses retries. FAILED, OUT_OF_MEMORY,
NODE_FAIL and other terminal states require manual review. Ordinary Slurm
preemption/requeue and the frozen runtime watchdog remain separate mechanisms.

Before every mutation, under the existing ledger lock, verify the authoritative
cell mapping, unchanged scientific exports/source/resources, exact dormant
predecessor hold, and absence of any other writer. Persist requeuehold intent
before the command. An ambiguous requeue is never repeated. Release requires the
exact expected restart increment, the owned requeue hold, the same validated
advancing checkpoint, and unchanged memory, walltime and the final registered
node/GPU request. Preparation accepts nodes205/207 (A6000) or the reviewed broader
205/207/302 pool (including the already-qualified A100 route); each item freezes
its actual scheduler profile after all routing amendments. Any
post-requeue normalization back to node208 fails closed with the job held.

No new retry or held-job release is permitted after the absolute deadline. A
same-ID action left held at the deadline is reported for manual review. A fresh
status artifact records every monitoring pass. The supervisor records its own
identity separately and uses a singleton lock. Routine CPU supervisor requeues
may preserve the absolute deadline; at most seven are allowed. All existing
campaign guards and shared helpers remain unchanged; the mutable presentation
helper campaign_stats.py is not a hash dependency of this guard.

Scheduler commands have a120-second process-local timeout. Transient scheduler
command/query errors receive up to three consecutive observation attempts;
unresolved mutation intents remain durable and are reconciled without repeating
an uncertain requeue. A successful observation resets the transient-error count.
Scientific/resource validation failures stop the affected cell immediately.

The proposed CPU entrypoint requests one CPU,2GiB RAM, no GPUs, and a25h10min
allocation on node915 or node917. The supervisor renews at23hours while preserving
the absolute seven-day deadline. Checkpoint metadata validation measured37,760KiB
peakRSS in the read-only pilot. Submission is held; a separate registration binds
the exact CPU job ID, entrypoint hash, submission and resources to the scientific
guard plan before the CPU job is released. Preparation and registration never
submit, release or otherwise mutate scheduler jobs.
