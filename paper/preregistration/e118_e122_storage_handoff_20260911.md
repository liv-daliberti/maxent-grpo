# Pending E118 storage reservations yield to two E122 starts — September11,2026

The user requested another couple of E122 jobs now. Temporarily defer only six
currently pending E118 Qwen3B allocations:31151400,31124282,31124283,31124285,
31193187,and31048143. These jobs retain all occupied checkpoints and their exact
IDs,scientific submissions,source/runtime snapshots,seed,run paths,resources,
walltime,Nice,restart history,and effective afterany dependencies. No running GPU
allocation is held,canceled,requeued,or replaced; no checkpoint or metadata is
removed. No scientific outcome selects the queue order. The six rows remain
registered and visibly held pending automatic storage admission.

The read-only shared classifier counts every known nonheld GPU writer,including
released pending allocations. It recognizes complete current-job checkpoints
from bounded ZIP metadata and file sizes; one future rounded checkpoint is
reserved when this exact job ID has a complete checkpoint,two otherwise. Existing
occupied files are already included in statvfs usage. Unknown writers,conflicting
identities,unsafe paths,over-capacity E122 release counts,and ambiguous allocation
states block release. E122 retains125GiB for all100terminal exports,96GiB for six
16GiB peaks,and64GiB shared headroom throughout the handoff. Each proposed E118
release is forced back into the current reservation calculation before admission.
E122 pair completion alone never overrides a failed full-storage calculation.

Preparation records identity/checkpoint/resource proofs and a hypothetical
six-hold budget without changing Slurm. After independent plan review,one exact
CPU supervisor is submitted held and registered. At the existing E118 ledger lock
and new shared storage admission lock,persist each GPU hold intent,hold only an
unchanged pending job,and verify its exact owned hold. A start race removes only
this operation's hold from the unchanged running allocation. Other failures or
uncertain acknowledgments preserve receipts for reconciliation; no command is
blindly repeated. Existing Pantry timeout guards explicitly leave deliberate
pending holds under their owner's control. Their unresolved retry state is
checked before31048143is held; no guard file is modified.

The CPU supervisor requests2CPUs,2GiB,noGPUs,25h10min,on915/917. It polls at60s,
renews only its own CPU ID after23h,allows at most eight CPU restarts,and never
extends its absolute seven-day deadline. It holds a singleton and registers a
fresh heartbeat. Before each individual release it acquires the shared storage
admission lock used by the E122 pair release package,then the E118 ledger lock;
checks frozen resources,source,checkpoint,canonical identity and exact owned hold;
recomputes full shared storage with the candidate counted as a writer; and releases
only when approved. Preserve an afterany dependency's natural fulfillment rather
than replacing or clearing it. Preserve Nice; Slurm decides which eligible job
starts. Released GPU jobs are never automatically held again.

All mutation intents and acknowledgments are durable. An ambiguous release is
charged in future admission checks until exact scheduler state resolves it; it is
never blindly reissued. Unexpected identity/resource/checkpoint state leaves the
affected hold for review. Transient storage/query failures permit later read-only
polls. At the seven-day deadline,no new release is allowed; remaining owned holds
are explicitly reported for manual review. Current occupied checkpoint state is
preserved at every exit. The shared classifier is a conservative admission test,
not a filesystem reservation; other admission controllers must coordinate through
the same lock or remain inactive during this priority handoff.

The seventh owned hold is the existing CPU-only E124 waiter31164037. Pause it
before any E118 hold, after acquiring E124's existing controller mutation lock
without waiting and verifying all31canonical E124 GPU jobs remain originally held.
This excludes an in-progress E124 GPU transaction. Prefer an ordinary pending-job
hold; if its CPU allocation is running, journal one exact same-ID requeuehold and
verify the CPU becomes inactive before proceeding. Preserve its submitted command,
requested CPU resources,original gate,controller source,scientific plan,and all GPU
holds. Record the expected restart increment only for this CPU requeue. Never
repeat an ambiguous pause. This is temporary admission scheduling under the user's
E122 priority request; no E124 scientific or GPU operation changes.

Restore the E124 CPU last, after every E118 hold actually acquired by this package
has been returned, only if a fresh full E122 shared budget also leaves an additional
220GiB for E124's next-run peak. Its unchanged controller then resumes its original
admission checks. If setup fails after acquiring only a subset of holds, the
registered observer can still activate and safely restore those owned holds; it
never acquires the remaining holds automatically. Durable unresolved intents block
restoration until observed state resolves them. The seven-day deadline also bounds
the E124 CPU restore. No E124 gate or transaction file is edited by this observer.
