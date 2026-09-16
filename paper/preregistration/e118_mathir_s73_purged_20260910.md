# E118 MathIR ReplayMaxRL seed73 manual checkpoint recovery — September 10, 2026

The user requested restarting the failed existing workload and accelerating
completion of E118/E119/E120. E118 Qwen3B MathIR ReplayMaxRL seed73 hourly job
31159778 timed out, then disappeared from the Slurm controller. Accounting
retains TIMEOUT. Its bounded hourly guard recorded `manual_stop` because a
same-ID requeue cannot proceed after that controller record is purged.

Restore its exact owned-held original long allocation 31158506. Preserve its
job ID, frozen command, source/model/data, treatment, seed, optimizer, replay
state, run directory, 3072-step horizon, 1 A5000, 16 CPUs, 116 GiB, lowprio partition,
node105/node202/node203/node204 pool, 72-hour limit and exclusions. Returning to
the original command restores its 192-step resume-checkpoint interval from the
hourly command's 48-step interval; no scientific treatment changes. Record the
specific reason `controller_record_purged`, rather than mislabelling this as
lack of checkpoint progress or an exhausted retry/deadline budget.

The new standalone controller prepares a reviewable plan without scheduler
mutation. It uses existing frozen validators to verify the authoritative
source/aggregate mapping, owned hold, complete frozen SubmitLine, Slurm TIMEOUT
accounting, absent controller record, absence of another writer, absence of a
terminal receipt, and latest valid checkpoint step 1920. Inspect model, optimizer,
prompt-progress counters and the saved replay-bank state. Snapshot exact input
hashes, checkpoint file metadata and original coordination files.

Applying the prepared plan under the existing E118 ledger lock stages and
promotes source and 150-cell aggregate mappings before releasing job31158506.
Persist release intent, verify the released allocation, and reconcile the
existing deployment and manual-stop guard to `fallback_complete`. Preserve
all unrelated rows and historical events. Submission of a new job, cancellation
of another allocation and modification of frozen helper code are outside scope.
Interrupted promotion/release must reconcile exact recorded before/after states;
never blindly repeat an uncertain scheduler action. Observe checkpoint restore
and subsequent optimizer progress separately after release.
