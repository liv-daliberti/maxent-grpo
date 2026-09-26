# Bounded prospective timeout guard — September 8, 2026

The user requested requeueing broken E118/E119/E120 jobs and accelerating
completion. Four currently running Pantry cells retain their original
allocation limits: E118 Qwen2.5-3B jobs 31048143, 31048144 and 31048145 have
12-hour limits; E119 Qwen2.5-0.5B Level-2 job 31048182 has a 36-hour limit.
This prospective operational guard handles future scheduler timeouts
without interrupting their current training allocations. It does not
inspect scientific evaluation outcomes.

The site rejects all post-submission walltime changes, including for held
pending jobs, with “Walltime may not be modified after submission.”
Consequently this guard retains the exact original 12/36-hour limits and
only requeues an inactive TIMEOUT under its existing Slurm ID. Requeueing
starts a new allocation clock while retaining the existing job registration
and scheduler-controlled queue accounting. There is no new submission,
walltime update, pending-job hold or experiment-ledger modification.

Freeze the exact live SubmitLines, source-launcher hashes, scientific cell
identities, effective resources, initial restart counts and allocation
limits in `var/artifacts/campaign_timeout_guard_20260908/plan.json`.
Preserve all exported variables, resources, eight-pass/3072-step targets,
checkpoint cadence, queue priority policy and original walltime limits.

`ops/exp_scaling/guard_campaign_timeouts_20260908.py prepare` freezes the
reviewable plan without scheduler mutation. `once` is read-only by default;
`once --apply` performs one inspection-and-repair pass. `watch --apply`
checks every 30 seconds for at most 48 hours from its first applied start,
retaining that deadline across restarts. A singleton lock prevents concurrent
guards; the experiment-ledger promotion lock protects each identity check
and possible operation. The guard may run in a separate CPU-only allocation.

Never requeue, hold or alter a RUNNING, CONFIGURING, COMPLETING, SUSPENDED
or PENDING allocation. Deliberate pending holds and ordinary queued jobs
remain under their existing owner. A terminal TIMEOUT is eligible only
while its same Slurm ID still has a controller record, accounting confirms
TIMEOUT, the ID is absent from the active queue, there is no terminal
training receipt, the latest saved model/optimizer checkpoint validates
and no other active or pending job can write that run directory.

Allow at most three guard-initiated requeues for each 12-hour E118 target
and at most one for the 36-hour E119 target during the bounded watch.
Require a valid checkpoint strictly between steps 0 and 3072. After a
previous guard retry, the next saved checkpoint must be strictly newer
than the checkpoint used for that retry. A repeated or regressed checkpoint
stops the target for manual inspection, preventing an unproductive loop.
The fixed retry caps and unchanged 3072-step target bound all continuation
work; a completed cell is never restarted.

Reconfirm eligibility immediately before one same-ID `scontrol requeuehold`
call for each authorized retry. Write durable intent and preserve the
preceding scheduler record before that mutation. Reconcile the resulting
owned hold only when its reason is `job_requeued_in_held_state`, its restart
count has increased by exactly one, and its frozen recipe and all resources,
including TimeLimit, remain identical. Treat Slurm's `NumNodes=1` and
`NumNodes=1-1` displays as equivalent exactly-one-node allocations; reject
any larger node count. Recheck the authoritative ID, latest
valid nonregressing checkpoint and sole-writer condition, then release only
that owned hold. Record the released checkpoint step before another retry
can become eligible. Durable intents and scheduler evidence permit
reconciliation after an interrupted requeuehold or release; never blindly
repeat an uncertain requeue command.

A missing controller record, a changed recipe/resource/identity, a missing
or nonadvancing valid checkpoint, another writer, an exhausted retry cap or
any other terminal error produces an explicit manual-stop event, without
blind replacement submission. A terminal receipt finishes that target for
the guard. Existing unrelated workloads remain outside its scope.
