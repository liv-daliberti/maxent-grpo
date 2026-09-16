# Bounded hourly MathIR timeout recovery — September9,2026

The user requested recovering broken workloads and accelerating completion.
This guard covers only the two staged hourly MathIR seed72 replacements of
jobs31158503/31158504. Initial model/optimizer checkpoints are768(MaxRL) and
960(ReplayMaxRL). Capacity staging determines the exact new Slurm IDs.
The hourly recipe preserves scientific/evaluation settings; its sole runtime
change is rolling resume checkpoint cadence192→48. The existing long jobs
are intended to remain held as audited dormant fallbacks; their activation
transaction is owned by the capacity controller.

Freeze source, exactSubmitLine, full exports, source and aggregate identity,
all effective resources, initial restart counts, and initial durable progress
floors. Permit only a same-ID requeue of an inactive, accounted TIMEOUT with
an intact checkpoint strictly newer than the last resumed checkpoint. The
very first retry must exceed768/960; a missing prior retry never exempts it.
Require directory tag on the48-step grid, complete model and optimizer ZIP
directories, readable CRC-checked pickle metadata, modelglobal_steps,
global_step,policy_sgd_step,prompt_batches_consumed_total and optimizerstep
all equal to that directory tag, plus retained replay-bank state. Inspect
pickle opcodes and metadata only; never deserialize tensors or execute pickle.

Require exact scientific mapping and no unapproved active/pending writer.
Never requeue RUNNING,CONFIGURING,COMPLETING,SUSPENDED or PENDING jobs.
Record durable intent before one requeuehold call; release only its audited
owned hold with exactly one incremented restart count, unchanged recipe and
resources, and revalidated advancing counters. Stop on uncertain operations,
changed source/resources/identity, missing/corrupt counters or other failures.

Limit retries to24 per cell and a persisted absolute24-hour retry deadline.
After the retry deadline, initiate no new hourly allocations; monitor an
already running final allocation through its fixed one-hour limit before
completion or coordinated fallback. The CPU supervisor is bounded separately
at25h10m to permit this final cleanup. A no-progress checkpoint, exhausted
cap or retry deadline must not leave the cell silently stranded: preserve
its complete original72-hour route and use only the separately reviewed
capacity-owned atomic fallback transaction. It must make the hourly job
inactive, restore both authoritative E118 mappings and release only the
exact previously held long fallback, ending all hourly retries. No new
science submission or overlapping writer is permitted.

Actual staged IDs and all helper/source hashes are frozen in the guard plan
before launch. The capacity controller exposes fallback(new_job_id,reason)
and retire_completed(new_job_id), both called under the shared E118 ledger
lock. Only its exact dormant predecessor, still in the audited JobHeldUser
state, is exempt from the duplicate-writer check. At terminal completion,
retire_completed cancels that exact unused hold. Each item has independent
fallback state so restoration of one long route cannot stop its hourly peer.
No guard or science mutation is launched by writing this protocol or running
local tests.

An owned requeuehold transition initiated before the retry deadline may finish
its one already committed release during the first five minutes of cleanup.
Require a persisted pre-deadline intent, the exact owned requeue hold and
Restarts increment, plus all progress/counter/resource/writer checks. This
exception never calls requeuehold again and leaves a full hour for that
allocation before the65-minute cleanup limit. Later reconciliation stops
for explicit review. A deadline hold racing a genuine allocation is released
unchanged by the capacity helper; the guard continues observing it until
inactive and does not initiate a competing fallback writer.
