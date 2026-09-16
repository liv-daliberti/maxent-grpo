# E122 additional Graph pair — September 10, 2026

The user explicitly requested another couple of E122 jobs after three were
running. This new operational amendment releases exactly the existing held
Graph seed43 Dr.GRPO31158698 and ReplayDr.GRPO31158699 jobs. Selecting the paired
replay comparison and lower64GiB resource class does not inspect efficacy.
Preserve the100-cell factorial,model/data/source,all exports,3072-step horizon,
oneGPU,8CPUs,36-hour request,evaluation/checkpoint policy,and all originalIDs.
After the original64GiB held audit and release, reduce only the scheduler host
memory request to40GiB using the separately pinned completed E119 Graph40GiB
qualification (27.52GiB measured MaxRSS;12.48GiB margin).
No new training submission occurs.

Full frozen campaign authentication is performed once per retained process,
followed by the original cached full-input verification before preparation and
application. Original exact held-job audits remain mandatory immediately before
release. A fresh external storage admission must enumerate every own nonheld
GPU allocation, reject unknown writers, and reserve all external checkpoint
peaks plus E122's64GiB shared headroom,125GiB unfinished terminal exports and
6x16GiB full checkpoint-replacement peaks. New unaccounted GPU writers or reduced
free space stop release. Two extra jobs temporarily raise the admitted count to
six. The unchanged persistent controller retains cap4 and counts these same
original-format journals until its ordinary policy allows further releases.

Preserve the existing watcher through a parent-owned,non-inherited shared flock.
The release child is initialized before lock acquisition; wait for a fresh
healthy watcher heartbeat before the parent takes the lock. The parent performs
only local pipe/process/timer operations while holding it; all scientific file
reads,filesystem scans,durable journals and scheduler calls run in a separate
process. Give that child13seconds and at most8seconds for bounded,read-only
scheduler reconciliation under a24-second total budget. Kill the process group
without an unbounded wait before releasing the lock. A worker never inherits
the parent's lock descriptor. Complete immutable JSON is published atomically,
so interrupted temporary files are never mistaken for final journals.

The unchanged watcher creates each immutable status filename from the UTC time
immediately after its status scan. It writes complete JSON and fsyncs the file
and parent directory while owning the legacy lock, then returns, prints, and
sleeps60seconds. Pin that original source and actual60-second watch command.
Use the timestamp embedded in the latest immutable status filename as the
publication basis. Force server directory and file metadata, read exactly the
published byte count, require newline-terminated complete JSON, and restat with
unchanged size and mtime. Retry writes that race this read-only observation.
Verify campaign binding and healthy status; retain the scheduler snapshot time
with snapshot no later than filename stamp (1ms rounding allowance).
Accept filename-stamp age at most15seconds, and at most16seconds after opening
and acquiring the lock. Briefly retry a nonblocking lock while an observed
publisher is finishing fsync, entirely within that16-second window. No worker
receives GO before this succeeds. The next legacy cycle cannot begin before
filename stamp plus60seconds; at least44seconds therefore remain for the
24-second parent budget, leaving at least20seconds of collision margin.
Stdout mtime is not used: its non-fsynced append showed delayed NFS writeback.
The two unused preparations and their source snapshots remain in the dated
retired_attempt directories; neither reached an execution intent or GPU mutation.

Only an actual successful scheduler-command acknowledgement captured from the
child before its receipt write may finish a missing receipt after a killed
child. Pass that exact acknowledgement from parent memory; current scheduler
state must corroborate it. Scheduler state alone never fabricates a process exit
code. Do not repeat release commands,remove consumed intents,or
invent acknowledgement for a still-held job. Any ambiguous or partial outcome
requires reconciliation and preserves the original fail-closed policy. A fresh
subsequent original heartbeat and exact readbacks are required for successful
handoff. No SSH access,new CPU watcher,or old-controller cancellation is needed.

After successful release and outside the shared controller lock,add node105,
node203,node204 and node208 to each still-pending job's original205/206/207/302
pool, and amend its host-memory request to40GiB. Retain all other resources.
Slurm excludes temporarily unavailable nodes from admission as usual. E119's same
Qwen0.5B Graph runtime completed all20cells on24GiB A5000 hardware with identical
model revision,context,response,microbatch,evaluationbatch andvLLM fraction.
Preserve any allocation already started and all non-placement resource fields.
Slurm enforces each qualified40GiB request; eligibility does not promise an
immediate start when other finishing experiments occupy the available memory.
The prior four-cell burst remains blocked and is never invoked by this amendment.

Coordinate the final budget check and pair release/resource amendment with the
E118 temporary-hold observer using the new shared_storage_admission_20260911.lock.
That observer never takes the original E122 watcher lock. The parent holds the
shared storage lock until both released jobs are counted and resource amendments
are complete, preventing a held E118 writer from being released against the same
headroom concurrently.

Immediately before worker startup, refresh the full shared-storage report under
the shared storage lock and pin that immutable final proof. Check current writer
IDs and free bytes against it within the bounded child; perform no full external
checkpoint scan inside the legacy lock. Before claiming an execution intent,
run the entire child read-only status/storage/held-audit sequence outside that
lock and require a measured runtime of at most6.5seconds, leaving deadline margin
for the two small journaled release operations. Persist its actual timing.

Preparation itself now includes and publishes the live read-only worker timing
for root review before any application. That request is marked read-only and
cannot invoke a releasing worker. Rehearse again after the final fresh budget
check during application to detect changed filesystem or scheduler latency.

Immediately before each resource update,re-read the full pending request and
compare it with the audited snapshot. Preserve a job that has started. Slurm
25.11.8 itself rejects a minimum-memory change for a non-pending job; if that
remaining race causes rejection and the allocation has started,verify unchanged
resources and record preservation without retry.

Bind the completed six-hold package and its registered observer31220594. Require
its original hold_scope_complete receipt, valid current hold/release states, and
a fresh healthy heartbeat. Before each new E122 release,recheck that independent
E124 CPU coordinator31164037 remains in this exact journaled inactive hold with
unchanged submitted command/resources and the owned restart count. Its legitimate
job_requeued_in_held_state reason is accepted only with that exact ownership,
PENDING state,priority0,no allocation and zero runtime. The observer alone later
restores that coordinator when its conservative extra budget allows it.

Read-only preparation and rehearsal may run after the complete owned holds and
registered observer submission while that CPU awaits allocation. Application
still requires its actual running state and fresh healthy heartbeat.
