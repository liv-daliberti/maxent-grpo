# E118/E119/E120 pending scheduler acceleration — September 8, 2026

The user explicitly requested requeuing failed cells and getting all three
workloads finished as quickly as possible. This amendment addresses pending
allocations only. Failure continuations are recorded separately.

Preserve every scientific cell, current scheduler ID, run path, seed, objective,
optimizer, training horizon, frozen source and launcher, runtime export,
checkpoint cadence, host RAM, CPU count, GPU count/type, nice value, account,
requeue setting, output paths and existing explicit PVL exclusions. No live
training allocation is restarted. No primary or continuation ledger is changed.
The controller records exact before/after scheduler states and immutable runtime
fingerprints in `var/artifacts/campaign_pending_acceleration_20260908/`.

Preserve every existing walltime, including 12-hour, 36-hour and 72-hour
requests. The site rejected an in-place walltime extension on pending job
31037827 with `Walltime may not be modified after submission. Please cancel
and resubmit.` The unchanged request was released safely. Accepted test-only
submissions do not authorize or validate in-place scheduler updates. Archive
the original controller, protocol, plan and partial transaction plus the exact
amendment in `walltime_preservation_amendment/` under the audit directory.
Continue the same transaction without creating a new plan or repeating released
amendments. Timeout recovery is handled by the separately audited recovery
workflow. The eight-pass scientific target remains unchanged.

Remove only single `afterany` resource gates whose
predecessor is an authoritative E118/E119/E120 cell with a different run path.
These independent cells can then compete for allocatable resources immediately;
Slurm CPU, host-memory and GPU limits bound simultaneous allocations. Existing
owner-node A100 requests retain node302/mltheory. Unrelated jobs and barriers,
including E121, are outside this amendment.

Preserve every existing partition and account. Root inspection of the site's
`/etc/slurm/job_submit.lua` found that in-place TimeLimit, Account and Partition
changes are all forbidden. Submission logic also normalizes a multipartition
request to the account-derived partition unless the requested partition is
exactly lowprio. The earlier accepted dry runs therefore did not validate the
proposed multipartition eligibility. Archive the second controller/plan amendment
under `partition_preservation_amendment/` and continue the same transaction.

Broaden only candidate nodes compatible with each job's existing partition and
GPU family. Pending E119 Pantry requests on cs add node206 to the A6000 pool
node205/node206/node207. E118 ratio-0.25 cs requests retain that same three-node
A6000 pool; existing lowprio requests may use node205/node206/node207/node208.
Existing ratio-0.40 A5000 lowprio requests may borrow from
node105/node202/node203/node204, preserving their typed A5000 GRES. E120 Graph
seed73 retains its generic single-GPU request and already permitted node302
fallback while adding the compatible A5000 nodes. No vLLM ratio changes.
Owner-account node302 requests retain their owner partition and typed A100.
E120 Qwen-3B Pantry remains on node302.

The achievable amendments remove independent resource gates and expand compatible
node pools. They do not raise partition priority, extend an existing walltime, or
create a new allocation. Preserve queue age and existing requests; separately
recorded failure recovery handles terminal timeouts. Slurm resource availability
still limits concurrent execution.

Plan against authoritative effective mappings. Re-read each mapping, scheduler
state, sole-writer condition and exact submitted command before holding a pending
job. A race to running skips the amendment. Never act on a preexisting hold.
Hold each pending target, update only the enumerated scheduler fields, audit all
preserved fields and frozen launcher SHA-256, and release the same allocation.
Persist each action before the next mutation. If a scheduler rejection leaves
all requested fields unchanged, release the owned hold. A partial unverified
update stops for explicit reconciliation using its saved records. Completed
transactions are idempotent, and retries never create new scheduler jobs.

No result values are inspected for placement decisions. Queue eligibility and
scheduler forecasts do not establish actual training progress; report allocation
and current-attempt optimizer evidence separately.
