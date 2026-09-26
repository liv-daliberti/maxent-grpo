# E118/E119 owner priority with A6000 borrowing — September 8, 2026

The user requested finishing existing E118/E119/E120 workloads as quickly as
possible and requeuing failed allocations. Following the separate pending-job
amendment, this scheduler-only extension gives pending E118/E119 owner-account
requests access to compatible borrowing capacity while retaining their original
node302 owner priority. No new scientific cell or scheduler job is created.

The eligible cells are existing pending E118 Qwen2.5-3B Countdown/Pantry and E119
Qwen2.5-0.5B Pantry continuations with account mltheory, partition mltheory,
node302, one typed A100, 128 GiB host RAM, 72-hour walltime, no remaining
scheduler dependency, and unchanged vLLM ratio 0.25. E118 uses 16 CPUs; E119 uses
eight. Preparation occurs after the preceding amendment has removed independent
resource gates. Record the exact selected job IDs in the plan before mutation.

Change only Partition to `mltheory,lowprio`, the candidate node pool to
`node205,node206,node207,node208,node302`, and the requested GRES from
`gpu:a100:1` to `gpu:1`. The same allocation can then use an owner A100 on node302
or borrow a 48-GB A6000 on nodes205–208. Current E118 Pantry jobs
31048143/31048144/31048145 are advancing on A6000 with the same frozen runtime
and ratio; E119 Pantry also has a healthy A6000 allocation, 31048182. Countdown
has an existing validated A6000 checkpoint continuation. Both GPU families have
at least 48 GB; no 24-GB A5000 node is eligible. Preserve all CPU and host-memory
requests. Slurm remains responsible for enforcing aggregate resource capacity.

Preserve account, job ID, run path, seed, horizon, objective, optimizer, data,
model, frozen launcher/source, every exported runtime variable including the
vLLM ratio and automatic resume, output paths, nice value, requeue setting,
checkpoint/evaluation cadence, and the complete existing PVL exclusion. Primary
and continuation ledgers remain byte-for-byte untouched by this controller.
E120 Qwen-3B Pantry and other E120 jobs retain their current placement because
this extension does not establish their A6000 runtime compatibility.

Use a separate plan, exact scheduler records, hashes and transaction receipt in
`var/artifacts/campaign_owner_borrowing_20260908/`. Validate effective mapping,
sole-writer identity, pending status and unchanged launcher before holding each
job. Never alter a preexisting hold or a running allocation. Audit the generic
single-GPU request, absence of a remaining typed-A100 TRES constraint, all
preserved resources, runtime exports and candidate nodes before releasing the
same job. Persist mutations and release receipts. If Slurm rejects the update
without changing requested fields, release only the hold created by this
controller. A partially accepted or unverified update stops for reconciliation.
No account update, cancellation or resubmission is attempted. Completed apply
transactions are idempotent.

Submission dry runs verify scheduler acceptance only. Allocation and current
optimizer progress require separate observations.

## Status: unapplied; site policy does not support this amendment

Root inspection of `/etc/slurm/job_submit.lua` established that Partition cannot
be changed after submission. The submission hook also normalizes multipartition
requests to one account-derived partition unless exactly lowprio was requested.
No plan was prepared and no scheduler mutation occurred for this owner-borrowing
proposal. The controller now refuses execution. Existing owner-node requests
retain node302 and gain only the separate removal of independent resource gates;
no replacement is created merely to change this placement.
