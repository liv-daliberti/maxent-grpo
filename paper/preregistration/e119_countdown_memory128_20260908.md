# E119 Countdown memory-throttling recovery — September 8, 2026

The user authorized fixing broken workloads and improving completion speed. This
operational amendment selects jobs by independently measured host-memory pressure,
optimizer progress and checkpoint integrity; scientific outcomes do not select jobs.

Jobs 31048154 (Dr.GRPO seed 45) and 31048161 (Dr.GRPO seed 47) have respectively
100.48 and 101.24 GiB of non-cache working memory against 96 GiB memory.high,
61.75 and 23.45 million high events, and processes blocked in
mem_cgroup_handle_over_high. They have no OOM kills. ReplayMaxRL seed 45 job
31048157 is also selected using Slurm batch AveRSS 107130896 KiB (102.17 GiB),
MaxRSS 109252660 KiB (104.19 GiB), 1179-second median weight synchronization
and the directly confirmed matching failures in the other two Countdown cells.
Its bounded cgroup probes could not start or complete under the affected allocation.
The third diagnosis is an explicitly recorded inference from scheduler RSS above
its 96 GiB reservation and severe synchronization delay, not a direct cgroup
measurement. The root coordinator approved this evidence-based extension.

For these three targets, requeue the existing job into a transaction-owned hold,
wait for its writer to stop, validate the latest complete model/optimizer archive
and the three saved step counters, increase host memory from 96 to 128 GiB, retain the original
36-hour walltime, audit scheduler readback, then release the hold.
Keep the same job ID, cell, seed, run directory, full submitted scientific command,
frozen launcher/runtime, model, optimizer, data, evaluation, endpoint, CPU/GPU
resources, walltime, placement, account, partition, QOS, exclusions, and retry policy. The
normal same-ID requeue increases Restarts by one. Training must auto-resume from
a valid checkpoint, never initialize afresh. Both experiment ledgers remain
unchanged by this controller and their hashes are checked before release.

At the initial audit job 31048154 had validated step 2496 and logged step 2679
(183 repeated updates). Job 31048161 had validated step 2304 and a throttled,
incomplete step-2496 optimizer archive; its last logged step was 2495. A valid
older save permits quiescing this documented stalled writer with at most 192
completed updates repeated. Preserve the partial save; revalidate after cleanup
and use step 2496 if it finishes, otherwise retain the valid step 2304. Do not
delete or modify checkpoint files. Other newer valid saves supersede these initial
baselines. Archive logs, metric streams, scheduler records, checkpoint metadata,
source hashes, and restart counts before and after cleanup.

The controller records intent before scheduler changes. It never repeats an
ambiguous requeue; a held job with exactly one additional restart can reconcile
that intent. It releases only a verified transaction-owned hold. A cleanup wait
is bounded at 300 seconds with periodic progress messages and can be continued.
Larger memory may reduce simultaneous admissions, but observed pressure makes
smaller requests unsuitable. Startup must subsequently confirm restored optimizer
progress and absence of new memory throttling.

## Pre-execution walltime policy amendment

Before any target was stopped, the coordinator verified that the site submission
plugin rejects walltime changes after submission, including pending jobs, with:
`ERROR: Walltime may not be modified after submission. Please cancel and resubmit.`
Accordingly this transaction increases memory only and preserves the existing
36-hour limit. Same-ID requeue resets the allocation clock; removing measured
memory throttling is the direct throughput repair. Do not send a TimeLimit update.
The original controller, protocol and prepared plan are retained under
`var/artifacts/e119_countdown_memory128_20260908/walltime_policy_amendment/`,
with a hash-linked amendment receipt. No scheduler actions preceded this revision.
