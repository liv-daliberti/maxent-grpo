# E122 completion-transition controller recovery

The existing release watcher deliberately stopped at 13:28:14 UTC on September
11 when squeue still reported allocation 31158680 while sacct already reported
its completion. No release was issued from that unknown snapshot. The same job
now has positive COMPLETED/0:0 accounting and its original terminal receipt and
export; the four other completed E122 allocations have the same evidence.

The user requests continued E122 progress as E118 capacity becomes available.
This additive supervisor retains the original 100-cell plan, held ledger,
scientific source, treatment/seed order, resource requests, and once-only release
intent/result journals. It calls the original controller's advance_once rather
than replacing its admission or release implementation. Its persistent cap is
still four unfinished reservations. A separately pinned storage adapter accounts
for every released pending/running training writer and explicitly identified
inference array task, including full E122 terminal and six-slot peak reserves.
The exact already released E124 systems allocation31161634 is separately pinned
to its source, worker, plan and submit identity, with its full original220GiB
peak allowance and no checkpoint credit. Unrecognized helpers remain blocked.

Each release decision runs under the shared storage admission lock and the E124
and E123 per-admission transaction locks, then the original E122 journal lock.
Busy locks skip this cycle without modifying another controller. Original
unknown/ambiguous/failed-terminal states never permit a release. A clean unknown
scheduler snapshot is reobserved up to five times, allowing a completion race
to settle; an ambiguous existing release intent stops advancement immediately.
No uncertain scheduler release command is repeated.

The new supervisor is one explicitly registered, held-then-released CPU job on
node917 (2 CPUs, 8 GiB). Its immutable plan pins the wrapper, tests, protocol,
unchanged original sources, compatibility auditor, storage adapter and its
evidence. It has a 21-day deadline and may requeue only its own CPU allocation
after 36 hours to remain within its three-day walltime. The original watch log
and stopped status remain intact. Initial advance calls may be made from the
same authenticated preparation process while this exact CPU remains held,
using the same admission locks and journals; the CPU is then activated for
continued refill. This recovery itself performs no GPU resource, checkpoint,
model, data, or scientific-result edits.
