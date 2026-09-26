# E119 targeted memory recovery to 96 GiB — September 5, 2026

The user explicitly authorized cancelling/restarting the twelve E119 allocations
with independently confirmed host-memory throttling, adding more memory. The
operational implementation uses the same Slurm job IDs: requeue into our own
hold, increase MinMemoryNode from 64 to 96 GiB, audit, then release. This avoids
new scientific cells or continuation-ledger changes.

Exact targets: 31045872, 31045873, 31045874, 31037849, 31048154, 31048159,
31048160, 31048163, 31048156, 31048157, 31048158, 31048164. All are Countdown
cells. No other E119 cell, E118/E120 job, deliberate hold, or placement is changed.

At 23:28 UTC, every target still had anonymous plus shared memory above the
64-GiB memory.high threshold, zero evictable file cache, and increasing throttle
events. The largest observed working set was 70.86 GiB. The 96-GiB request adds
about 25 GiB of headroom over that sample; stability must be checked across
actual evaluation and save cycles. Fewer simultaneous allocations can fit:
at the capacity check, at most eight of the twelve could fit existing routes
without competing admissions. Remaining cells retain their ordinary queue routes.

Before stopping each learner, preserve logs/metrics and validate the latest
model/optimizer archive structure, three saved progress counters, and replay-bank
state. Revalidate after the writer stops. All twelve initially had a usable
checkpoint; no restart from initialization is authorized or needed. The initial
independent review found 353 logged updates to repeat across six cells, with a
maximum of 95 per cell; execution allows at most 96 and reselects newer valid
saves. Some saves precede evaluation logging by one step. No partial candidate
is discarded; a newly partial writer requires review before proceeding.

Preserve the full submitted arguments/exports, models, seeds, objectives,
datasets, optimizer settings, evaluation law, terminal budgets, CPU/GPU requests,
accounts, partitions, QOS, nice values, and node constraints. Only the memory
request and operational requeue counters change. Main and continuation ledgers
remain byte-identical. Release only holds created and audited by this transaction.

Job 31045873 is already at restart count 12. The actual frozen watchdog checks
that budget only when deciding whether to retry a failed process, so the manual
memory recovery can start at 13/12; it will not automatically requeue a further
watchdog failure. No runtime or retry-policy patch is included in this memory
transaction.

Plan, independent reviews, archived diagnostics, checkpoint certificates,
transaction receipts, and startup checks are recorded under
`var/artifacts/e119_memory96_recovery_20260905/`.

Execution encountered a Slurm cleanup delay beyond the controller's initial
120-second limit after three successful held-state resizes. The fourth job
finished cleanup into its transaction-owned hold shortly afterward. The
continuation reconciles those exact receipts, preserves the original plan and
script hashes, and stops the remaining selected learners sequentially while
allowing independent node cleanup to overlap. It waits up to 600 seconds for
cleanup, validates each stopped writer before its update, and releases only
after all twelve audited requests are 96 GiB. No completed mutation is repeated.
