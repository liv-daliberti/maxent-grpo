# E118 Pantry timeout successor — September 10, 2026

The user requested completion of existing E118/E119/E120 workloads and recovery
of failures. Three E118 Qwen-3B Pantry cells, MaxRL seed70 (31048143), ReplayMaxRL
seed70 (31048144), and MaxRL seed71 (31048145), may need another 12-hour allocation.
Their current timeout supervisor31159945 expires at2026-09-11T02:18:48.621432Z.
Pending ReplayMaxRL seed71 (31048146) has a separate guard and is outside this scope.

This successor never changes an existing guard, its deadline, allocations,
submissions, scientific exports, checkpoints, evaluations, or ledgers. Preparation,
the default `once` inspection, and CPU registration do not mutate Slurm.

The new CPU watcher cannot retry or release science jobs before the predecessor's
original absolute deadline. At takeover it must find predecessor CPU31159945
inactive, acquire the predecessor singleton lock, and retain that lock as well as
its own singleton. The existing continuation-ledger lock covers takeover and all
science decisions. The predecessor's final plan binding, deadline, and transaction
are checked; any unresolved mutation or stopped-error cell blocks takeover.
Completed cells suppress further retries. Its final transaction hash cannot change
after takeover. Thus two guards cannot perform retries for the same cell together.

Each cell receives at most two additional same-ID retries, only after inactive,
accounting-confirmed TIMEOUT. Each requires a complete model/optimizer checkpoint
with matching model/global/prompt and optimizer counters and saved replay bank and strictly more updates than both its
prepared checkpoint floor and the predecessor's final resume floor. Every further
retry must advance beyond the previous successor resume floor. A newer incomplete
checkpoint, changed identity/source/resources, other writer, or an ambiguous
requeue is retained for manual review. No new GPU job or fallback is created.
Any scheduler-command exception leaves durable intent and stops that cell visibly;
uncertain acknowledgements are not automatically retried. Purged controller records produce a visible per-cell manual stop; the watcher
does not guess a replacement. Non-timeout failures also stop visibly.

The existing reviewed requeuehold/release transaction primitives persist mutation
intent and require exactly one restart increment before releasing an owned hold.
The successor additionally checks its absolute window immediately before every
science requeuehold and release. The new deadline is exactly seven days after the
predecessor's original deadline; restarts never extend it. Expiry retains unresolved
owned holds and permits no further science release. Retry counts survive CPU renewal.

The prepared CPU command requests two CPU threads,2GiB RAM,no GPUs,a25h10min allocation,
and only node915/node917 on lowprio/mltheory. Deployment submits that exact command
held, then registers its identity and verifies actual resources before releasing
the CPU job. The watcher verifies registration on every process start and renews
only that CPU job after23hours, at most eight times. Polling is60seconds; scheduler
commands have45-second process timeouts. A fresh status/ready receipt distinguishes
waiting before takeover, active monitoring, expiration, and per-cell manual stops.

Source, core helpers, CPU entrypoint, and frozen science runtime are fingerprinted.
The mutable presentation module `campaign_stats.py` is not used for mapping or
fingerprinted: this successor resolves and cross-checks the authoritative150-cell
aggregate and Qwen-3B source ledgers directly. No outcome selects a retry.
