# E113-R3-S1: Qwen science partition amendment

**Frozen:** 2026-08-19 16:31 EDT, after M1's scheduler-only S3 amendment
started the Qwen smoke on an A6000, but before its first accepted optimizer
update or any M1 training outcome. No E113-R3 job had been submitted.

The M1 queue diagnosis established that the same registered A6000 node set is
available through `lowprio` at scheduler tier 1 and `all` at tier 100. Moving
the zero-update smoke to `all` resolved the scheduling blockage without
changing its compute device or experimental environment.

For all 25 unsubmitted E113-R3 Qwen cells, use partition `all` with account
`mltheory`, the frozen A6000 node set `node[103-104,205-208,805]`, one
`gpu:a6000`, and the original 64 GB host-memory request. This supersedes only
the launcher's Qwen partition field. Falcon placements and all scientific
fields—including model, data, seeds, complete 16-response batches, optimizer,
offloads, objective, query cap, checkpoints, and horizon—remain exactly as
frozen in E113-R3.

This is a prospective scheduling amendment with no M1 efficacy observation and
no E113-R3 outcome. The R3 ledger must bind this document and audit every held
Qwen job's expanded `Partition=all` record before the atomic release.
