# E117 Stage 1-S2 CPU audit `cs` route amendment

Frozen: 2026-08-30 after replacement transaction jobs 30980275--30980311
were canceled under user hold and before any Stage 1 allocation, run directory,
response, or optimizer update existed.

All 36 training jobs passed exact held-record validation under the S1 effective
`cs` rule. The final CPU-only audit job 30980311 was also normalized by the
site `job_submit/lua` plugin from requested `all`/`allcs` to effective `cs`.
The launcher still expected effective `all` for that audit record, failed
closed, and canceled all 37 held jobs. Accounting verifies 37 canceled jobs,
zero elapsed runtime, and no assigned node. The provisional ledger was removed.

For the next replacement transaction, require effective partition `cs` for the
CPU-only terminal audit as well as the exact-node training jobs. Retain its
`all` request, `allcs` account, two CPUs, 16 GiB, four-hour limit, no-requeue
policy, complete 36-job `afterany` dependency, frozen audit command, and user
hold. No scientific export, training resource, source, data, endpoint, gate,
causal contrast, or confirmation boundary changes.
