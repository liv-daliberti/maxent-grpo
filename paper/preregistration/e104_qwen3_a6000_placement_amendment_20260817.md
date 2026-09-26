# E104 Qwen-3B A6000 placement amendment

**Conditionally registered before the capacity preflight completed, before any
Qwen2.5-3B E104 job started, and before any post-update E104 evaluation result
was inspected on 2026-08-17.**

If and only if the separately registered one-update Qwen2.5-3B A6000 capacity
preflight completes normally and passes its outcome-blind mechanism audit, the
five still-pending Qwen2.5-3B E104 jobs 30637795--30637799 may be changed from
partition `mltheory`, node `node302`, and one A100 to partition `lowprio`, nodes
`node[103-104,205-208,805]`, and one A6000.  The account remains `mltheory` and
the already registered two-hour backfill limit remains unchanged.

The amendment must abort without changing a job if the preflight gate is not
complete and passing, if its auditor inspected outcome metrics, or if any of
the five target jobs has started, left the pending state, or differs from its
registered scientific configuration.  After applying the scheduler-only
change, every target must still be pending with zero runtime and must have the
exact amended account, partition, node list, GPU type, and time limit.  A
partial or invalid change is rolled back to the original A100 placement.

This changes no model, immutable source/ops snapshot, data, domain, prompt,
seed, optimizer, decoding setting, group size, objective, coefficient,
telemetry criterion, checkpoint cadence, evaluation cadence, or 64-update
stopping rule.  It does not authorize any E105 placement change.  No PointMaze
run is included.
