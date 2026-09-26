# E119 scheduler-placement amendment — 2026-09-01

This operational amendment was made after release and before any affected job started.
It changes scheduler placement only. The frozen dataset, source snapshot, model, seeds,
arms, optimizer, update horizon, evaluation schedule, and all treatment variables remain
unchanged.

- Original placement: `mltheory` / `mltheory`, `node105`, `gpu:a5000:1`.
- Jobs already running and left unchanged: `31014398`–`31014404` (7 jobs).
- Jobs pending and updated: `31014405`–`31014497` (93 jobs).
- Updated request: `all` / `allcs`, nodes
  `node203,node204,node205,node207`, `gpu:1`.
- Effective eligible hardware: A5000 on node203–204 and A6000 on node205/207.
- Excluded at amendment time: drained node202/node206 and lower-memory GPU nodes.

Rationale: E119 was unnecessarily restricted to one A5000 host, leaving 92 jobs pending
with `ReqNodeNotAvail`. The amended routing follows the E117/E118 convention of using
the `all` request with the `allcs` account and a high-memory node allowlist. It broadens
scheduling capacity without changing the learning algorithm or data.

Immediate post-update audit: all 93 affected jobs were still pending, requested exactly
the four-node allowlist with generic `gpu:1`, and reported `Priority`; all 7 unaffected
jobs were running on node105. No running job was modified.

## Balanced concurrency expansion — 2026-09-02

A second scheduler-only amendment moved 36 still-pending jobs to the higher-priority
`mltheory` route with `ReqNodeList=node105,node302` and generic `gpu:1`. The moved
cells are seeds 43–45 for MathIR, PantryPlan, and Python Factors: nine complete
domain/seed quartets, exactly nine jobs per algorithm and twelve jobs per domain.

This selection starts previously untouched domains sooner while preventing hardware
placement from being confounded with treatment. The remaining pending jobs retain the
`allcs` high-memory pool. No running or completed job was modified, and no scientific
configuration changed.

Immediate post-update audit: four MathIR seed-43 jobs (one per algorithm) started on
four available node302 A100 GPUs, increasing E119 concurrency from 19 to 23. The other
32 moved jobs remained eligible on the `mltheory` pool under ordinary priority/resource
scheduling.

## Measured memory right-sizing — 2026-09-02

Before this amendment, nine completed E119 jobs had measured batch peak RSS between
27.5GB and 29.9GB while reserving 64GB. Pending E119 jobs were therefore changed to a
40GiB host-memory request, preserving at least 10GiB headroom over the observed peak.

Pending E118 jobs were not changed: completed Falcon-1B runs measured 62.5–64GB peak
RSS, and Qwen-3B lacked sufficient completed evidence for a safe reduction. No running
job and no scientific setting was modified. The E119 change increased concurrency from
24 to 31 immediately.

## Additional safe-node allowlist — 2026-09-02

Pending E119 jobs on the `allcs` route were expanded from node203–207 to include
node104 (A6000) and node403 (L40). Both hardware classes were already proven by active
E118 jobs and provide sufficient device memory for the Qwen-0.5B runtime.

No job started immediately because node104 capacity was scheduler-planned for other
higher-priority work and node403 filled during the audit. The expanded allowlist remains
available for future backfill. No running job, model setting, data, or treatment changed.
