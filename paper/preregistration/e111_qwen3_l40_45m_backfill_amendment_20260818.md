# E111 Qwen-3B L40 45-minute backfill amendment

Date frozen: 2026-08-18, before changing the five job time limits and without
inspecting any E111 endpoint evaluation result.

After the prospective L40 placement amendment, Slurm reports node403 as
available but planned for other work and assigns the two-hour E111 requests a
2026-08-19 start. Completed matched Qwen-3B 64-step mechanism jobs took
approximately 27–34 minutes. The separately frozen runtime-ops durability
amendment now saves exact optimizer/RNG state every eight steps, so even a
short allocation that does not finish is useful and recoverable.

For exact pending jobs `30674758`, `30674759`, `30674760`, `30674761`, and
`30674762`, change only `TimeLimit` from `02:00:00` to `00:45:00`. Node403,
L40 GRES, partition, account, CPU/memory/GPU requests, job IDs, environments,
run directories, data, models, seeds, optimizer, objectives, and target steps
remain unchanged. No job is canceled, requeued, or reset. E112 retains its
paired A6000 placement and full preregistered limits. PointMaze remains
excluded, and no endpoint outcome motivates or gates this change.

The final E111 auditor must validate exact before/after scheduler records,
unchanged submit lines, the E111 ledger and amendment digest, and the sole
`TimeLimit` transition above.
