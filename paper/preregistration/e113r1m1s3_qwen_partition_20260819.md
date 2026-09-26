# E113-R1-M1-S3: Qwen smoke partition amendment

**Frozen:** 2026-08-19 16:27 EDT, after the S2 wall-time amendment and while
M1 job 30790590 remained pending at zero updates with no run directory.

The same A6000 nodes are exposed through both the preemptible `lowprio`
partition (priority tier 1) and the normal `all` partition (priority tier 100).
After S2 made the request backfillable, the job remained pending for priority,
not resource incompatibility.

Move only pending job 30790590 from `lowprio` to `all`. Keep its job ID,
`mltheory` account, A6000 node set and GRES, two-hour ceiling, 64 GB host-memory
request, one-GPU topology, output path, batch, optimizer, offload settings,
model, domain, seed, data, objective, and all DAPO limits unchanged. The update
is valid only before any output path exists. Capture complete scheduler records
before and after in an atomic amendment ledger.

This scheduling change carries no efficacy information and does not create a
new trajectory or science cell.
