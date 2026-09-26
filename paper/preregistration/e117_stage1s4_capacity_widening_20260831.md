# E117 Stage 1-S4 capacity widening

Frozen: 2026-08-31T22:04:10-04:00 after seven Stage-1 cells completed,
before any allocation or run directory for the target cells below, and in
response to the user's explicit request to maximize safe near-term execution.

## Operational diagnosis

The registered A5000 and amended A100 pools are fully allocated.  Pending
Stage-1 work remains serialized across context blocks even though three
64-GiB-compatible accelerator slots are presently available before the
2026-09-01 06:00 maintenance reservation: one A6000 on node207 and two L40s
on node403.  Shared storage has 541 GiB free.

## Prospective placement amendment

Make only the first registered arm in each of these previously unseen
context/seed blocks dependency-free and place the complete block on one
physical node:

- graph_coloring seed 203: jobs 30980490--30980492, F-C-P, node207 A6000;
- python_factors seed 201: jobs 30980493--30980495, C-P-F, node403 L40;
- python_factors seed 202: jobs 30980496--30980498, P-F-C, node403 L40.

Retain `afterok` dependencies inside each block, all scientific exports,
source/data/request identities, one GPU, eight CPUs, 64 GiB RAM, restart and
checkpoint policy, and the terminal audit's complete `afterany` dependency.
Use a 07:30:00 wall-time bound so jobs can backfill before maintenance;
automatic resume remains enabled if the bound is reached.

This changes placement, GPU class, cross-block concurrency, and wall-time
only. It changes no arm, seed, training or evaluation data, optimizer-update
target, endpoint, analysis threshold, confirmation boundary, or scientific
export. The terminal execution audit must disclose and accept this amendment
before interpreting outcomes.
