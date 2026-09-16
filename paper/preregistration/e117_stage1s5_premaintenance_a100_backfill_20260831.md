# E117 Stage 1-S5 pre-maintenance A100 backfill

Frozen: 2026-08-31T22:08:53-04:00 before allocation, run-directory
creation, or optimizer updates for the target cells, in response to the
user's request to maximize Stage-1 progress before the 2026-09-01 06:00
cluster maintenance reservation.

Three running Countdown cells are expected to release three node302 A100s
around 00:40--01:00. Make the first registered arm in each previously unseen
block dependency-free and place each complete block on node302:

- python_factors seed 203: jobs 30980499--30980501, F-C-P;
- Falcon MathIR seed 201: jobs 30980502--30980504, C-P-F;
- Falcon MathIR seed 202: jobs 30980505--30980507, P-F-C.

Retain `afterok` ordering within each block, every scientific export and
endpoint identity, one GPU, eight CPUs, 64 GiB memory, checkpoints, automatic
resume, and the terminal audit dependency. Use a five-hour wall-time bound so
the first cells can backfill before maintenance; a timeout remains resumable.
This changes placement, GPU class, cross-block concurrency, and wall time
only and must be disclosed by the terminal execution audit.
