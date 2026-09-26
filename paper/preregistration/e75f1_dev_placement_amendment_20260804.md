# E75F1 development-gate placement amendment

**Frozen:** 2026-08-04, after data job 30258124 completed and before
development job 30258125 started or any E75F1 model evaluation occurred.

The completed, byte-equivalent E75R3 development evaluator used 2,228,676 KiB
peak host RSS. E75F1 development job 30258125 requested 64 GiB and one A6000
on node208. Node208 currently has two unallocated A6000s but less than 64 GiB
of unallocated scheduler memory, so Slurm projects a multi-day wait despite
adequate evaluator capacity.

This operations-only amendment changes job 30258125 to request 16 GiB host
memory and permits placement on A6000 nodes 103, 104, or 208. Sixteen GiB is
more than seven times the observed peak. The GPU family remains A6000. Model,
checkpoint, source and execution snapshots, 64 development maps, sampling
seeds, K=8, horizon, thresholds, hashes, zero-update rule, dependencies, and
all downstream scientific identities are unchanged. No evaluation output
existed when this amendment was frozen.
