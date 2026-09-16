# E119 public A6000/A100 pool amendment — September 9, 2026

This prospective operational amendment broadens the ten newly queued Pantry
continuations from public A6000 nodes205/207 to public nodes205/207/302, with
one generic GPU request. Node302 supplies A100 GPUs and already runs an E119
Pantry continuation using the same compatible training profile. The change
preserves each scheduler job ID, full scientific environment, seed, method,
source and data, run directory, checkpoint, physical batch settings, host memory,
CPU request, walltime and original immutable submission text. It introduces no
new scientific cells or outcome-dependent selection.

The nine 116 GiB, 36-hour requests and one 128 GiB, 72-hour request retain their
memory and walltime. The existing lowprio/mltheory account and partition remain.
The A6000 nodes remain eligible; node302 adds an opportunity when memory becomes
available, including after current workloads complete. Individual scheduler
forecasts are provisional and do not imply that ten jobs can start together.
Drained nodes206/208, incompatible smaller GPUs and excluded private/PVL nodes
remain outside the pool.

A never-released, owned diagnostic hold verified that Slurm accepts an in-place
change from gpu:a6000:1 to gpu:1 plus the expanded node list. That diagnostic was
cancelled without execution. Each scientific job must still be pending before
an owned hold is applied. A job that allocates during the race is preserved;
only a confirmed hold owned by this operation may be cleared. Before release,
audit the exact GPU count and normalized node list, unchanged remaining resource
fields and exports, complete unchanged checkpoint, held predecessor, sole-writer
condition and authoritative continuation identity. Persist each intent before
its scheduler mutation, and reconcile ambiguous acknowledgements without
repeating them blindly.

Under the existing continuation-ledger lock, record only the amended actual
scheduler profile and provenance. Preserve all 100 scientific cells, all 75
continuation rows, current job IDs and predecessor history. This prospective
record supplements the earlier A6000 placement amendment; the controller and
machine-readable receipts live in
`var/artifacts/e119_generic_pool_amendment_20260909/`.
