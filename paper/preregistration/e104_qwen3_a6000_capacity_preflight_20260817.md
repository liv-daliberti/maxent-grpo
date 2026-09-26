# E104 Qwen-3B A6000 capacity preflight

**Frozen before the first held submission and before any post-update E104
evaluation result was inspected on 2026-08-17.**

The registered Qwen2.5-3B E104 cells require node302's A100 queue, whose
current estimated starts are multiple days away.  This preflight asks only
whether the exact immutable E104 runtime fits and performs one finite optimizer
update on a 48GB A6000.  It is not an outcome experiment and cannot replace an
E104 cell by itself.

The preflight uses the Qwen2.5-3B Graph Coloring seed-70 template, the exact
E104 model, source/ops snapshot, objective, optimizer, group size, and decoding
settings, but stops after one training row/update.  Sampled mode-coverage
evaluation is disabled.  The scheduler may select an A6000 from nodes 103,
104, 205--208, or 805 in partition `lowprio` under account `mltheory`.

It passes only if the job terminates normally after update 1, all recorded
numbers are finite, v6 group centering is active, legacy semantic and RMS
controller paths are off, the centered mean and magnitude bounds hold, replay
is applied, and neither log contains a traceback, assertion, CUDA error,
out-of-memory marker, or non-finite failure.  No evaluation metric is parsed or
read by the auditor.

If it passes, the still-pending five Qwen2.5-3B E104 mechanism cells may receive
a separately recorded execution amendment to the same A6000 node pool.  Their
scientific configuration, run directories, step-64 criterion, and immutable
snapshot remain unchanged.  E105 retains its registered paired placements and
is not amended by this capacity test.

## Held-submission normalization record

The first held submission, job 30638151, requested the partition name `all`.
Slurm interpreted that value as meta-selection rather than the literal
partition and normalized the held record to `mltheory`, which is incompatible
with most nodes in the requested A6000 list.  The launcher's held audit rejected
the record, cancelled the job before allocation, and removed its ledger.  No
code or model ran.  The successful retry must use the concrete authorized
partition `lowprio`; this changes only scheduler routing.
