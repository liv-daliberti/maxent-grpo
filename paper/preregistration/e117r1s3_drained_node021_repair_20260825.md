# E117-R1-S3 drained-node021 placement repair

Frozen: 2026-08-25 at 01:29 EDT while all 12 E117-R1 jobs remain pending at
zero runtime and before any E117 run directory, optimizer update, or endpoint
exists. PointMaze remains excluded.

## Trigger and non-outcome selection

Slurm's node-health checker changed node021 to `MIXED+DRAIN` with reason
`NHC: check_nv_smi_temp: 10 GPUs are overheated`. The six zero-runtime Qwen
Countdown and Graph C/P/F jobs require node021 and therefore became
unschedulable. This is a physical-node fault, not a scientific or mechanism
result.

Contemporaneous read-only inspection found node103 and node104 healthy in the
same `lowprio`-capable pool. Both expose ten Ampere A6000 GPUs and sufficient
CPU and memory for the registered one-GPU, 8-CPU, 64-GiB jobs. Assign one
complete causal block to each adjacent, identical GPU-class node:

- jobs 30873695--30873697, Qwen Countdown C/P/F: node021 to node103;
- jobs 30873698--30873700, Qwen Graph C/P/F: node021 to node104.

The choice uses only node health, resource compatibility, and whole-block
placement. No E117 efficacy or mechanism telemetry exists or was inspected.

## Transaction

The installer must fail closed unless all six exact jobs are pending with zero
runtime and zero restarts, have no run directory, retain the `lowprio`
partition, `mltheory` account, generic one-GPU/8-CPU/64-GiB/eight-hour
resources, and have byte-exact scientific scheduler exports matching the
release ledger.

Transactionally user-hold the six jobs, verify the same invariants while held,
change only `ReqNodeList`, verify the held records and environment hashes,
write a provenance artifact, then release all six. On failure, restore
node021, release every user hold, and remove only the incomplete S3 artifact.

Preserve all job IDs, audit dependencies, run paths, source snapshot,
treatments, seed, data, optimizer, requests, and training configuration. Merge
the new effective nodes with the existing Python=node101 and MathIR=node203
repairs, and retain the S2/S3 scheduler-amendment chain. The terminal A4 audit
must verify actual completed physical nodes against this effective map.
