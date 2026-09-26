# E65R1 compatible accelerator placement

Frozen 2026-07-27 13:29:40 EDT, before this placement mutation.

After the registered RTX 2080 compatibility rollback, the five affected job
identities were pending on the original A5000 route with zero current runtime,
one recorded infrastructure requeue, and still zero optimizer metrics.
Read-only node inspection and `srun --test-only` checks found:

- A6000 node103 had four unallocated GPUs and enough Slurm CPU/memory capacity
  for two 8-CPU, 64-GB jobs; a matching request was eligible at 15:51 EDT.
  E61-R1 Graph/MathIR jobs already run on this device family.
- A100 node302 had three unallocated GPUs and enough Slurm CPU/memory capacity
  for three such jobs; a matching request was eligible at 15:51 EDT.
- Both Ampere device families support the frozen bfloat16 execution path.

The following pending jobs may be user-held and amended in place:

- Graph seeds 44/45, jobs 30127479/30127481:
  `nodelist=node103`, `gres=gpu:a6000:1`.
- MathIR seeds 43/44/45, jobs 30127489/30127490/30127491:
  `nodelist=node302`, `gres=gpu:a100:1`.

All remain `account=mltheory`, `partition=lowprio`, 8 CPUs, 64 GB, and a
24-hour limit. Job identity, source and operations snapshots, protocol
identity, dtype, model, data, seed, optimizer, controller/actuator, evaluation,
and checkpoint settings must remain unchanged. Every resolved held record is
audited before release. Accelerator placement is reported per seed and cannot
select a checkpoint or result.
