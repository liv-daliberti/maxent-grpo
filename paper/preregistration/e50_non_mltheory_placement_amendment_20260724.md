# E50 non-MLTheory placement amendment

**Status: FROZEN BEFORE REPLACEMENT — 2026-07-24 10:48 EDT**

After E50 launch, seven of twelve jobs allocated on the frozen node302 A100
placement and five Countdown jobs remained pending without starting or
producing run data. The user directed that those pending jobs be launched on
available non-MLTheory compute.

This amendment changes only the scheduler placement of the following pristine
pending jobs:

| Job | Arm | Seed |
|---|---|---:|
| `30073581` | `online_canonical_haarnoja` | 43 |
| `30073582` | `grpo` | 44 |
| `30073583` | `online_canonical_haarnoja` | 44 |
| `30073584` | `grpo` | 45 |
| `30073585` | `online_canonical_haarnoja` | 45 |

Their E50 run stamps, source and execution snapshots, model, data, objective,
seeds, evaluation, checkpointing, 50-pass budget, and uncapped-alpha contract
remain byte-identical. The amended placement is:

- partition: `all`;
- account: `allcs`;
- eligible nodes: `node020,node023`;
- accelerator: one `rtx_3090` GPU per job;
- CPUs, memory, and wall time: unchanged at 8, 64 GiB, and seven days.

Nodes 020 and 023 were both Slurm `IDLE` with ten configured RTX 3090 GPUs
each immediately before this amendment. RTX 3090 under `allcs` is an
established repository placement for Qwen2.5-0.5B experiments. The jobs are
held during the scheduler-field update, audited in their amended pending
state, and released together. No already-running E50 job is moved.
