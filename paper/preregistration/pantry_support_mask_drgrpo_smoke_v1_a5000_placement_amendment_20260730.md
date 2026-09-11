# PantryPlan Dr.GRPO smoke A5000 placement amendment

**Status: FROZEN BEFORE JOB 30187473 STARTED — 2026-07-30**

PantryPlan smoke job `30187473` remains pending with runtime `00:00:00`, no
node allocation, and an estimated A100/node302 start on 2026-08-05. Its
scientific identity binds the exact Qwen2.5-0.5B model, source and execution
snapshots, Pantry data, seed 76201, Dr.GRPO arm, 16 rollouts, 32 optimizer
updates, learning rate, masks, and disabled MaxEnt actuators. The identity and
protocol do not require A100 or node302.

This placement-only amendment keeps the same job ID and changes only:

- partition `mltheory` to `all`;
- requested node `node302` to no fixed node; and
- accelerator `a100:1` to `a5000:1`.

CPU count 8, memory 64 GB, four-hour limit, account, command, environment,
source/execution/data/model hashes, seeds, update schedule, evaluation cadence,
and all scientific gates remain unchanged. The job is held before the update,
the post-update scheduler record is verified, requeue remains disabled, and
the same job is released only after all checks pass.
