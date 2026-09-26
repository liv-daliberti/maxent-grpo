# Pantry support-mask Dr.GRPO smoke v1 r2 query-budget repair

**Status: FROZEN AFTER THE TERMINAL R1 AUDIT AND BEFORE R2 CONFIGURATION OR EXECUTION — 2026-07-30**

## Antecedent

R1 training job `30199460` completed cleanly and proved that the generic
six-position learner sampler and exact finite-policy-tree sensor execute on
PantryPlan. It emitted four optimizer updates, six active decisions per
rollout, 64 policy leaves, 63 exact-entropy prefix rows, finite policy loss and
gradient norm, Pantry task identity, positive verified rewards, and sampled
multi-mode evaluation prompts. Independent audit job `30199464` nevertheless
failed, correctly, because only four rather than 32 updates were present.

The cause is a launch-budget unit mismatch. `max_queries` counts sampled
responses, not prompts. With 16 responses per prompt, the established
canonical launcher convention for exactly 32 optimizer updates is
`32 * 16 - 16 = 496`; r1 incorrectly set it to 32. The audit also requested
`train/pg_loss_inf` and `train/pg_loss_nan`, fields the trainer does not emit.
The metrics stream instead contains a directly finite `train/pg_loss` plus
the emitted nonfinite counters `train/zero_pg_loss_count_inf` and
`train/zero_pg_loss_count_nan`.

R1 remains failed. Its receipt, identity, submission, stdout, manifest, and
metrics are immutable antecedents and cannot be overwritten or reinterpreted.

## Authorized repair

R2 makes exactly two plumbing changes:

1. set `OAT_ZERO_MAX_QUERIES=496`, while retaining 32 train rows, one prompt
   epoch, 16 responses per prompt, and target optimizer updates 32; and
2. have the independent audit require finite `train/pg_loss`, exact-zero
   emitted `train/zero_pg_loss_count_inf` and
   `train/zero_pg_loss_count_nan`, and the unchanged exact-zero policy-gradient
   norm nonfinite counters.

All model, data, seed `76201`, prompt, six-bit support, reward, optimizer,
learning rate, batch sizes, evaluation, thresholds, and disabled MaxEnt
actuators remain identical. The previously approved placement-only amendment
remains `partition=all`, one A5000, no fixed node, with requeue and recovery
disabled. R2 uses fresh run, identity, submission, manifest, audit, log, and
checkpoint namespaces.

## Gate

The unchanged independent audit must observe exactly the initial row, updates
1 through 32, and the terminal evaluation row; exact learning rounds 1 through
32; finite six-step sampler, entropy, policy, reward, and gradient telemetry at
every update; at least one positive verified-reward update; sampled
multi-mode evaluation; exact source/execution/data/model/scheduler identities;
and a clean Slurm exit. Failure stops PantryPlan before a paired MaxEnt smoke.
