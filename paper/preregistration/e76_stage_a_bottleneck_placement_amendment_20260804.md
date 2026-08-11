# E76 Stage-A bottleneck placement amendment (2026-08-04)

This scheduling-only amendment is recorded before either replacement cell takes optimizer step 1. The Falcon/Pantry replacements were clean failures at argument parsing and retained no optimizer metrics.

The CS/PVL A6000 pools are saturated by a higher-priority 512-task array, node208 is hardware-drained for GPU temperature, and Slurm projects the Falcon/Pantry queue no earlier than 2026-08-05. Meanwhile node302 has two spare A100 GPUs but is host-memory reservation constrained.

To advance the E76A critical path, the two Falcon/Pantry GRPO cells at learning rate 5e-8 (KL beta 0 and 0.01) move from A6000/64 GiB to A100/64 GiB. They were initially admitted with an 18-hour backfill limit based on the live Falcon/graph rate. Hardware placement is treated as systems-only; data, seeds, optimizer, hyperparameters, prompts, validation, stopping, selectors, and gates remain frozen.

Qwen graph job 30244364 is requeued before its first 320-step checkpoint after approximately 13 minutes, then made dependent on both amended Falcon cells so it restarts automatically when they exit. The other running Qwen cells are untouched. The E76 stage controller continues to depend on all 48 current cell job IDs.

## Activation note

During activation, Slurm briefly started Qwen job 30244365 while the amended jobs still inherited an incompatible CS QoS/CPU-task constraint. Job 30244365 was requeued after one optimizer step and before its first checkpoint. Those constraints were corrected before either Falcon cell took optimizer step 1. To prevent another race, every pending Qwen E76A sibling now depends on both Falcon jobs; jobs 30244362 and 30244363 continue uninterrupted. Both Falcon jobs then entered RUNNING on node302.

The observed Pantry sequence length made the 18-hour bound insufficient, and Slurm denied extending running jobs. Before either job reached its first 320-step checkpoint, jobs 30252796 and 30252797 were canceled and superseded by held, otherwise identical jobs 30253165 and 30253166 with 48-hour limits. Qwen dependencies and the 48-cell controller were retargeted before cancellation; the partial pre-checkpoint executions are excluded from selection.

## Serialization release

At 2026-08-04 12:11 EDT, amended Falcon job 30253166 had completed cleanly
and job 30253165 was already running on node302. The temporary dependency on
30253165 had therefore finished its sole purpose of admitting the Falcon pair
before their Qwen siblings. It was removed from the 22 still-pending Qwen
Stage-A jobs so the scheduler can backfill a Qwen cell whenever an incumbent
Qwen job releases enough node302 memory. Running jobs were not interrupted.

Four stale E74 user/requeue holds (30241813, 30241814, 30241815, and 30241848)
were also released so the frozen-recipe transfer work can finish normally.
These actions change only scheduler eligibility. Model, data, seed, arm,
learning rate, KL beta, evaluation, stopping, selection, and test gates remain
unchanged, and Slurm resource accounting continues to protect the running
Falcon job.
