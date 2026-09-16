# E119 Pantry recovery and node105 capacity — 2026-09-05

Following the user's authorization to fix E119 and check further useful
capacity, the live audit observed Pantry ReplayDr.GRPO seed 43 job 31041178
fail after its twelfth restart. Its old allocation's two-hour watchdog fired
while sampled evaluation at step 480 had no visible log progress. The complete
model and optimizer checkpoint at step 480 has agreeing global and prompt
counters. Continue the same cell, original job 31014459, with its exact prior
submission: node302/mltheory, one GPU, eight CPUs, 40 GiB host memory and
36-hour walltime. The new allocation inherits the already amended artifact
progress watchdog and starts a fresh twelve-retry budget. Submit held, audit,
register the continuation lineage, then release. Preserve scientific settings,
checkpoint contents, evaluation and the run directory.

For jointly reviewed node105 capacity, the highest-durable-progress pending
non-Pantry E119 continuation is Python ReplayDr.GRPO seed 45 job 31048204,
original job 31014487. Its complete step-1152 model and optimizer checkpoint
has agreeing saved counters. It uses learner microbatch 1 and fits the supported
24-GB A5000 runtime. The proposed scheduler-only backfill preserves the scientific cell, all
submitted environment settings, exclusions, eight CPUs, one GPU, 40 GiB
host-memory request and 36-hour walltime. The site forbids modifying existing
job accounts or partitions. Hold the old pending job; submit its exact command
with only account/partition changed to mltheory and requested node to node105;
audit the held replacement, register the continuation, cancel the old held
pending allocation, then release the replacement. No running cell is stopped.

This one E119 job accompanies three proposed E120 Falcon allocations at 64 GiB
each: combined incremental requests are four GPUs, 32 CPUs and 232 GiB host
memory. Root coordinates the joint capacity review and release; the E119
backfill remains a dry-run plan until that review is complete. Selection uses
durable training progress and hardware fit, not scientific outcomes.

Evidence and exact before/after records are in
`var/artifacts/campaign_health_capacity_20260905/`, including
`e119_pantry_s43_recovery.json` and the node105 backfill plan/receipt.
