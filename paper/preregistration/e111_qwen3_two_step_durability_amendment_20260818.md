# E111 Qwen-3B two-step durability amendment

Date frozen: 2026-08-18, before changing the frozen runtime script and without
inspecting any E111 endpoint result.

The effective eight-step durability amendment was consumed by four Qwen-3B
jobs. Scheduler and trainer-step telemetry then showed one allocation ending
after approximately 3.5 minutes at trainer step 2, before checkpoint 8. Thus
the eight-step interval remains longer than an observed A6000 backfill window.
This decision uses only allocation duration, trainer step, and checkpoint
presence—not reward, accuracy, coverage, or another endpoint.

For future allocations of exact E111 Qwen-3B jobs `30674758`–`30674762`,
change only the runtime storage cadence:

- save interval/from: 8 -> 2 steps
- resume interval: 8 -> 2 steps

The fail-closed checks against the original submitted values (`32`), exact job
IDs, source/ops snapshots, and treatment variant remain. Atomic checkpoint
serialization, two-checkpoint retention, model, seed, data, prompt order,
optimizer, learning-rate schedule, MaxEnt/ReplayDr objectives, proposal policy,
evaluation, hardware, and target steps remain unchanged. Running attempts use
their already copied eight-step script and are not signaled or reset. PointMaze
remains excluded.

The amendment record must chain the prior runtime record/block digests to the
new before/after script and block digests, capture exact scheduler records and
restart counts, and state that endpoint outcomes were not inspected. The final
E111 auditor must validate the live two-step block and this digest chain.
