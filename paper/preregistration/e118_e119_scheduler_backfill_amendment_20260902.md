# E118/E119 scheduler backfill amendment — 2026-09-02

This is an operational scheduling amendment only. It changes no model, data, seed,
algorithm, optimizer, horizon, evaluation, or treatment setting, and modifies no running
or completed job.

E118 had fallen to four running jobs with a large pending queue because its remaining
Falcon-1B cells require a measured 62.5–64GB of host RAM and its Qwen-3B cells reserve
128GB. E119 Qwen-0.5B jobs were occupying the small host-memory gaps that remained,
preventing enough memory from accumulating for E118 backfill.

The following balanced pending cells were rerouted:

- 16 Falcon-1B jobs: eight complete MaxRL/ReplayMaxRL pairs, routed to `mltheory`
  on node105.
- 24 Qwen-3B jobs: twelve complete MaxRL/ReplayMaxRL pairs, routed to `mltheory`
  on node302.
- The remaining Qwen-3B jobs retain the `allcs` A6000 route on node205/node207.
- 27 pending E119 jobs on `mltheory` were restricted from `node105,node302` to
  node105, preventing small jobs from continually refilling node302 while Qwen-3B
  waits for 128GB of allocatable host memory.

This creates three E118 continuation lanes: Falcon on node105, Qwen-3B on node302,
and Qwen-3B on node205/node207. All moved E118 cells were selected as complete paired
blocks, so scheduler hardware is not confounded with MaxRL versus ReplayMaxRL.

Immediate audit: four E118 Qwen-3B jobs remained running normally on node205/node207,
