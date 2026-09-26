# AntMaze v10 exact-slate cross-node audit

**Status: FROZEN DURING V10 CONTROLLER TRAINING, BEFORE ITS OUTCOME OR ANY V10 ROUTE EXECUTION — 2026-07-29**

This audit is conditional. It may run only if the frozen v10 controller gate
and the separately frozen 11x11 route-admission gate both pass. A failed or
absent antecedent stops the launch.

The audit replays the exact 12 admitted data rows, two recorded witness
programs, reset seeds, specifications, expected canonical keys, v10 controller
receipt/model, and route-job source snapshot. It does not regenerate maps from
current source. A deterministic export of the admitted Arrow rows and route
identity is hash-bound before the cross-node job is released.

Three distinct Slurm nodes each execute every one of the 24 map-route pairs
three times: 72 executions per node and 216 total. Every execution must pass
the unchanged executable verifier, finish within the frozen environment goal
threshold 0.5, reproduce the recorded canonical topology key and directed
gate, and retain all controller/environment/spec identities. No stronger or
weaker endpoint threshold is introduced here.

The aggregate passes only if all 216 executions pass, all three receipts bind
the same protocol/source/execution/spec export, and the receipts name three
distinct nodes. Raw executions and the first validation error remain in the
replica receipts. There is no map, program, seed, threshold, node-result, or
route substitution.

A pass authorizes only the separately frozen Qwen2.5-0.5B capability gate on
the already admitted train/development split. It does not authorize a main
seed or paper result. A failure stops AntMaze v10.
