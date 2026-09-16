# E94-PT / E96-PT incomplete-chain execution repair

Date: 2026-08-14. This amendment was written after five terminal scheduler
jobs were found below the registered 3,072-update endpoint and before any
additional updates were run.

The affected cells are E96-PT semantic seeds 43 and 45, E94-PT control seeds
58 and 59, and E94-PT replay seed 59. Their retained checkpoints are at 2,688,
2,880, 2,880, 2,880, and 2,880 updates, respectively.

The original chains used `afterany`. A failed or timed-out chunk therefore
consumed one scheduled chain position, later chunks resumed correctly, and the
finite chain ended short. This is an execution-accounting fault, not a change
to an arm or estimand.

Each affected cell receives exactly one additional one-hour chunk with the
original command and scientific environment. The dependency is removed, the
job is submitted held and audited, and execution is constrained to healthy
A6000 nodes 103, 104, 205, 207, and 805, avoiding the fault history on node206
and all drained nodes. The
existing checkpoint, metrics stream, target, model, data, seed, arm, and
evaluation settings are unchanged. The runtime stops at 3,072 even when the
configured chunk size is larger than the remaining work.

The primary ledgers retain every old chunk ID and append the repair ID. A
separate repair ledger records the pre-repair checkpoint, old terminal job,
held scheduler record, node, and replacement job. No below-target terminal
cell may be reported as complete before its checkpoint and endpoint receipt
reach the registered target.
