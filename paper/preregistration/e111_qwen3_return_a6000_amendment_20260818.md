# E111 Qwen-3B return-to-A6000 amendment

Date frozen: 2026-08-18, before changing placement and without inspecting an
E111 endpoint evaluation result.

The prospective L40 placement produced no allocation or training step:
node403 is reserved/planned for other work through the relevant backfill
window. All five exact Qwen-3B jobs remain pending. Return jobs `30674758`,
`30674759`, `30674760`, `30674761`, and `30674762` to the original matched
A6000 pool:

- required node list: `node403` -> `node[103-104,205-208]`
- GRES: `gpu:l40:1` -> `gpu:a6000:1`

Keep the separately frozen 45-minute backfill limit and effective eight-step
runtime checkpoint cadence. Those two storage/scheduler changes allow short
A6000 allocations to preserve exact optimizer/RNG state instead of repeatedly
discarding a sub-32-step prefix.

Partition, account, CPU/memory/GPU counts, environments, stored batch scripts,
run directories, models, seeds, data, optimizer, MaxEnt and ReplayDr
objectives, proposal policy, evaluation settings, and target steps remain
unchanged. No job is canceled, requeued, reset, or duplicated. E112 remains
A6000-paired. PointMaze remains excluded, and no endpoint outcome motivates
or gates the amendment.

The final E111 auditor must validate exact before/after scheduler records,
unchanged submit lines, the E111 ledger/protocol digests, the pending status
before return, and the exact node/GRES transition above.
