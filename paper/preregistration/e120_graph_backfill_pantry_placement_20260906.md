# E120 Falcon capacity changes — September 6, 2026 UTC

The user explicitly approved both E120 capacity changes identified in the live
throughput review: move Falcon Graph seed59 to node302's low-priority queue and
allow Falcon Pantry seed55 to run on a healthy A6000 node instead of only drained
node206. The separately authorized E119 memory repair is recorded independently.

Graph: replace ordinary pending31048529 (original31033699) with one held and
audited continuation on node302, accountmltheory, partitionlowprio. Preserve
64GiB,8CPUs,oneGPU,72hours,Nice100 and the complete original export argument,
source wrapper, scientific configuration and checkpoint auto-resume. The latest
validated checkpoint is2496;134previously logged unsaved updates are repeated.
Record the replacement in the existing authoritative continuation ledger before
cancelling the old pending identity. Release only after the old writer is inactive.
Low-priority allocations can be preempted and requeued; startup must be checked.

Pantry: preserve job31033700 and replace its single requested node206 with the
established non-PVL A6000 pool node205,node207. Preserve accountallcs,partitioncs,
64GiB,8CPUs,oneA6000,72hours,Nice100,source and scientific settings. It has not yet
produced optimizer metrics or a checkpoint, so its intended first training start
remains from initialization. Hold the pending job during the placement update,
audit its actual scheduler record, record the placement amendment in the
continuation ledger's operational metadata, and release the owned hold.

The original45-cell ledger remains byte-identical. All ten deliberately held
E120Qwen3B cells and E118 allocations retain their settings and state. Plans,
checkpoint certificates, scheduler records and execution receipts are under
`var/artifacts/e120_capacity_actions_20260906/`.
