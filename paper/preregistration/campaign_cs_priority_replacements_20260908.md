# Six existing cells: cs priority replacements — September 8, 2026

The user explicitly requested requeuing broken cells and completing E118, E119
and E120 as quickly as possible. The separate pending-only scheduler amendment
removes independent resource gates and expands viable node pools. Site policy
forbids modifying Account, Partition or TimeLimit after submission and normalizes
multipartition requests. This amendment follows the supported cancellation and
resubmission path to move six ordinary pending lowprio requests to cs.

The exact predecessor IDs and existing cells are:

| Predecessor | Existing cell | Unchanged vLLM ratio | New cs candidate pool |
|---|---|---|---|
| 31146150 | E118 Qwen-3B Countdown MaxRL seed73 | 0.25 | node205/node206/node207, A6000 |
| 31141497 | E118 Qwen-3B MathIR MaxRL seed71 | 0.40 | node202/node203/node204, A5000 |
| 31141741 | E118 Qwen-3B MathIR ReplayMaxRL seed71 | 0.40 | node202/node203/node204, A5000 |
| 31141972 | E118 Qwen-3B Countdown MaxRL seed74 | 0.40 | node202/node203/node204, A5000 |
| 31144883 | E118 Qwen-3B Countdown ReplayMaxRL seed74 | 0.40 | node202/node203/node204, A5000 |
| 31144919 | E120 Qwen-3B Graph fresh-frequency seed73 | 0.40 | node202/node203/node204, A5000 |

Each retains account allcs, 72-hour walltime, CPU count, host-memory request,
GPU count and current requested GPU type, nice value, requeue eligibility,
exclusions, complete exported environment, frozen launcher/source, model, data,
seed, objective, optimizer, eight-pass horizon and evaluation/checkpoint cadence.
Keep each SAVE_PATH and RUN_STAMP. Automatic resume uses the latest structurally
valid full model/optimizer checkpoint in that same run directory. If no durable
checkpoint exists, record that fact; do not relabel earlier pre-checkpoint work
as recovered. No result values select placements. Current ratio0.40 A5000
compatibility and ratio0.25 A6000 compatibility have separately audited runtime
evidence; this amendment introduces no runtime parameter changes.

The new partition is cs: observed priority tier100 versus lowprio tier1, and
cs is not subject to lowprio requeue preemption. Replacing these same-day pending
allocations loses their existing queue age and narrows lowprio borrowing routes,
but supplies access through the higher-priority owning partition for the entire
72-hour request. Slurm resource limits still decide allocation time; this is not
a claim that resources are immediately free. Preserve all other pending and
running allocations, including existing node302 owner requests and E120 Pantry.

Prepare only after the first pending amendment completes. Archive source,
150-cell aggregate and E120 continuation before-images and hashes. Validate the
immutable E120 45-cell primary fingerprint, six exact scientific identities,
ordinary pending status, no dependencies or dependents, no completion marker,
sole-writer identity, checkpoint file identity, launcher SHA-256, and complete
frozen source/ops code fingerprints. Test-only submission is read-only; effective
held-job records, rather than the dry-run forecast, establish accepted resources.

Under shared E118 and E120 ledger locks, hold a pending predecessor, submit one
held replacement, record its ID durably, and audit account/partition, candidate
nodes, all preserved resources and exact export equality. Uncertain submission
outcomes reconcile the unique scheduler comment; never blindly submit another
writer. Keep both requests held until every intended replacement is verified.
Stage all after-images before promotion; atomically replace each ledger file,
with before/after hashes and staged images permitting interrupted-pair recovery.
Update E118 source and aggregate IDs together, append previous IDs, and retain
full lineage. Update only the E120 continuation mapping and its scheduler
history; keep its 45-cell primary unchanged.

Only after all three effective mappings agree on the held replacements, cancel
the superseded held predecessors. Check every predecessor is absent before any
replacement is released. Recheck immutable runtime and checkpoint identity and
sole-writer state, then release each registered replacement. Record actual final
scheduler state and update E120's release receipt. A race to running or uncertain
partial operation stops for reconciliation with exact archived IDs and held
states. Resume the same transaction; completed transactions are idempotent.

Implementation: `ops/exp_scaling/replace_campaign_cs_priority_20260908.py`.
All plans, staged ledgers, fingerprints and action receipts reside in
`var/artifacts/campaign_cs_priority_replacements_20260908/`.
