# E122 Slurm 25.11 node-count display amendment

September 9, 2026, after the first held submission and before any E122 training.

E122 remains the registered Qwen2.5-0.5B-Instruct Level-3 factorial: five domains,
four arms and seeds 43–47, with all 100 scientific configurations unchanged.
The original plan is frozen at SHA-256
67c506a40b9a7fb7b984335d2ac47c859e01c9a62e5eddba5891aca9401cec2f.

The first submission succeeded exactly once as job 31158645, Countdown Dr.GRPO
seed 43. Slurm returned it as PENDING / JobHeldUser, with zero runtime and zero
restarts. Its command and all 128 environment entries match the frozen plan.
The subsequent audit stopped because Slurm 25.11 displayed NumNodes=1-1
(minimum one, maximum one), whereas the original audit expected NumNodes=1.
Independent inspection confirmed that this display spelling is the sole audit
difference. Both spellings specify exactly one node.

A new, separately authenticated adapter normalizes only this single field for
comparison by the original strict audit. It preserves the raw scheduler record
in all saved evidence and rejects any wider node range or duplicate field.
The original launcher, controller, runtime snapshot and plan remain unchanged.
The controller uses its original release and storage rules; its persisted
binding also records the additive amendment path and SHA-256. Every subsequent
campaign load rechecks the amendment's source and evidence hashes.

A separate continuation reconciles and retains job 31158645. It requires the
exact original submission claim, one successful index-000 result, the unchanged
prospective ledger, no later submission intents, no run directories and the
single expected held E122 allocation. It takes the original shared submission
lock and records a new exclusive continuation intent before submitting only
indices 001–099. All new attempts retain the existing per-cell intent/result
protocol. Any ambiguous result or interruption stops automatic continuation;
no existing attempt is repeated. All 100 jobs must pass the adapted original
strict audit twice before the complete held ledger is published.

No treatment outcome informed this amendment. No training configuration,
checkpoint policy, priority, dataset, model, seed, stopping rule, storage budget
or release limit changes.
