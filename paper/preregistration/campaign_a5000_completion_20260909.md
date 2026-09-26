# E118/E120 A5000 completion amendment — September 9, 2026

The user requested additional concurrent E118, E119 and E120-R1 jobs and hoped
for completion today. This operational amendment admits five existing cells
to available non-PVL CS-node capacity through the permitted borrowing queue; it adds no scientific cells and changes no
scientific endpoints, training horizon, selection rule or frozen source.

Four Qwen2.5-3B E118 MathIR cells (MaxRL and ReplayMaxRL, seeds 72 and 73;
current jobs 31142254, 31142763, 31142770, 31143241) are pending on node208,
which is draining due to GPU overheating. The fifth cell is Qwen2.5-3B E120-R1
Graph fresh-frequency seed 74, original 31033709/current continuation31048518.
Its registered run directory and latest valid checkpoint remain authoritative.

The replacement allocation is one A5000, 16 CPUs, 72 hours, explicit nodes
202/203, account mltheory, partition lowprio, unchanged Nice/requeue/exclusion settings,
and unchanged 116 GiB host memory for each MathIR job and 128 GiB for Graph.
Only OAT_ZERO_VLLM_GPU_RATIO changes from 0.25 to 0.40, the previously qualified
24-GB actor cache allocation. This is the same resource allowance already
running for both MathIR seed71 arms (31151404,31151409) and E120 Graph seed73
(31151416), with the corresponding identical frozen training-source snapshots.
Before preparation and application, verify these admission runs are running
on their qualified nodes and show recent actual optimizer progress.

The cache allowance is not a sampling-temperature, rollout-count, sequence-
length, optimizer, offloading, replay, treatment or learning-rate change.
Keep every other runtime export byte-equivalent, including SAVE_PATH, RUN_STAMP,
model/revision, seed, source/ops roots and automatic checkpoint recovery.
The prior qualification and sleep/wake/cache rationale are documented in
`e118_owner_backfill_20260908.md` and `e120_graph_owner_fallback_20260908.md`.
E119 Pantry remains restricted to its proven larger-GPU routes: five prior
learner CUDA OOMs disqualify these 24-GB A5000 slots for that workload.

Use held predecessor/replacement transactions, explicit checkpoint and active-
writer checks, preserved launcher/source hashes, and source plus aggregate
ledger promotion under the existing E118/E120 ledger locks. Retire a held
predecessor only after its replacement is audited and ledgers are committed;
release replacements only after all predecessors are inactive. Preserve E120's
primary ledger byte-for-byte and append this placement only to its continuation
record. Store exact before/after IDs, resources and receipts in
`var/artifacts/campaign_a5000_completion_20260909/`.

No healthy running allocation is interrupted by this transaction. Scheduler
capacity and GPU memory class, rather than efficacy outcomes, determine these
moves. Current availability allows five allocations geometrically, but actual
starts remain subject to scheduler priority and other users' reservations.

## Queue selection before execution

The initial allcs/cs preparation forecast September17 starts despite unallocated
hardware. Independent test-only requests found allcs/lowprio forecast tomorrow,
while the user's existing mltheory association on lowprio forecasts immediate
starts for the same72-hour requests. The final plan therefore uses that eligible
account and borrowing partition. The site submit plugin assigns its normal QoS;
no QoS override is requested. Jobs remain preemptible with automatic requeue and
validated checkpoint recovery. The unused first plan is archived as
plan.initial_cs.json; no predecessor was held or changed during that preparation.
