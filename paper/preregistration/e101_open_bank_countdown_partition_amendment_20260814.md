# E101 zero-step partition amendment

Recorded: 2026-08-14, while jobs 30579498--30579500 remained pending with no
allocation, run directory, or metric.

After the same-family node/memory repair, all three jobs were resource-eligible
but remained behind a large lowprio array. The lowprio partition has priority
factor 1 and permits preemption; the all partition accepts the same allcs
account and exact node103/node104/node205/node206 A6000 list, has priority
factor 100, and does not preempt jobs.

The jobs are moved from lowprio to all. The exact node list, A6000 request, 48
GB host memory, CPUs, 55-minute cap, model, immutable source snapshot, data,
seed, objectives, and evaluation contract remain unchanged. Original and
post-amendment scheduler records are retained in the ledger.
