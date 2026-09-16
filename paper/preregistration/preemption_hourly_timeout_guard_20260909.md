# Hourly recovery for three repeatedly preempted cells, September 9, 2026

The user requested recovery and accelerated completion of E118, E119, and E120.
This guard covers only the hourly successors of existing jobs 31158505 (E118
MathIR MaxRL seed 73), 31158506 (E118 MathIR ReplayMaxRL seed 73), and 31158507
(E120-R1 Qwen-3B Graph seed 74). Their durable checkpoints are respectively
960, 960, and 576. Five repeated lowprio preemptions left those checkpoints
unchanged. Ordinary one-hour all/mltheory allocations forecast immediate
capacity on the qualified A5000 pool node105/node202/node203/node204.

Capacity staging preserves all scientific and evaluation settings, the existing
0.40 actor cache ratio, 16 CPUs, one A5000, and 116/116/128 GiB host memory.
The sole runtime export change is rolling recovery checkpoint cadence 192 to
48 updates. The original jobs remain held as exact dormant 72-hour fallbacks.
The E118 source and aggregate mappings and E120 continuation mapping remain
coherent; E120's original scientific ledger remains unchanged. The capacity
controller owns promotion, fallback and completion retirement.

The guard freezes exact SubmitLine, effective resources, source helpers,
scientific identity, initial restart counts, and durable progress floors.
It verifies model and optimizer ZIP metadata without deserializing tensors:
model global_steps/global_step/policy_sgd_step/prompt cursor and all optimizer
step counters must match the 48-grid directory tag, with replay bank state.
Only an inactive, accounted TIMEOUT with a checkpoint strictly newer than
its last resumed checkpoint may receive one same-ID requeuehold/release.
The first retry must exceed its registered initial floor. Every transition
checks exact authoritative identity and excludes unexpected active or pending
writers; the only exemption is its exact audited held predecessor.

Retries are limited to 24 per cell and a persisted absolute 24-hour deadline.
A running final allocation remains untouched through its one-hour limit;
monitoring continues through a 65-minute cleanup window. An already committed
pre-deadline requeue transition may finish its one release within the first
five minutes of cleanup, after rechecking the exact owned hold and restart
increment. No new requeue is permitted then. An uncertain scheduler call is
never blindly repeated. Lost acknowledgements for persisted retry, fallback or
retirement transactions receive at most eight bounded reconciliation errors
within the same cleanup deadline; reconciliation never issues another requeuehold.
After the five-minute release grace, a persisted pre-deadline requeue intent with
exact restart increment, unchanged identity/resources, advancing valid checkpoint
and exact old hold may normalize only its proven requeue-owned hold to a user hold.
The intent survives a lost normalization acknowledgement. A durable receipt in the
capacity item's deadline fallback records that ownership before cancellation and
restoration. Unrelated user/admin holds are never claimed. Other failures, changed
identity/resources, missing or corrupt checkpoints stop for diagnosis.

A no-progress TIMEOUT, exhausted cap, or deadline uses only capacity.fallback
under the common ledger lock. It makes the successor inactive, restores the
exact original authoritative mapping, and releases only its owned predecessor.
A deadline hold racing allocation preserves the active allocation and defers
fallback. Terminal receipts invoke capacity.retire_completed to retire only
the unused dormant hold. Each cell has independent retry and fallback state.

All actual successor IDs and hashes are frozen before the CPU guard starts;
capacity release requires a verified running guard. The CPU supervisor has
25 hours 10 minutes, with the earlier persisted internal deadlines authoritative.
No scientific jobs are submitted by preparing this guard or running its tests.
