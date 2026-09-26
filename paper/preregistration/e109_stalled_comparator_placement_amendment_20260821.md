# E109 amendment: unstall the six remaining Python ReplayDr comparators

Frozen on 2026-08-21 EDT before application, while all six target jobs are
`PENDING` at `RunTime=00:00:00`. This amendment uses scheduler state, node
inventory, requeue counters, and the frozen E109 ledger only. No E105, E109,
E111, or E112-R1 post-update evaluation outcome was inspected. PointMaze is
excluded.

## Trigger and diagnosis

E109 released fifteen repaired-parser Python ReplayDr.GRPO comparators on
2026-08-18. Nine reached `TRAINING_COMPLETE.json`. The six that remain are
stalled for two distinct scheduler reasons, neither of which is scientific.

**Falcon-1B seeds 55, 56, 58, 59** (jobs `30659546`, `30659547`, `30659549`,
`30659550`) are `Priority=0`, `Reason=JobHeldUser`, `Restarts=0`,
`RunTime=00:00:00`, with no run directory and an unchanged
`SubmitTime=2026-08-18T10:19:35`. The launcher released all fifteen jobs at
submission, so this hold was not applied by the campaign. Each of the four is
pinned to exactly one node: `node207`, `node205`, `node207`, `node205`. Every
other Falcon `cs` job pinned to exactly one of those two nodes is in the same
state (`e112r1-f1-*`, `e100-pantry-rlep-*`), while the E109 Falcon cell pinned
to `node206`, seed 57, allocated and completed, and the two-node
`node[205,207]` E100 replacements remain eligible. All `cs`/`all` A6000 nodes
restarted `slurmd` between 16:48 and 17:09 EDT on 2026-08-20, and the stalled
jobs were last scheduler-evaluated at 19:50 EDT that day. The single-node
requirement, not the science, is what makes these cells unschedulable.

**Qwen2.5-3B seeds 73, 74** (jobs `30659554`, `30659555`) are eligible but sit
on `lowprio`, which is `PriorityTier=1` and `PreemptMode=REQUEUE`. They have
been preempted and requeued 11 and 9 times respectively. Seed 73 holds a
`step_00192` checkpoint and resumes; seed 74 has never survived long enough to
write its first checkpoint at step 192, so every one of its nine allocations
was discarded. The A6000 pool is fully allocated and `node805` is `DOWN`.

## Authorized scheduler-only change

While each job remains pending at zero runtime with its exact frozen
scientific environment:

- For the four Falcon-1B cells: change partition `cs` to `all`, change the
  single-node requirement to the registered A6000 pool
  `node[103-104,205-208,805]`, then release the hold. Retain account `allcs`,
  one A6000, 8 CPUs, 64 GiB, and the three-day limit.
- For the two Qwen2.5-3B cells: change partition `lowprio` to `all`. Retain
  account `mltheory`, node pool `node[103-104,205-208,805]`, one A6000,
  16 CPUs, 128 GiB, and the three-day limit.

The A6000 pool and the `cs`-to-`all` move are the same ones already registered
in `e106_falcon_a6000_all_partition_pool_amendment_20260817.md`; the
`lowprio`-to-`all` move for a Qwen-3B A6000 cell is the one already registered
in `e106_qwen3_preemption_clean_restart_all_partition_amendment_20260818.md`.
Partition `all` is `PreemptMode=OFF`, so the change also removes the requeue
loop that is destroying seed 74's prefixes.

The GPU type does not change, so the registered cell-by-cell hardware-class
match to the paired E105/E112-R1 Python treatment cells is preserved. The
model, source and ops snapshots, seed, data, prompts, parser surface,
optimizer, learning rate, group size, replay objective and coefficient,
semantic coefficient, evaluation cadence, checkpoint cadence, stopping rule,
output path, auto-resume, and requeue behavior do not change. The application
script must verify that the full `--export` environment of every target is
byte-identical to the environment frozen in
`var/artifacts/e109_repaired_python_replay_comparators_jobs.json`.

No run directory is archived or deleted. Seed 73 resumes from its existing
step-192 checkpoint exactly as it would have on `lowprio`; seed 74 has no
resumable checkpoint and starts from the same frozen initialization, which is
the behavior it has had on every previous allocation.

The application is transactional: if any update, release, or post-update audit
fails, every job changed by that invocation is restored to its recorded
partition and node list, and every job that was held is re-held. A
content-addressed artifact records the protocol, script, ledger, node
inventory, requeue counters, and before/after scheduler records.

## Gate consequence

This amendment changes placement eligibility only. It creates no endpoint,
authorizes no outcome inspection, and cannot change the E109 estimand. Each
cell must still reach 3,072 optimizer steps and write its own completion
receipt to count toward the fifteen-cell Python ReplayDr comparator set. If a
cell fails after this change, it fails as a scientific or infrastructure
outcome on its own record.
