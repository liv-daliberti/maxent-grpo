# E100 amendment: unstall the five remaining Falcon PantryPlan cells

Frozen on 2026-08-21 EDT before application, while all six target jobs are
`PENDING` at `RunTime=00:00:00`. This amendment uses scheduler state, node
inventory, requeue counters, run-directory manifests, and the frozen E100
ledgers only. No E100 outcome endpoint was inspected. PointMaze is excluded.

## Trigger and diagnosis

E100 is 17/22 terminal. All five nonterminal cells are PantryPlan, and all
five are stalled on placement rather than on science.

Science jobs `30572828` (seed 56), `30572829` (seed 57), and `30572831`
(seed 59) are `Priority=0`, `Reason=JobHeldUser` at `RunTime=00:00:00`. The
E100-R2 recovery of 2026-08-19 released these three exact job IDs, so the hold
is not the campaign's. Each is pinned to exactly one node --- `node207`,
`node205`, `node207` --- and every single-node-pinned Falcon `cs` job across
E100, E109, and E112-R1 is in the same state, while the two-node
`node[205,207]` replacements submitted the same day stayed eligible. This is
the auto-hold registered in
`e109_stalled_comparator_placement_amendment_20260821.md`. Their requeue
counters are 2, 6, and 2 and their run directories contain `train_metrics`
prefixes with no checkpoint at all, so every one of those allocations was
discarded.

Seed 55's replacement science job `30790683` and seed 58's replacement pool
collection `30790680` are eligible but pending on `Priority` against a fully
allocated A6000 pool restricted to two nodes. Seed 58's science job `30790684`
is `afterok`-blocked behind that pool collection and its audit `30790681`, so
one two-node pool job currently gates two of the five remaining cells.

## Authorized scheduler-only change

While each job remains pending at zero runtime with its exact frozen
scientific environment, for jobs `30572828`, `30572829`, `30572831`,
`30790680`, `30790683`, and `30790684`:

- change partition `cs` to `all`;
- change the node requirement to the registered A6000 pool
  `node[103-104,205-208,805]`; and
- release the three held science jobs.

Retain account `allcs`, one A6000, 8 CPUs, 64 GiB, the three-day limit, the
`afterok:30790681` dependency of `30790684`, and every element of the frozen
scientific environment: source and ops snapshot, model, sparse RLEP-Dr
objective, replay eligibility rule, domain, seed, data, prompts, optimizer,
group size, evaluation and checkpoint cadence, stopping rule, and output path.
The application script must verify that the full `--export` environment of
every target is byte-identical to the environment frozen in
`var/artifacts/e100_sparse_rlep_dr_falcon1b_jobs.json` or, for the pool
collection, `var/artifacts/e100_pantry_infrastructure_recovery_jobs.json`.

The pool and node list are the ones already registered in
`e106_falcon_a6000_all_partition_pool_amendment_20260817.md`. The GPU type
does not change. Partition `all` is `PreemptMode=OFF`, so the change also
removes the requeue exposure that has been discarding the three held cells'
prefixes.

No run directory is archived, altered, or deleted. None of the three held
cells holds a resumable checkpoint, so each starts from the same frozen
initialization it has started from on every previous allocation. The CPU-only
audit job `30790681` already runs in partition `all` and is not modified.

## Gate consequence

This amendment changes placement eligibility only. It creates no endpoint,
authorizes no outcome inspection, and cannot change the E100 estimand or the
registered zero-eligibility classification of the three gate-blocked cells.
Each cell must still reach 3,072 optimizer steps and write its own completion
receipt, and seed 58 must still pass its independent pool audit before its
science job may run.
