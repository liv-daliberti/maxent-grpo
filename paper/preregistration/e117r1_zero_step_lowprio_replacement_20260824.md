# E117-R1 zero-step lowprio replacement

Frozen: 2026-08-24 before replacement submission and without reading any E117
run artifact or endpoint. PointMaze remains excluded.

## Trigger and retirement boundary

Original E117 jobs 30873543--30873554 are unschedulable because the submit
router resolved their eight-hour request to partition `mltheory` while their
frozen nodes 021/022 are members of `lowprio`, not `mltheory`. E117-S1 attempted
an in-place repair. Its transaction stopped before the first partition update
because this site's plain `scontrol hold` created an administrative hold rather
than a user hold. The rollback could not release that administrator-owned hold.

All 12 original jobs remain `PENDING`, `RunTime=00:00:00`, `Restarts=0`, on
partition `mltheory`, with no allocation and no run directory. No model,
optimizer, replay, proposal, semantic, evaluation, or endpoint state exists.
Audit job 30873569 depends on those original IDs and is therefore stale.

## Authorized replacement

Submit exactly 12 replacement jobs using the immutable E117 snapshot
`e76_tuned_scale_96ca9fe5ff052c7d`. Every exported environment byte, run path,
model/data/seed, optimizer, replicated sampler, proposal budget, C/P/F arm,
semantic coefficient, checkpoint/evaluation schedule, node binding, GPU count,
CPU/memory limit, time limit, and Nice value must match the original release
ledger. The only submission difference is an explicit satisfiable partition:

- original resolved partition: `mltheory`;
- replacement partition: `lowprio`.

Keep account `mltheory` and frozen nodes 021/022. Submit every replacement held,
verify the complete export against its original held record and the ordinary
E117 held-job contract, and verify `Partition=lowprio`. Only after all 12 pass:

1. cancel stale audit job 30873569;
2. cancel and preserve the 12 zero-step original job records;
3. atomically record original-to-replacement mappings;
4. release all 12 replacements; and
5. submit a new after-any mechanism audit against the replacement ledger using
   the original immutable audit script.

If held replacement validation fails before retirement, cancel only the new
held jobs. The scientific experiment remains E117: this is a zero-step
scheduler replacement, not a new treatment, seed, or analysis opportunity.
