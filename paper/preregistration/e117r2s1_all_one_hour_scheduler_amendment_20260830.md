# E117-R2-S1: remaining cells to all with a one-hour bound

Frozen: 2026-08-30, after E117-R2 job 30970803 completed and before any of
jobs 30970804--30970814 started. Inspection used scheduler accounting and
wall-clock runtime only. No mechanism telemetry, accuracy, arm contrast,
audit outcome, or efficacy statistic was inspected for this amendment.

## Trigger

Job 30970803 completed all 64 requested updates successfully on node202 in
00:11:07. The remaining 11 authoritative jobs are still pending with zero
runtime, no assigned node, and no output directory. Their current effective
route is partition `cs`, account `allcs`, QOS `none`, and a two-hour time
limit; Slurm projects starts from 2026-08-31 03:00 through 20:00.

The directly matched E117-R1 cells completed in 11:06--48:37. A one-hour
limit therefore remains above every matched observed runtime while making
the untouched R2 jobs suitable for the site's short `all` backfill surface.

## Exact scheduler-only amendment

Apply a user hold to jobs 30970804--30970814 and verify every job is still
pending with RunTime=00:00:00, Restarts=0, no assigned node, the exact frozen
exported environment, and no output directory. Then update only:

- partition: `cs` to `all`;
- account: `allcs` to `mltheory`;
- time limit: `02:00:00` to `01:00:00`.

Preserve QOS `none`, every job ID, dependency, physical node202/node203 pin,
GPU/CPU/memory request, run stamp, source snapshot, model, domain, arm, seed,
data, mechanism configuration, target updates, and audit rule.

After all 11 held records pass the amended audit, durably record the
transaction in the authoritative E117-R2 ledger and a separate receipt, then
release all 11 jobs together. If any pre-ledger check or update fails, restore
the original scheduler route and release the unchanged jobs. After a ledger
write, fail closed rather than attempt an implicit rollback.

Job 30970803 and audit job 30970815 are excluded from mutation. The official
audit remains dependency-gated on all 12 original science job IDs and must
still pass before Stage 1. A one-hour timeout is an official R2 failure and
may not be waived, extended retrospectively, or combined with an old cell.
