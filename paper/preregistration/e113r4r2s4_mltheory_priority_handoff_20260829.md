# E113-R4-R2-S4 mltheory priority handoff

Frozen: 2026-08-29T19:09:17-04:00 after E117-R1 became 12/12 terminal and
before this scheduler transaction.

Status: scheduler-only account/fair-share amendment. No incomplete E113-R4
endpoint was inspected. This amendment does not change a scientific cell,
official-verl source, runtime image, model/data/seed assignment, DAPO recipe,
sampling distribution, reward, optimizer, checkpoint, evaluation surface, or
stopping rule.

## Trigger

E117-R1 is terminal with all 12 jobs completed `0:0`, releasing the campaign's
active execution priority. All 46 nonterminal E113-R4-R2 jobs
`30869115`--`30869160` are released, pending at zero runtime and zero restarts,
and blocked only by `Priority`. Owner-side `scontrol top` is disabled on this
cluster.

The jobs currently use account `allcs`, whose contemporaneous user fair-share
factor is 0.369963. The user's `mltheory` association has fair-share factor
0.802198. Partition `all` explicitly permits both accounts and all QOS values;
its seven-day maximum exceeds the jobs' frozen 12-hour limits. The completed
E117-R1 jobs also demonstrate that `Account=mltheory, Partition=all` is a live
in-place scheduler route.

The original R4-P1 repair was required because submit-time routing changed an
`mltheory` submission to the A5000-only `mltheory` partition. This amendment is
an in-place transaction that explicitly retains `Partition=all` and the exact
`gres/gpu:a6000:1` request, so it does not restore that invalid placement.

## Authorized transaction

For exactly jobs `30869115`--`30869160`, verify the authoritative R4-R2 ledger,
the S3 release record, pending/priority state, zero runtime, zero restarts,
absent run directories, byte-exact scientific exports, `Partition=all`,
`Account=allcs`, `QOS=long`, requeue enabled, no dependency, one A6000 GPU,
16 CPUs, 128 GiB, and a 12-hour limit.

Transactionally user-hold all 46 jobs, revalidate the boundary, change only
the scheduler billing account from `allcs` to `mltheory` while explicitly
retaining `Partition=all`, validate every job while held, record provenance,
and release all 46 together. Preserve QOS, hardware class, resource requests,
time limit, priority nice value, job IDs, run directories, commands, and all
scientific exports. On any pre-release failure restore `Account=allcs`, retain
`Partition=all`, and release every temporary hold.

After release accept pending or running state, require that no job remains
held, and report live scheduler placement. This transaction changes accounting
and fair-share priority only; it does not preempt, cancel, hold, or otherwise
modify another user's work.
