# E114-R1-S3 next-slot priority fence

Frozen: 2026-08-26 EDT after the user explicitly requested that the final E114
cell run next, before this scheduler transaction, and without reading the
incomplete MathIR seed-72 evaluation endpoint.

## Outcome-blind scheduler diagnosis

Final E114 job `30790267` is released and pending on its registered A100
node302 with a valid step-1,728 checkpoint, 24-hour limit, and `Nice=0`. Twelve
pending E112 Qwen-3B jobs are also pinned to node302 in partition `mltheory` and
currently have priority 8,918 versus E114's 8,709. A second E114 scheduler row,
Pantry seed 73 job `30790272`, is pending at priority 8,918 even though its run
directory contains a valid `TRAINING_COMPLETE.json` at step 3,073 and a
terminal export. Its resume checkpoints were pruned on successful completion;
allowing that stale requeue to run would duplicate an already terminal cell.

## Authorized scheduler-only transaction

- Cancel only stale completed scheduler row `30790272`, after validating its
  frozen release-ledger identity and terminal receipt.
- Temporarily change only `Nice=100` to `Nice=1000` for the twelve exact
  pending E112 node302 competitors: `30791537`, `30791538`, `30791539`,
  `30791540`, `30791541`, `30791542`, `30791543`, `30791544`, `30791545`,
  `30791546`, `30791549`, and `30791554`.
- Do not hold, cancel, preempt, or modify any running job. Do not modify E114
  job `30790267` itself.
- As soon as `30790267` is observed RUNNING (or terminal), restore `Nice=100`
  on each fenced E112 job that remains pending.

This fence changes owner-level scheduling order only. It retains every E112
and E114 scientific environment, job ID, run directory, checkpoint, hardware
class, resource request, node constraint, time limit, evaluation surface, and
stopping rule. External users and running E112 jobs are unaffected. The fence
can ensure that no pending job in this exact higher-priority campaign set takes
node302 before E114; it cannot override cluster reservations or other users.

