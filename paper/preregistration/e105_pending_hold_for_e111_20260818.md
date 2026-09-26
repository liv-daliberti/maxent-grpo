# E105 pending hold while E111 completes

Date frozen: 2026-08-18, before applying holds.

E105 v6 has already been prospectively superseded by E111/E112. To prevent
additional superseded E105 cells from starting while the corrected E111 gate
waits for hardware, place a reversible user hold on exactly the E105 jobs that
are `PENDING` at capture time. Do not signal, cancel, requeue, suspend, or
otherwise modify any E105 job that is `RUNNING` or terminal.

This is an administrative resource hold, not an outcome or treatment change.
The record must preserve the E105 ledger digest, supersession-note digest,
exact pending/running ID sets, before/after scheduler records for held IDs,
and prove that held jobs have `JobHeldUser` state/reason. No endpoint outcome
is inspected. PointMaze remains outside the corrected goal.

The hold does not satisfy E112's inactive-E105 release interlock: explicit
authorization is still required before canceling/retiring E105. All holds are
reversible with `scontrol release`.
