# E109-R1-S4 / E114-R1-S2 owner-queue reorder

Frozen: 2026-08-26 EDT after the scheduler-only backfill amendment was applied
and before this queue reorder, without reading any incomplete-cell endpoint.

The prior amendment made E109 jobs `30874012` and `30874013` compatible with
the full registered A6000 pool and reduced E114 job `30790267` to a 24-hour
A100 reservation. A subsequent scheduling cycle showed E109 blocked by
`Priority` behind the same owner's larger campaign backlog. E114 retained a
high priority but remained capacity-blocked on node302.

Use Slurm's owner-scoped `scontrol top` operation on the three exact completion
jobs. This operation reorders each job only among jobs belonging to the same
user with matching scheduler associations; it does not cancel, preempt, hold,
or alter another user's work. Retain all scientific settings, hardware class,
resource requests, node constraints, time limits, run directories,
checkpoints, and stopping rules installed by the preceding amendment.

Record exact before/after scheduler rows and accept either pending or running
state after the reorder. This is a scheduling-priority action only; E109 stays
13/15 and E114 stays 19/20 until their existing cells write terminal receipts.
