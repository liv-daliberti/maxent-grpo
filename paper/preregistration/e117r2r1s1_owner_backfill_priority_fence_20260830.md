# E117-R2-R1-S1 owner backfill priority fence

Frozen: 2026-08-30T09:18:00-04:00, before this scheduler-only transaction and
without inspecting an incomplete E117 scientific endpoint.

Status: reversible owner-queue priority amendment. This amendment does not
change an E117 cell, source snapshot, model, data, seed, C/P/F mechanism,
optimizer, evaluation, stopping rule, job ID, command, scientific export,
resource request, node placement, account, partition, QOS, or audit dependency.

## Trigger and diagnosis

The signal-53 recovery released E117 replacement jobs `30977267`--`30977277`.
All eleven are pending at zero runtime and zero restarts on their frozen A5000
nodes, using `Partition=all`, `Account=mltheory`, `QOS=none`, one-hour limits,
and `Nice=0`. Their live multifactor priority is approximately 8075.

Nodes 202 and 203 have physically unallocated GPUs, but those slots have
backfill plans for other users. More importantly, the live Slurm configuration
sets `bf_max_job_user=64`. Eighty-three owner jobs eligible for scheduling have
priority strictly above E117, so E117 is outside the per-user backfill-test
window. A second fail-closed dry run minutes later observed 93 such rows,
showing that this eligibility set changes as dependencies and resources turn
over. Owner-side `scontrol top` is disabled. Moving E117 to
`Partition=cs, Account=allcs, QOS=short` would reduce its fair-share priority
to approximately 4300 and is therefore rejected.

## Authorized transaction

Install a temporary Nice fence on exactly these 52 owner-controlled jobs:

- E113-R4 signal-53 replacements `30977240`--`30977266` (27 rows); and
- all 25 maintenance-blocked E115 rows `30790293`--`30790302` and
  `30790304`--`30790318`.

Before mutation require every competitor to be owned by `od2961` and pending.
Require the E113 rows to retain their recorded `Nice=0` baseline and the 25
E115 rows to retain their pre-existing `Nice=100` priority-fence baseline.
Transactionally user-hold all 52 competitors, revalidate the boundary, change
only their Nice value to 1000, validate, and release them. On any pre-release
failure restore each row's distinct baseline and release all temporary holds.

The fence reduces the contemporaneous count of owner jobs strictly above E117
from the latest observed 93 to 41, retaining a 12-row ingress buffer. Thus all
eleven E117 rows fit in the configured 64-job user
backfill window. Do not hold, cancel, preempt, renice, or otherwise modify
another user's work. Do not alter the eleven E117 jobs or audit `30977278`.

## Restoration

After all eleven E117 rows have accounting evidence of a non-null start time,
restore a still-pending E113 competitor to Nice 0 and a still-pending E115
competitor to its pre-existing Nice 100. Running or already terminal
competitors require no mutation. Restoration is fail-closed and records both
the E117 start evidence and every restored scheduler record.

This fence changes queue order only. It cannot revoke another user's existing
backfill reservation and therefore promises eligibility at the next scheduler
opportunity, not preemption or an instantaneous allocation.
