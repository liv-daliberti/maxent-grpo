# E109-R1-S3 / E114-R1 completion-backfill acceleration

Frozen: 2026-08-26 EDT after the user explicitly requested that the final E109
and E114 cells be made runnable, before this scheduler transaction and without
reading either incomplete cell's evaluation endpoint.

## Trigger and outcome-blind diagnosis

E109 continuation jobs `30874012` and `30874013` remain released and pending at
zero continuation runtime with independently validated checkpoints at steps
2,304 and 2,112. They request one A6000, 16 CPUs, 128 GiB, and twelve hours from
`node[103-104,205-207,805]`. The compatible, previously registered A6000 pool
also includes node208; current delay is accelerator and memory saturation, not
a scientific or checkpoint failure.

E114 job `30790267` is released and pending after one watchdog requeue, with an
independently validated step-1,728 checkpoint and 1,344 of 3,072 registered
optimizer steps remaining. It is pinned to its registered A100 node302 and
requests 16 CPUs, 128 GiB, and 72 hours. The current delay is node302 memory
saturation and backfill reservation fit, not a scientific failure.

## Authorized scheduler-only transaction

Provided each exact job remains pending, released, and scientifically
unchanged immediately before mutation:

- widen only E109 jobs `30874012` and `30874013` from
  `node[103-104,205-207,805]` to the compatible registered A6000 pool
  `node[103-104,205-208,805]`, retaining their twelve-hour limits;
- reduce only E114 job `30790267` from 72 hours to 24 hours, retaining node302
  and one A100; and
- attempt to normalize `Nice=100` to `Nice=0` for all three jobs if the
  scheduler permits the owner to decrease nice. Rejection of this optional
  priority adjustment does not invalidate the node/time changes and must be
  recorded.

The transaction temporarily holds the exact jobs, applies the amendments,
then releases them. It retains job IDs, seeds, variants, models, data, source
snapshots, run directories, checkpoints, optimizer and sampling settings,
evaluation surfaces, stopping rules, automatic resume, and watchdog behavior.
It does not change hardware class, inspect an incomplete endpoint, or create a
replacement scientific cell.

## Consequence

This amendment increases scheduling opportunities only. E109 remains 13/15
and E114 remains 19/20 until each existing cell reaches 3,072 optimizer steps
and writes its registered terminal receipt. All before/after scheduler records,
checkpoint validations, attempted changes, and any scheduler rejection are
captured in a generated provenance artifact.
