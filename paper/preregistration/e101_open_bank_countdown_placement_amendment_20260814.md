# E101 zero-step A6000 placement amendment

Recorded: 2026-08-14, before any E101 allocation or metric materialization.

Jobs 30579498, 30579499, and 30579500 were submitted to node103/node104
with 64 GB host memory and remained pending at step zero. Both nodes had only
23,574 MB of unallocated scheduler memory despite seven free A6000s, so none of
the registered 64 GB cells was eligible to backfill.

The eligible node list is broadened to node103, node104, node205, and node206.
Every node exposes the same A6000 GPU type. Requested host memory is reduced to
48 GB. Three prior completed 0.5B Countdown cells (30263960, 30263962,
30263964) used at most 27.75 GB resident host memory with an evaluation batch
of 64; E101 uses an evaluation batch of 32, leaving more than 20 GB headroom.

The model, source snapshot, data, seed, arm objectives, coefficients, decoding,
evaluation schedule, 55-minute allocation cap, and all other resources remain
unchanged. This is a scheduler-eligibility repair only. Original held records
remain in the ledger; post-amendment records and this file's digest are appended
before the jobs are released.
