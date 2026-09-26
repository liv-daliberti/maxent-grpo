# E104/E106 amendment: one-hour Falcon mechanism backfill

**Frozen on 2026-08-17 before any target job allocated, while all four targets
were `PENDING` at runtime `00:00:00`, and without inspecting any post-update
E104/E106 evaluation outcome.**

The completed Falcon3-1B E104 Countdown cell, job `30637791`, ran the identical
64-update model, optimizer/offload, group-size, v6 semantic, verified-replay,
checkpoint, and initial-evaluation envelope in `00:20:06`. The four remaining
Falcon mechanism cells still request two hours:

- E104 Graph `30637790`;
- E104 MathIR `30637793`;
- E104 PantryPlan `30637794`; and
- E106 repaired Python `30640330`.

Each target may have only its Slurm time limit changed from `02:00:00` to
`01:00:00`, giving just under three times the observed completed Falcon
runtime. Before any change, every target must remain pending at zero runtime,
retain its exact ledger-bound source snapshot and scientific environment, and
request the registered node/GPU placement. After the change, the same checks
are repeated with the one-hour horizon. A partial or invalid change rolls all
changed jobs back to two hours.

This changes no model, snapshot, data, domain, prompt, parser, seed, optimizer,
decoding, group size, objective, coefficient, stopping rule, checkpoint,
evaluation cadence, output path, partition, account, node constraint, GPU,
CPU, or memory. Qwen-3B remains at two hours. The amendment reads scheduler
records and the completed reference elapsed time only, never evaluation
values. PointMaze remains excluded.
