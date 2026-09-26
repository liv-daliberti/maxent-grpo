# E109-R1-S2 completion-backfill scheduler amendment

Frozen: 2026-08-25 at 17:31 EDT while continuation jobs `30874012` and
`30874013` remain pending at zero runtime and before reading either E109 or
E112 evaluation endpoint. PointMaze remains excluded.

## Trigger and outcome-blind diagnosis

The two registered Qwen2.5-3B Python ReplayDr comparator continuations have
remained eligible on the non-preempting `all` partition since 00:01 EDT, but
the requested A6000 pool `node[104,205-207,805]` is GPU-saturated. Slurm
projects seed 73 on node104 at 03:40 EDT on August 26 and gives seed 74 no
projected start. Both jobs retain `RunTime=00:00:00`, `Restarts=0`, and their
exact continuation exports and run directories.

The earlier registered E109 A6000 pool also contains node103. Contemporaneous
read-only inspection shows node103 healthy in partition `all`, exposing ten
A6000 GPUs and sufficient CPU and memory. It is currently GPU-saturated by
shorter work, so restoring it to the eligible pool creates an additional
backfill path as those allocations end without changing accelerator class.

The latest independently validated checkpoints remain step 2,304 for seed 73
and step 2,112 for seed 74, leaving at most 768 and 960 of the 3,072 registered
optimizer steps after resume. The completed prefixes reached training steps
2,391 and 2,197 in scheduler records totaling roughly thirteen hours apiece.
A twelve-hour continuation reservation is therefore more than twice the
training time implied by the completed prefixes for the remaining checkpoint
horizon, while preserving room for startup and terminal evaluation. The
existing 192-step checkpoint cadence and automatic-resume behavior remain
active if infrastructure overhead is unexpectedly larger.

## Authorized scheduler-only transaction

For exact jobs `30874012` and `30874013`, while both remain user-held at zero
runtime:

- widen `ReqNodeList` from `node[104,205-207,805]` to the previously
  registered healthy pool `node[103-104,205-207,805]`;
- reduce only the Slurm `TimeLimit` from `3-00:00:00` to `12:00:00`; and
- release both jobs together only after a complete held-state audit.

Retain partition `all`, account `allcs`, `Nice=100`, one A6000, 16 CPUs,
128 GiB, job IDs, seeds, output paths, model and optimizer checkpoints,
scientific environment, source snapshots, data, parser, replay objective,
evaluation surface, stopping rule, and all checkpoint/resume settings.

The application must compare each live scientific export hash to the frozen
E109-R1 continuation artifact, revalidate the selected checkpoint, record the
before/held/after scheduler rows and node inventory, and fail closed. On any
failure it restores the 72-hour limit and prior node pool, releases the
transactional holds, and removes only its incomplete provenance artifact. It
does not inspect an endpoint or delete, archive, or rewrite any run artifact.

## Consequence

This amendment changes scheduler backfill eligibility only. Each cell still
must reach 3,072 optimizer steps and write its own terminal completion receipt
before E109 can move from 13/15 to 15/15 or feed the complete paired analysis.
