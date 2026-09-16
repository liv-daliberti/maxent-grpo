# E117-R1-S6 zero-step capacity repair

Frozen: 2026-08-29T18:45:38-04:00 while the nine affected E117-R1 jobs
remain pending at zero runtime and before any affected run directory or
endpoint exists.

Status: scheduler-only continuity amendment. PointMaze remains excluded.

## Trigger

The S5 user-hold release succeeded on 2026-08-28, but none of the nine
released Qwen jobs started. Live Slurm inspection now reports
`ReqNodeNotAvail, May be reserved for other job` for all nine:

- Countdown C/P/F jobs `30873695`--`30873697` remain pinned to `node103`,
  which entered `MIXED+DRAIN` because six GPUs are overheated;
- Graph C/P/F jobs `30873698`--`30873700` remain pinned to fully allocated
  `node104`, with projected serial starts on 2026-08-31; and
- Python C/P/F jobs `30873701`--`30873703` remain pinned to fully allocated
  `node101`, with projected serial starts on 2026-08-31.

The Falcon MathIR C/P/F block is already terminal and is not modified. The
replacement audit job `30874713` remains dependency-blocked on the nine
nonterminal scientific jobs.

## Outcome-blind placement

Contemporaneous scheduler inspection found two healthy adjacent A5000 nodes
in the same `all` pool with enough GPU, CPU, and memory headroom for the exact
whole-block placement below. An initial preflight rejected `node204` before
any job was touched when another array reduced its free GPUs from four to two.
Select the surviving nodes only from health and free scheduler capacity,
without inspecting an affected endpoint:

- Countdown C/P/F: `node103` to `node202`;
- Graph C/P/F: `node104` to `node203`; and
- Python C/P/F: `node101` to `node203`.

Hardware remains a block variable: every C/P/F member for a sentinel is pinned
to the same exact physical node and accelerator type. The A5000 is an
Ampere-class accelerator supporting the frozen bfloat16/vLLM path already used
by the terminal Falcon MathIR block on `node203`.

## Authorized transaction

For exactly the nine job IDs above, verify `PENDING`, zero runtime, zero
restarts, absent run directories, the frozen scientific export hashes,
`Partition=all`, `Account=mltheory`, one GPU, eight CPUs, 64 GiB, and a six-hour
limit. Verify the two target nodes are healthy, expose ten A5000 GPUs, are in
the `all` partition, and have contemporaneous aggregate capacity for the three
jobs assigned to `node202` and six jobs assigned to `node203`.

User-hold the nine jobs as one transaction, revalidate the zero-step boundary,
change only each complete block's `ReqNodeList`, write the provenance record,
and release all nine together. Preserve job IDs, priorities, accrued scientific
identity, audit dependencies, source snapshots, run paths, arms, seed, data,
optimizer, request streams, evaluation surface, and stopping rules. On any
pre-release failure restore the three prior node lists and release all holds.

After release, update the effective-node map consumed by the frozen terminal
audit and verify that every affected job is released on its registered target.
No E117 endpoint may be used to choose or justify this repair.
