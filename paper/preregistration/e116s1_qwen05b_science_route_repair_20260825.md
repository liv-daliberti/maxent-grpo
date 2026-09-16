# E116-S1 Qwen-0.5B scientific-cell route repair

Frozen: 2026-08-25 EDT before changing any E116 scientific job and after the
author explicitly requested completion of the four remaining Qwen2.5-0.5B
sparse RLEP-Dr cells.

## Trigger

E116 jobs `30790408`, `30790409`, `30790413`, and `30790414` are the frozen
Countdown and MathIR seed-46/47 scientific cells.  They remain `PENDING` at
zero runtime with no run directory after every cell-specific pool audit and the
Qwen2.5-0.5B smoke audit completed successfully.  The submit router placed the
jobs in partition/account `cs/allcs` while retaining the frozen
`ReqNodeList=node105` and `gres/gpu:a5000:1` request.  Node105 is exposed by
partition `mltheory`, not `cs`; this is the same routing mismatch repaired for
the four corresponding E116 pool jobs by the frozen 2026-08-21 scheduler
acceleration package.

## Authorized repair

While each target remains zero-runtime and nonterminal, change only:

- partition `cs` to `mltheory`; and
- account `allcs` to `mltheory`.

Retain node105, one A5000, 8 CPUs, 64 GiB, the 36-hour limit, `Nice=100`, model,
domain, seed, optimizer, sparse RLEP-Dr treatment, pool root, prompt and data
order, stopping rule, checkpoint policy, source/ops snapshots, output directory,
and every exported environment variable.  Do not submit a replacement cell,
change a dependency, inspect an endpoint, or touch an existing run directory.

## Application and evidence

`ops/exp_scaling/apply_e116s1_qwen05b_science_route_repair.py` must fail closed
unless the immutable E116 ledger identifies exactly these four cells, their
live environments equal their frozen held environments, all five prerequisite
audits are successful, every target is pending at zero runtime, the four output
directories are absent, and node105 still exposes A5000 GPUs in `mltheory`.
It records the before/after scheduler records and source digests in
`var/artifacts/e116s1_qwen05b_science_route_repair.json`.

This is a scheduler-only placement repair.  It changes no scientific
environment, estimand, evidence rule, or terminal endpoint requirement.
