# E115 Qwen-0.5B amendment: repair the routed-away node105 cells

Frozen on 2026-08-21 EDT before application, while all four target jobs are
`PENDING` at `RunTime=00:00:00` with `Reason=BadConstraints` and absent run
directories. This amendment uses scheduler state, partition membership, and
the frozen E78 parent and E115 ledgers only. No E115 outcome endpoint was
inspected. PointMaze is excluded.

## Trigger and diagnosis

The four nonterminal E115 Qwen2.5-0.5B UCPO cells --- Countdown seeds 46 and
47 (`30790285`, `30790286`) and MathIR seeds 46 and 47 (`30790290`,
`30790291`) --- have been `Priority=0`, `Reason=BadConstraints` since
`SubmitTime=2026-08-19T15:53:47`, with `Restarts=0` and `RunTime=00:00:00`.

`direct_comparator_completion.clone_command` reproduces the paired parent's
`SubmitLine` verbatim, so these four inherited E78's
`--partition=all --account=allcs --nodelist=node105 --gres=gpu:a5000:1
--time=1-12:00:00`. Submit-side routing sends any request longer than one hour
to `cs`, and `cs` contains only `node[202-207]`. The required node `node105` is
therefore not a member of the job's partition, which is unsatisfiable rather
than merely busy, so the four cells can never be scheduled where they stand.

This is the same submit-time routing failure already registered in
`e111_scheduler_partition_amendment_20260818.md`, and the same one their own
parents hit: E78 `30263965`--`30263968` and `30263985`--`30263988` were also
recorded at submission as `Partition=cs` and ultimately ran on `node105` in
partition `mltheory` under account `mltheory`. The three completed E115
Qwen-0.5B sibling seeds 43, 44, and 45 likewise ran in `mltheory`.

## Authorized scheduler-only change

While each job remains pending at zero runtime with its exact frozen
scientific environment, change for jobs `30790285`, `30790286`, `30790290`,
and `30790291`:

- partition `cs` to `mltheory`; and
- account `allcs` to `mltheory`, which partition `mltheory` requires.

Retain `node105`, one A5000, 8 CPUs, 64 GiB, the 36-hour limit, `Nice=100`,
and every element of the frozen scientific environment: source and ops
snapshot, model and revision, UCPO variant and `tau=0.2`, domain, seed, data,
prompts, optimizer, group size, evaluation and checkpoint cadence, target of
3,072 steps, and output path. The application script must verify that the full
97-key `--export` environment of each target is byte-identical to the
environment frozen in
`var/artifacts/e115_ucpo_qwen05b_domain_extension_jobs.json`.

This restores the placement of the paired E78 control cell for the same
domain and seed --- `node105`, A5000, partition `mltheory` --- so the
hardware class of the UCPO cell and its Dr.GRPO comparator continue to match
cell by cell.

The application is transactional: if any update or the post-update audit
fails, every job changed by that invocation is restored to partition `cs` and
account `allcs`. A content-addressed artifact records the protocol, script,
ledger, partition membership, and before/after scheduler records.

## Gate consequence

This amendment changes placement eligibility only. It creates no endpoint and
authorizes no outcome inspection. Each cell must still reach 3,072 optimizer
steps and write its own completion receipt to count toward the ten-cell E115
Qwen-0.5B UCPO extension.
