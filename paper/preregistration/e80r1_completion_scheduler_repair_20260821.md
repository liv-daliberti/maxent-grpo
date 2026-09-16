# E80-R1 completion scheduler repair

Frozen on 2026-08-21 EDT before application. This amendment uses the released
E80-R1 and E87 ledgers, scheduler state, node inventory, update/checkpoint
progress, and prior operational records only. It does not inspect an evaluation
outcome or change the E80-R1 estimand. PointMaze is not part of E80-R1.

## Trigger and diagnosis

E80-R1 has 31 of 50 terminal cells and 19 unfinished cells. All 19 are
`PENDING` at scheduler runtime `00:00:00`:

- Python Factors replay seed 73 and control/replay seed 74 (jobs `30277399`--
  `30277401`) have never started. They retain the original
  `mltheory`/`node302`/A100 placement and report that node302 may be reserved
  for another job.
- MathIR control/replay seeds 71--74 (jobs `30277404`--`30277411`) and
  PantryPlan control/replay seeds 71--74 (jobs `30277414`--`30277421`) use the
  matched A6000 placement registered for E105. All 16 are in preemptible
  `lowprio`, all name `node805`, which is currently `DOWN`, and 15 report
  `UnavailableNodes:node805`. Each has already been requeued one or two times.
  Their recorded progress ranges from 0 to 892 updates; seven have a durable
  checkpoint at step 384 or 768 and retain auto-resume. No checkpoint or run
  directory is removed by this amendment.

All 19 unfinished jobs have `Nice=500`. This is not the registered E80-R1
priority: E80-R1 was submitted at `Nice=100`. The campaign log records that the
34 cells pending on 2026-08-09 were temporarily demoted to `Nice=500` solely so
the five-cell E87 directional probe could run first, and explicitly records
`Nice=100` as the reversible setting. E87 is now 5/5 terminal, so that temporary
priority intervention has completed its purpose.

The live inventory exposes A6000s with at least 128 GiB host memory in the
non-preempting `all` partition on `node[103-104,205-208]`. Account `mltheory`
is allowed in `all`; `PreemptMode=OFF`. Node805 is deliberately excluded.

## Authorized scheduler-only change

While each target remains pending or starts during the transaction with its
exact frozen scientific environment:

- restore `Nice=100` on all 19 unfinished E80-R1 jobs;
- for the 16 matched MathIR/PantryPlan jobs, change partition `lowprio` to
  `all` and change the A6000 node pool from
  `node[103-104,205-208,805]` to `node[103-104,205-208]`; and
- leave the three Python Factors jobs on `mltheory`, `node302`, and one A100.

Both arms of every affected unfinished MathIR/PantryPlan seed move together.
The Python seed-73 control has already completed on the original A100, so its
unfinished replay mate is intentionally kept on that same hardware class.
Python seed 74 likewise keeps both pending arms on A100.

Retain account `mltheory`, one GPU, 16 CPUs, 128 GiB, the three-day time limit,
all source and ops snapshots, model revision, domain, seed, prompts, optimizer,
learning rate, group size, replay objective and coefficient, evaluation and
checkpoint cadence, 3,072-update stopping rule, output path, requeue setting,
and auto-resume policy. No job is cancelled, resubmitted, released, or held.

## Application and evidence boundary

`ops/exp_scaling/apply_e80r1_completion_scheduler_repair_20260821.py` must
validate E87 completion, the exact 19-cell ledger identity, the byte-identical
live and frozen `--export` environments, current placement and resources, the
live node inventory, and the prior paired A6000 amendment before mutation. It
must update each job in one scheduler command, postflight every job, and make a
best-effort rollback of any still-pending changed job if the transaction fails.

The content-addressed artifact
`var/artifacts/e80r1_completion_scheduler_repair_20260821.json` records the
protocol, script, ledgers, prior placement amendment, inventory, progress and
checkpoint positions, and full before/after scheduler records. Training and
evaluation files are read only for non-outcome completion/progress accounting;
no endpoint or treatment effect is inspected.

This amendment only restores the registered priority and removes scheduler
constraints that have become operationally pathological. A cell remains
nonterminal until it reaches its registered 3,072 updates and writes the
ordinary completion evidence. Partial progress after this amendment is not a
reportable endpoint.
