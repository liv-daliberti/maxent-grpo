# E128 preregistration: matched Dr.GRPO control for E126 and E127

Date frozen: 2026-09-18, before submission of any E126, E127, or E128 cell.

## Why this cohort exists

E97 (UCPO) and E115 could difference their comparator against the completed E78
Dr.GRPO control and state honestly that only the objective differed: the cells
inherited E78's node and E78's immutable runtime snapshot. E126 (GAPO) and E127
(SetPO) can do neither.

**The runtime is gone.** E78's ledger pins
`e76_tuned_scale_96f68ebb47757af8`. The user-approved cleanup recorded in
`var/artifacts/cleanup_20260904_retired_e105_old_logs_unused_source_snapshots.json`
retired 528 source snapshots "not referenced by live queue" on 2026-09-04 and
is marked irreversible. That set included the E78 base runtime and both E97
runtimes. No commit in the repository's 248-commit history reproduces the tree,
so it was built from a working tree that was never committed and cannot be
rebuilt. A new arm cannot inherit it, and the size of the difference between it
and any present-day runtime cannot even be measured.

**The placement differs.** E126 and E127 run on the `cs` partition's A5000
nodes. The 50 E78 cells are pinned to mltheory node302 (36 cells, A100) and
node105 (14 cells, A5000). This campaign has already observed that terminal
values cluster by GPU pool.

**The evaluator differs.** The E78 record was measured before
`eval_mode_coverage_disjoint_draws`, under which a prompt's consecutive
evaluation draws share vLLM child streams and pooled mode-coverage statistics
count repeats as independent. E126, E127, and E128 all take the corrected
evaluator.

Differenced against E78, a GAPO or SetPO effect would therefore carry a
GPU-model term, a runtime term that cannot be bounded, and an evaluator
difference on the breadth metric the comparison is about. E128 removes all
three by re-running the control arm under the same conditions as the arms it
serves.

## Question and estimand

E128 is a control cohort and carries no hypothesis of its own. Its estimand is
the terminal-pass value of ordinary Dr.GRPO within domain and seed, measured
under the E126/E127 runtime, placement, and evaluator, so that the GAPO and
SetPO effects can be formed as paired within-set differences.

A secondary, descriptive quantity is recorded but not tested: the difference
between each E128 cell and its E78 counterpart at the same domain and seed.
That is a combined measure of the runtime, placement, and evaluator change, not
an effect of any objective, and it must never be reported as one. It is worth
recording because it is the only available evidence about how much those three
changes move this panel, and the paper should be able to say so rather than
assert the change is small.

## Frozen cohort

- Model and initialization: Qwen2.5-0.5B-Instruct, exactly as E78.
- Objective: `e78.fixed_objective("control")`, unmodified. The launcher asserts
  an empty override set, so "this is the control" is machine-checked rather
  than restated.
- Domains: Graph Coloring, Countdown, PythonFactors, MathIR, PantryPlan.
- Seeds: 43, 44, 45, 46, 47 (25 scientific cells).
- Training: 384 prompts, 8 passes, 16 rollouts per prompt, learning rate 2e-7,
  beta=0, one PPO epoch, and the E78 decoding/evaluation schedule.
- Runtime: the snapshot shared with E126 and E127, derived from
  `e76_tuned_scale_970e16dc21f47834` --- the one surviving `e76_tuned_scale`
  snapshot that still hashes to its own recorded identity, and the runtime E121
  ran on --- with the declared objective and evaluator files patched. All three
  ledgers must record the same `snapshot_root`; a mismatch is a submission
  failure.
- Placement: `cs` partition, `allcs` account, A5000, node203 or node204.
  Partition and account are audited from the scheduler's own record before
  release.

No domain-specific setting, seed exclusion, early stopping choice, or post-hoc
replacement is permitted. A separate 32-query Graph/s43 learner smoke is
operational only and is excluded from every estimate. All scientific cells
depend on that smoke completing successfully.

## Outcomes and reporting

Report the same terminal accuracy and verified breadth metrics as the arms it
serves, on all five seeds, at the registered K=8 evaluation.

When the GAPO and SetPO rows appear in the alternatives table, their control is
E128 and the other rows' control is E78. The table must say so. A reader who
assumes one common control would otherwise read a runtime and placement change
as part of a method effect, which is exactly the error this cohort exists to
prevent.

## Failure rules

The immutable runtime must be byte-identical to the one E126 and E127 use.
Submission is held until the snapshot and scheduler exports are audited. A
failed smoke blocks all 25 scientific jobs. A scientific cell may be retried
only for an infrastructure failure under the same frozen configuration.

If E128 cannot be completed, E126 and E127 are not reported against E78 as a
substitute. They are reported as uncontrolled, or not reported.
