# E129 amendment 1 — the comparator stays E78, and what that costs — September 19, 2026

Amends `e129_drgrpo_reference_kl_05b_20260918.md`, which is registered,
submitted and released. The 75 cells were released earlier today; at the time
of writing the lead cells are near step 220 of 3,072 and no endpoint, PMD value
or paired difference has been read, so no result influenced the choice recorded
here.

The original document says each cell "reproduces the E78 control configuration
exactly". That is true of the objective, the data, the seeds, the schedule and
the placement. It is not true of the runtime, and this amendment records the
difference, the reasoning that was applied to it, and the decision taken.

## What the comparison inherits

`ops/exp_scaling/cohorts.py` states, above the E126/E127/E128 block, that
reading a new cohort against the completed E78 control "would put placement,
runtime and evaluator terms inside a reported method effect", which is why
those three cohorts carry their own matched control rather than differencing
against E78. E129 differences against E78. Of the three terms:

- **Placement is matched, cell by cell.** Every E129 cell inherits
  `source_node` from the same frozen E72 manifest E78 read, and the launcher
  refuses to submit a cell whose node differs from that of its matched control.
  This is the term that forced E128 into existence: E126 and E127 run on cs
  A5000 rather than mltheory node302/node105. E129 does not have that problem.
- **The evaluator is matched at the level of the reported metric.** The
  manuscript's Level-1 PMD is the 32-disjoint-stream resample recorded in
  `paper/results/mode_diversity_terminal_resampled_partial.json`, not the
  registered draws, and that resample covers the `drgrpo` and `replay_drgrpo`
  arms at exactly the 16 a100 / 9 a5000 split E129 reproduces.
- **The runtime is not matched.** E78's source snapshot
  `e76_tuned_scale_96f68ebb47757af8` was destroyed by the irreversible
  2026-09-04 cleanup and is not reproducible from any commit, so E129 runs
  current source under `e76_tuned_scale_73e35a064068bdac`.

## The decision

No matched zero-coefficient control arm will be run. The comparator remains the
completed E78 control at the same domains, seeds and nodes.

The evidence bearing on the unmatched term is that the two runtimes differ by
205 inserted lines and no deleted ones, all of them additive comparator modules
reached only when `gapo_enabled` or `setpo_coefficient` is set, and none of them
touching any line of the `beta`, `k3`, `reg_loss` or `ref_model` path this
cohort exercises. That is an argument from inspection of a diff rather than
from a control that absorbs the term, and it is weaker than what E128 supplies
for E126 and E127. It is recorded here so that the asymmetry is visible in the
record rather than discovered in review, and so that any reported E129 effect
is read with the runtime term named rather than assumed away.

Nothing else in the registered design changes: the arms, coefficients, domains,
seeds, schedule, outcomes, estimands and failure policy all stand as written.

## Allocation change, September 19, after release

The 75 cells were submitted at `--mem=80G`, padded above E78's 64G for the
resident reference policy. Measured on the running cells at step ~1,300,
`sstat` reports MaxRSS 29.4 GB and AveRSS 20.7 GB, so the request was roughly
2.7 times the observed peak and was stranding memory on a node whose GPUs were
otherwise idle: node105 held six cells at 80G against 515G, leaving four of its
ten A5000s unusable.

The 69 cells still pending at that time were therefore updated to
`MinMemoryNode=48G`, which keeps 63 percent headroom over the observed peak.
The six cells already running keep the 80G they were allocated, since a running
job's allocation cannot be changed; node105 converges to ten concurrent cells
as they retire. One node302 cell started immediately in the memory E124 leaves
free, which at 80G it could not have entered.

This is a scheduling parameter and not a scientific one: no objective, data,
seed, schedule, coefficient or placement changed, and the per-cell coefficient
audit recorded in the ledger at submission stands. It is recorded because the
ledger's stored `command` and `held_scheduler_record` for those 69 cells read
80G, which is no longer what they will run under.
