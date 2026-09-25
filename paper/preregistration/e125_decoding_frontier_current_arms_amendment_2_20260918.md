# E125 amendment 2 — uniform evaluation hardware — September 18, 2026

Amends `e125_decoding_frontier_current_arms_20260914.md` and its first
amendment. It is written after the grid completed, so unlike amendment 1 it is
not blind to results; what it changes is placement, and the reason is a
scheduler constraint recorded below, not anything the numbers showed.

## What changed

Amendment 1 pinned every cell to the node that trained its checkpoint. The
node105 half ran that way and is untouched. The node302 half did not: the
cluster's only a100 host is node302, and a concurrent campaign held 496 GiB of
its 503 GiB behind three-day time limits, leaving 7 GiB against a cell that
peaks at 12.4 GiB. Twenty-eight of its 192 cells completed in nineteen hours;
the remaining 164 could not start at all, on any partition, because no partition
adds memory to a full node.

The grid is therefore measured on **one GPU model throughout** (a5000), rather
than each checkpoint on the model that trained it. The 28 cells already measured
on a100 were moved to `var/data/e125_frontier_a100_pinned` and are not read by
the analysis: a checkpoint's temperature curve must not span two GPU models, and
leaving them in place would have done exactly that for fourteen checkpoints.

This is a real loss and a real gain, and both should be stated. Lost: 32 of 50
checkpoints no longer reproduce their published terminal endpoints exactly, so
the reproduction gate covers 18 rather than 50. Gained: the sweep is internally
uniform, where the pinned design had the grid spanning two GPU models, so no
comparison **within** this table now carries a hardware difference.

## What the deviation costs, measured twice

The original preregistration anticipated unpinned running and set the rule:
interpret the sweep as evidence only if the cross-model term is small relative
to the replay-minus-control gap. Two independent measurements bound it.

**The hardware control.** Twenty a100-trained checkpoints re-measured at T=1 on
a5000, against their published a100 values, in a separate root
(`var/data/e125_frontier_hwcheck`) that the grid's readers do not look at: mean
|ΔPMD| = .0032 over twelve paired comparisons, maximum .0131, with `pass@8`
differing by at most .006. The replay-minus-control gap is ~.5, so the term is
roughly fortyfold smaller.

**The gate itself, split by hardware.** Of 200 metric checks, the 72 on
checkpoints whose training model matches the evaluation model pass exactly, to
the last representable unit. The 128 measured across models hold 125 inside the
same tolerance. The three exceptions are `greedy` (twice) and `mean` (once) ---
the statistics most sensitive to floating-point reduction order, greedy being
deterministic given hardware. No `distinct@8` or `pass@8` check fails.

Both readings agree that changing GPU model moves these measurements by
thousandths where the effect under study is halves.

## What is unchanged

The endpoint, the estimator, the support bar, the pooling, the domains, the
seeds, the temperatures and the draw seeding
(`eval_mode_coverage_disjoint_draws=0`) are all as registered. Stages B and C
remain unregistered and unrun.

## Result

All 300 cells completed: 50 checkpoints, five seeds per domain and arm, six
temperatures. The registered primary endpoint holds in every domain --- no
control temperature reaches its paired replay arm's breadth.

| domain | control \pmd{} over the sweep | control best | Re:Dr at T=1 |
| --- | --- | --- | --- |
| Graph | .000–.021 | .021 | .562 |
| Countdown | .001–.016 | .016 | .485 |
| PantryPlan | .000–.004 | .004 | .330 |
| MathIR | .001–.002 | .002 | .020 |
| Python | never clears the support bar | — | .000 |

One cell (Graph/replay/s47 at T=2.0) failed on first attempt when a per-job
source snapshot copied a `__pycache__` file that another concurrently starting
job was rewriting. It was resubmitted once and completed. No other cell failed.
