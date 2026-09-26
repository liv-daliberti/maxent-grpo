# E125 amendment 1 — pinned two-arm population — September 17, 2026

Amends `e125_decoding_frontier_current_arms_20260914.md`, which is registered
and unsubmitted. Nothing had been measured when this amendment was written, so
no result influenced any choice recorded here.

Two things changed, and they are connected: a stale hardware fact in the
original document made the design look harder than it is, and correcting it
removes the compromise the original registered.

## The hardware constraint was overstated

The original document reported that this cohort trained across five GPU models,
that node105 was **down**, and that the a5000 pool held 57 Qwen-0.5B terminal
cells. It concluded that no verified pool held a complete design and registered
an unpinned sweep with a measured cross-model term and a decision rule.

Re-deriving the inventory from Slurm accounting (`sacct -X --format=NodeList`
over the 100 terminal job ids, cross-checked against the `source_node` each
ledger already records — the two agree cell for cell) gives:

| GPU model | nodes | Qwen-0.5B terminal cells |
| --- | --- | --- |
| a100 | node302 | 39 |
| a5000 | node105 | 32 |
| rtx_3090 | node020, node022, node025, node026 | 17 |
| a6000 | node205, node206, node207 | 7 |
| a40 | node101 | 5 |

The five-model spread is real and the a100 and rtx_3090 counts stand. The a5000
count was wrong — 32, not 57; the original table sums to 125 against a
population of 100. And node105 is **not** down: it is up and accepting work in
the `mltheory`, `all` and `lowprio` partitions.

The decisive fact the original missed is that the spread is not uniform across
arms. Splitting the population by experiment:

* the **E78 arms** (`control` = Dr.GRPO, `replay` = Re:Dr) trained on exactly
  two hosts, node302 (32 cells) and node105 (18 cells), and **all 25
  domain/seed pairs have both arms on the same host**;
* the **E118 arms** (`maxrl`, `replay_maxrl`) carry the whole five-model spread,
  and **13 of 25 pairs are split across GPU models**.

The constraint the original document described is entirely a property of the
E118 half.

## What changes

**Population.** E125 runs the E78 arms only: `control` and `replay`, five
domains, seeds 43–47, 50 terminal checkpoints at step 3,073 (the export of
target step 3,072). The E118 MaxRL arms are removed from this experiment. They
are not abandoned; they are a separate question that needs the cross-model term
the original registered, and mixing a clean design with a compromised one inside
one table would make the clean half unreadable.

**Pinning replaces the cross-model measurement.** Every cell is pinned to the
node that trained its checkpoint, which is the E72 discipline
(`launch_e72_decoding_frontier.resolve_nodelist`) and needs no widening: both
node302 and node105 are up, and no cell requires a pool it does not have. The
hardware control, the |ΔPMD| distribution, and the decision rule that would have
interpreted it are therefore **withdrawn as unnecessary** rather than waived.
Since both arms of every pair sit on one GPU model, no paired contrast contains
a hardware difference, and the sweep inherits exactly the hardware provenance
the paper's own published numbers already have.

If any cell later proves unrunnable on its training node, it is reported as a
gap. It is not moved to another GPU model.

**Endpoint and gate are unchanged.** The primary endpoint remains frontier
dominance, with the correctness-matched secondary reading; PMD, pooling and the
support bar are as registered. The reproduction gate is unchanged and is run
first: the T=1.0, top-p 1.0, K=8, four-draw column must reproduce the published
terminal `pass@8`, `mean@8`, `distinct@8` and greedy trace for the same
checkpoints, within `aggregate_e72_frontier.py`'s standing tolerance. The gate
column is submitted and checked before the remaining temperatures are queued.

**The frozen-base reference row is deferred.** The original included a
frozen-model row at every temperature and domain. A base cell's published
counterpart is a pass-0 row, and pass-0 values on this cohort cluster by GPU
pool, so a single base row cannot serve checkpoints pinned to two different
hosts without being measured on both. It is deferred to a follow-up rather than
run halfway; the decoding objection is about what a *trained* policy can be made
to emit, and the appendix table it feeds carries no base row.

**The sweep reproduces the published draw seeding.** The gate probe was run
first and failed on the replay arm, and the cause is not decoding. Every
terminal evaluation this project has published seeded a prompt's four draws at
`base + draw_index`; under vLLM 0.8.4 V0 an `n=K` request with seed `s` expands
to children `s..s+K-1`, so consecutive draws shared child streams. Live source
now strides by K (`eval_mode_coverage_disjoint_draws`, default true, added
2026-09-16), which is correct for a fresh run and changes which samples are
drawn for a re-measurement. Draw 0 is unaffected, which is why the collapsed
control and the greedy trace still reproduced while the diverse replay arm's
set statistics moved.

E125 therefore runs with `OAT_ZERO_EVAL_MODE_COVERAGE_DISJOINT_DRAWS=0`, so the
grid is anchored to the numbers the paper prints and the only thing varying
across it is decoding. Sweeping under the corrected scheme would change the
samples at every temperature including T=1, which would answer a different
question than the one registered. The correction is out of scope here; what it
does to the published endpoints is a separate matter recorded below.

**Measured, in passing:** re-measuring Graph/s43 at T=1 under the corrected
seeding gives `distinct@8` 0.3594 → 0.3535 on the control (−1.6%) and
2.4355 → 2.2539 on the replay arm (−7.5%), with `pass@8` 0.9805 → 0.9570 on the
replay arm and unchanged on the control. Two cells are not an estimate of
anything, and this is recorded only because the probe produced it.

**Stages B and C remain unregistered** and are not submitted on the strength of
this document.

## Revised cost

| item | quantity |
| --- | --- |
| evaluation cells | 300 (50 checkpoints × 6 temperatures) |
| generations | 1,228,800 (300 × 128 prompts × 32 samples) |
| compute | ≈ 10 GPU-hours at the measured 1.9 min/cell |
| checkpoint restore | 46.0 GiB, of which 16.6 GiB is the node105 subset |

## Submission order

The node105 subset (18 checkpoints, 108 cells) is restored and submitted first;
the node302 subset (32 checkpoints, 192 cells) is held until E124 clears that
host, since node302 is where its a100 work is queued. This is a placement and
sequencing choice only: the two subsets are measured identically and the
analysis does not distinguish them.

Restores stage through `HF_HUB_CACHE` on the project share. The default cache
is under `$HOME`, which is a 5 GiB quota on this account and cannot hold even
one checkpoint's staging copy.

## Status

**Superseded in part by amendment 2 (2026-09-18)**, which records why the
node302 subset could not be measured on its training hardware and what the
resulting uniform-hardware placement costs. The node105 result below stands
unchanged.

Amended; the node105 subset is complete and the node302 subset is held.

The 108 node105 cells (18 checkpoints × 6 temperatures) finished on 2026-09-17.
The reproduction gate passes 72/72 checks with a maximum absolute deviation of
zero: every T=1.0 cell reproduces its published `pass@8`, `mean@8`,
`distinct@8` and greedy trace exactly. One cell (PantryPlan/control/s44 at
T=1.0) was killed by the stale-progress watchdog after an hour without metrics,
with no traceback and no memory error, while seven cells shared the host; it was
resubmitted once and completed in 2:05. No other cell failed.

On this subset the registered primary endpoint holds in every domain: no
control temperature reaches its paired replay arm's best PMD.

| domain | control PMD over the grid | control best | replay best |
| --- | --- | --- | --- |
| Graph | .000–.074 | .074 (T=2.0) | .561 (T=1.3) |
| Countdown | .000–.015 | .015 (T=1.6) | .511 (T=1.3) |
| Python | never clears the support bar | — | .013 (T=1.6) |
| MathIR | .000–.002 | .002 (T=1.6) | .026 (T=1.6) |
| PantryPlan | .000–.000 | .000 | .614 (T=2.0) |

This is 18 of 50 checkpoints, one or two seeds per domain and arm and a single
seed in Graph. It is not a five-seed result and is not reportable as one; the
node302 subset is what completes the design. Two readings are worth recording
now because they differ from the E72 grid the appendix currently prints:
PantryPlan's control is flat at .000 across all six temperatures, where the
older cohort's control reached .069 there and that was the largest value
decoding bought it anywhere; and Python's *replay* arm reads PMD ≈ .000 at
`pass@8` ≈ .999, against .440 on the older cohort. Graph at T=2.0 is the only
place decoding buys the control anything, and `pass@8` is already falling there.
