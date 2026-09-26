# E125 decoding frontier for the current arms — September 14, 2026

## Why this exists

A reader will ask whether verified replay buys anything that turning up the
sampling temperature would not. The manuscript cannot currently answer it. The
registered evaluation contract draws every trained checkpoint at temperature 1
with top-$p$ 1, plus one greedy trace, so no trained arm has ever been measured
across a decoding range.

`paper/figures/e72_decoding_frontier.pdf` looks like the answer and is not. Its
own record says so:

```
scope:  terminal checkpoints from a superseded 12-pass multi-component
        treatment; the sweep tests decoding of frozen policies only
status: complete five-seed historical control; not x-Mode evidence
```

Those checkpoints come from a 12-pass multi-component arm that the paper no
longer reports, and its y-axis is `distinct@8`, which the manuscript now treats
as a registered endpoint rather than a breadth axis. Presenting it beside the
current 8-pass Re:Dr.GRPO and Re:MaxRL results would show superseded data as
current evidence. This experiment measures the current arms instead.

## Question and primary endpoint

> Can the control reach the replay arm's verified-mode diversity by decoding
> differently?

For each arm we trace the curve of (`pass@8`, `PMD`) across temperature. The
**primary endpoint is frontier dominance**: whether there exists any swept
temperature at which a control arm attains `PMD` at least as high as its paired
replay arm attains at *its* best temperature, holding domain, seed, and fresh
objective fixed. The claim under test is the negative one, that no such
temperature exists; a single control temperature that matches replay falsifies
the paper's implicit assumption that replay is not a decoding effect in
disguise.

A secondary reading matches on correctness: at the control temperature whose
`pass@8` is closest to the replay arm's temperature-1 `pass@8`, is the control's
`PMD` still lower? This is the comparison a reviewer will actually picture, and
it is reported for every domain and seed pair.

`PMD` is the estimator already registered in `ops/mode_diversity.py`, pooled
over a prompt's draws, with the standing support rule: a prompt contributes only
with at least two verified responses, and a cell below 30 defined prompts is
reported as a gap rather than a number. Support is expected to fail at the
extremes of the sweep, and that failure is itself a finding: a control that
cannot be made diverse without ceasing to be correct is the outcome the
experiment is designed to detect.

## Population

Qwen2.5-0.5B only. It is the scale with a complete four-arm factorial, the
cheapest checkpoints to restore, and the scale where the collapse is starkest,
so it is where a temperature rescue would most plausibly work.

| field | value |
| --- | --- |
| arms | `drgrpo`, `replay_drgrpo`, `maxrl`, `replay_maxrl` |
| domains | Graph, Countdown, Python, MathIR, PantryPlan |
| seeds | 43–47 |
| checkpoints | 100 terminal exports at step 3,072 |
| evaluation | the registered 128 held-out prompts per domain, unchanged |

Nothing about the prompts, splits, grader, canonical keys, response limits, or
draw seeds changes. Only decoding changes.

## Sweep

Stage A of `ops/exp_scaling/launch_e72_decoding_frontier.py`, reused unchanged:
temperature $\in \{0.5, 0.7, 1.0, 1.3, 1.6, 2.0\}$, top-$p$ 1.0, $K=8$, four
draws per prompt. $K=8$ and four draws match the manuscript's evaluation exactly,
so the $T=1.0$ column must reproduce the already-published numbers for these
checkpoints; that reproduction is the gate described below. Stages B (budget)
and C (nucleus truncation) are **not** registered here and are not to be
submitted on the strength of this document.

A frozen-model reference row is included at every temperature and domain, so the
sweep shows where training started as well as where it ended.

## The hardware constraint, stated plainly

The E72 launcher pins each checkpoint to the node that trained it, because GPU
model changes floating-point reduction order and therefore the sampled tokens.
That discipline cannot be reproduced here. Slurm accounting shows this cohort
was scheduled opportunistically across at least five GPU models:

| GPU model | nodes | Qwen-0.5B terminal cells |
| --- | --- | --- |
| a100 | node302 | 39 |
| a5000 | node105 (now **down**), node202–204 | 57 |
| rtx_3090 | node020–026 | 17 |
| a6000 | node103/104, node205–208 | 7 |
| a40 | node101 | 5 |

No single verified pool holds a complete design. The a5000 pool is the largest
and still yields only 6 of 20 domain–arm blocks at five seeds, and 23 paired
seed-pairs; the a100 host yields 18 paired pairs and no complete block.
Restricting to one pool would answer the temperature question on a fraction of
the grid.

We therefore run the full grid unpinned and **measure the hardware term rather
than assuming it away**. A hardware control re-evaluates the 23 a5000-pool
paired seed-pairs a second time on node302 at $T=1.0$, under otherwise identical
settings. We report the distribution of $|\Delta \mathrm{PMD}|$ between GPU
models and compare it to the replay-minus-control gap. The sweep is interpreted
as evidence only if the cross-model term is small relative to that gap; if it is
not, the finding is reported as pinned-subset-only on the a5000 pool, with the
unpinned grid retained as a diagnostic. This decision rule is fixed now,
before any cell is measured.

## Reproduction gate

Before any frontier figure is produced, the $T=1.0$, top-$p$ 1.0, $K=8$,
four-draw column must reproduce the published terminal `pass@8` and `distinct@8`
for the same checkpoints. This is the same gate `aggregate_e72_frontier.py`
already implements. A cell that does not reproduce is not silently dropped: the
mismatch is recorded, and a systematic failure stops the experiment rather than
being worked around, since it would mean the restored checkpoint or the decode
path is not the one the paper reports.

## Cost

Measured, not estimated. The 106 completed Stage-A cells of the original E72
sweep averaged **1.9 minutes** of elapsed time each (minimum 1.0, maximum 3.1)
on one GPU with 8 CPUs.

| item | quantity |
| --- | --- |
| evaluation cells | 630 (100 checkpoints × 6 temperatures, plus 30 frozen reference) |
| generations | 2,580,480 (630 × 128 prompts × 32 samples) |
| compute | ≈ 20 GPU-hours; a few hours wall-clock across a handful of GPUs |
| checkpoint restore | ≈ 92 GiB from `od2961/maxent-grpo-models` (100 × 0.92 GiB) |
| hardware control | 46 extra cells (23 pairs re-measured on node302) |

The restore dominates and is the only part that touches shared storage at scale;
it is a one-time sequential fetch via
`ops/archive_completed_models.py restore --receipt <run>/MODEL_ARCHIVE.json`.
Restored weights are deleted after the sweep completes.

## What this cannot establish

The sweep measures decoding of frozen terminal policies. It cannot show that a
control *trained* differently, or trained at another temperature, would not
close the gap; temperature at evaluation is not temperature during rollout. It
covers one model scale. It does not speak to nucleus truncation, which is
Stage C and unregistered. And it measures how many verified modes a policy
emits, not whether those modes are worth having, which remains the paper's
standing limitation.

## Status

Registered, not submitted. No job is queued and no checkpoint is restored on
the strength of this document alone.
