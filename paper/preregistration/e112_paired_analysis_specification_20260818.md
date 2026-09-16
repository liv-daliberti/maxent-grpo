# E112 paired three-scale analysis specification

Date frozen: 2026-08-18, before E112 submission and without inspecting an
E109 or E112 task-outcome endpoint. PointMaze and every interactive domain are
excluded.

## Complete paired data requirement

The official analysis requires all 75 E112 treatment cells and their exact 75
registered ReplayDr.GRPO comparators: three model scales, five natural static
domains, and five paired training seeds. Every evaluation checkpoint from
update 0 through 3,072 at interval 192 is required. Each checkpoint must have
all four registered fixed-seed neutral sampled-K draws and its deterministic
greedy evaluation. Missing cells, checkpoints, draws, non-finite values, or
conflicting resume rows make the build fail. There is no complete-case subset,
best-checkpoint selection, seed replacement, or domain filtering.

Python Factors controls come only from the parser-repaired E109 cohort.
Non-Python controls come from the registered ReplayDr arms in E78, E79, and
E80-R1. Every treatment row must exactly bind its comparator by model, domain,
seed, run directory, run stamp, and job ID. The builder rejects an active
semantic objective in E109, a parser-surface mismatch, a horizon mismatch, or
any historical Python binding.

## Metrics and paired contrasts

At each checkpoint, sampled pass@8 is `any_correct_at_k`, mean correctness@8
is `mean_at_k`, distinct correct modes@8 is
`distinct_correct_modes_at_k`, and correctness-adjusted breadth is
`distinct@8 - pass@8`, each averaged over the four registered draws. Greedy
pass@1 is the fraction of deterministic first completions with positive
verifier score.

The terminal endpoint is update 3,072. Each trajectory AUC uses all 17
checkpoints, trapezoidal integration, and division by the 3,072-update horizon.
Every effect is E112 minus matched ReplayDr.GRPO within model, domain, and
seed. Family summaries are unweighted five-seed means with paired two-sided
95% Student-t intervals (df=4). Every paired seed, curve, endpoint, source
digest, and effect remains in the machine-readable result.

The terminal and trajectory-AUC forests use the paper's existing 3-by-5
baseline layout. They show pass@8 and correctness-adjusted breadth, with each
seed as an open circle, the five-seed mean as a diamond, the paired 95%
Student-t interval, and exact `n=5` in every panel.

## Frozen decision rules

A model/domain family improves the primary endpoint when its five-seed mean
terminal paired effect on `distinct@8 - pass@8` is strictly positive. The
general extension criterion passes when at least 8 of 15 families improve and
the unweighted mean of all 75 paired terminal pass@8 effects is non-negative.

For the stronger requested all-three-scales conclusion, a scale succeeds only
when the unweighted mean of its 25 terminal paired adjusted-breadth effects is
strictly positive and the unweighted mean of its 25 terminal paired pass@8
effects is non-negative. “Successful at all three scales” requires this rule
to pass separately for Qwen2.5-0.5B, Falcon-1B, and Qwen2.5-3B. These binary
summaries never replace the complete family effects or uncertainty intervals.

No efficacy result is emitted until the complete data requirement passes.
