# E105 paired analysis specification

Frozen on 2026-08-17 before E105 submission and while every E104/E106
post-update outcome remained blinded. This specification makes the estimands
in the original E105 protocol executable; it does not change the treatment,
cohort, stopping rule, or release gate. PointMaze and every other interactive
environment are excluded.

## Required data

The analysis contains all 75 E105 treatment cells and their 75 registered
ReplayDr.GRPO comparators: three model scales, five static domains, and five
paired training seeds. Every registered checkpoint from update 0 through
3,072 at interval 192 is required. At every checkpoint, all four fixed-seed
neutral sampled-K evaluation draws and the deterministic greedy evaluation
must be present. Missing cells, checkpoints, draws, or non-finite values make
the build fail; there is no best-checkpoint selection or domain filtering.

## Metrics and contrasts

For each run and checkpoint, sampled pass@8 is `any_correct_at_k`, sampled
mean correctness@8 is `mean_at_k`, and sampled distinct correct modes@8 is
`distinct_correct_modes_at_k`, each averaged over the four registered draws.
Greedy pass@1 is the fraction of evaluation rows whose first deterministic
completion receives a positive verifier score. Excess multiplicity is
`distinct@8 - pass@8`.

For each metric, the terminal value is update 3,072. AUC is trapezoidal over
all 17 registered checkpoints and divided by the 3,072-update horizon. Every
contrast is treatment minus its matched ReplayDr.GRPO run within model,
domain, and seed. Family summaries are unweighted means over the five paired
seed effects with two-sided 95% Student-t intervals (df=4). Curves, endpoints,
and paired effects for every seed and checkpoint remain in the machine-readable
result.

As amended before submission by the E109 repaired-comparator protocol, the 15
Python Factors controls come only from E109 under the same repaired parser and
snapshot as E105. The 60 non-Python controls remain the registered historical
ReplayDr.GRPO cells. The builder must reject a historical Python control or an
E109 ledger with any active semantic objective, parser/snapshot mismatch, or
incomplete scale-by-seed grid.

The registered endpoint forest uses the paper's existing 3-by-5 baseline
format. It shows treatment-minus-ReplayDr.GRPO paired effects for pass@8 and
correctness-adjusted breadth `distinct@8 - pass@8`, with every seed as an open
circle, the five-seed mean as a diamond, a paired 95% Student-t interval, and
exact `n=5` printed in every panel.

A separate registered trajectory-AUC forest uses the identical 3-by-5 layout,
seed encoding, uncertainty interval, and exact `n=5`. It shows the paired
treatment-minus-ReplayDr.GRPO effects on normalized pass@8 AUC and normalized
`distinct@8 - pass@8` AUC over all 17 checkpoints. The terminal and AUC forests
must both be emitted from the same complete machine-readable result; neither
may filter families or seeds after outcomes are observed.

## Fixed general-extension decision rule

A model/domain family counts as a breadth improvement when its five-seed mean
terminal paired effect on sampled distinct correct modes@8 is strictly
positive. “Most families” means at least 8 of the 15 families. Systematic
correctness loss means the unweighted mean of all 75 paired terminal pass@8
effects is negative. The repaired method is called a successful general
extension only if at least 8 families improve breadth and the 75-cell mean
pass@8 effect is non-negative. Otherwise the result is reported as
heterogeneous or negative. This binary summary does not replace the complete
family-level effects, uncertainty intervals, or correctness/breadth tradeoff
reporting required by the original protocol.
