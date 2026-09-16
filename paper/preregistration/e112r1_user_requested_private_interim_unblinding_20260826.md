# E112-R1 third user-requested exploratory unblinding

Frozen on 2026-08-26 before reading any E112-R1 task-evaluation endpoint that
was not already included in the 2026-08-23 thirty-three-cell private freeze.
The author explicitly requested inspection because E112-R1 had reached 50/75
terminal cells and waiting for all 75 had become operationally protracted.

## Repeated-look deviation

The original E112 analysis specification requires all 75 treatment cells and
all 75 exact comparators before emitting efficacy results. The private looks at
14 and 33 cells already broke continuous confirmatory outcome blindness. This
third sequential look is exploratory, was not preregistered, and must be
disclosed in any eventual E112-R1 report.

No result from this look may be used to cancel, reprioritize, relaunch, retune,
select, or otherwise change an E112-R1 cell or comparator. The registered
treatment, comparator mapping, seeds, domains, horizons, evaluation procedure,
checkpointing, and final 75-cell analysis remain unchanged.

## Outcome-blind membership freeze

Membership is every released E112-R1 ledger cell possessing a valid
`TRAINING_COMPLETE.json` marker when the freeze artifact is written. Selection
reads only the ledger and completion markers, not evaluation outcomes. The
artifact records and hashes the exact admitted membership.

At freeze time this should comprise two complete model-scale panels:
Qwen2.5-0.5B and Falcon3-1B, each with five domains and five seeds. Qwen2.5-3B
remains incomplete and is excluded from efficacy summaries in this look.

## Frozen exploratory analysis

Each terminal cell is paired to its registered ReplayDr.GRPO comparator.
For every complete five-seed model/domain family, report the paired-seed mean
and two-sided 95% Student-t interval for terminal sampled pass@8 and terminal
correctness-adjusted breadth, `distinct@8 - pass@8`. Also report each scale's
unweighted 25-seed mean for both metrics, the number of its five family means
with positive adjusted breadth, and whether that completed scale satisfies the
original scale rule (positive mean adjusted breadth and non-negative mean
pass@8). These are sequentially viewed exploratory summaries, not confirmatory
tests. Do not evaluate the 15-family general criterion or the all-three-scales
criterion while Qwen2.5-3B is absent.

The output must identify which seed effects were already present in the
thirty-three-cell freeze and which are newly revealed. The new 17-cell subset
receives no standalone hypothesis test because scheduler completion is not a
