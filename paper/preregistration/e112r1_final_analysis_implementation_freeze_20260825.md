# E112-R1 final-analysis implementation freeze

Frozen: 2026-08-25, while E112-R1 and two registered E109 Python comparator
continuations remain nonterminal. No additional task-evaluation endpoint file
was read to implement or test this builder.

This is an implementation and disclosure freeze, not a new efficacy
preregistration. The decision rules, 75 paired cells, five seeds per family,
17 evaluation checkpoints, four sampled draws, terminal endpoint, normalized
AUC, and Student-t intervals remain exactly those frozen in
`e112_paired_analysis_specification_20260818.md`. Previous private interim
looks cannot be undone and prevent a confirmatory-blind interpretation.

## Fail-closed materialization

The final builder must reject any missing treatment or comparator cell,
checkpoint, sampled draw, greedy result, non-finite value, conflicting resume
row, source-provenance drift, or paired-identity drift. It writes nothing
until all 150 trajectories have passed validation. Python controls come only
from E109. The registered E109 run identities remain authoritative; the
seed-73 and seed-74 Qwen-3B continuation artifact is required and retained as
execution provenance for those same run directories.

The builder reuses the already-tested E105 curve parser, registered-step
constructor, AUC implementation, and paired-interval implementation. It does
not clone a second endpoint reader. The terminal and AUC forest renderer is a
thin E112 adapter over the tested E105 3-by-5 forest grammar; it changes the
schema, labels, and disclosure payload, not the statistical extraction or
visual encoding.

## Metric and estimand disclosure

Every output retains sampled pass@8 and raw distinct-correct-modes@8 as the
primitive endpoint vector. Correctness-adjusted breadth is the derived value
`raw distinct@8 - pass@8`. The original E112 family and scale decisions still
use that derived adjusted value; raw breadth is reported so an accuracy rescue
cannot disappear from the scientific record.

E112-R1 uses a repaired request path and a later frozen source snapshot than
its historical ReplayDr comparators. Its final contrast therefore estimates
the effect of the complete E112-R1 verified-support-discovery bundle relative
to the registered historical ReplayDr runs. It must not be described as an
isolated causal effect of semantic v7, counterfactual proposal generation, or
replay admission. Component attribution belongs to the same-plumbing C/P/F
successor design.
