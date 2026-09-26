# E87: Qwen2.5-3B verified replay plus fixed semantic MaxEnt, seed 70

**Frozen before submission on 2026-08-09.**

## Question

Does fixed semantic MaxEnt on top of verified replay do anything at 3B, in the
same direction it does at 0.5B?

E80-R1 established the two reference arms on this cohort. E87 adds a third arm
on **one paired seed**, and does not re-run either E80-R1 arm.

## Why one seed, and why seed 70

Seed 70 is the only seed for which E80-R1 has **both** arms terminal in all five
domains. Every E87 cell is therefore immediately pairable against a completed
comparator rather than waiting on its own. The launcher fails closed if any of
those ten comparator runs is not terminal at submission time.

This is deliberately a directional probe, not an estimate. One seed per domain
supports a sign and a magnitude, not an uncertainty statement, and the protocol
forbids reporting it as one. If the sign agrees with the completed five-seed
0.5B result, the case for a full 3B cohort is made on evidence; if it does not,
the five-seed cohort is worth more than the compute it would cost to guess.

## Design

- Model: Qwen2.5-3B-Instruct at E80-R1's pinned revision, trained from base.
- Domains: Graph Coloring, Countdown, Python Factors, MathIR, PantryPlan.
- Seed: 70 only. 5 domains x 1 arm x 1 seed = **5 runs**.
- Training, optimizer, memory envelope, decoding, and placement are inherited
  from E80-R1 unchanged: eight passes, 3,072 updates, group size 16, AdamW with
  E80-R1's cosine horizon and `max_step_adjustment`, node302 A100.
- Evaluations and resumable checkpoints every 192 updates.
- Scheduling: `nice=50`, ahead of E80-R1's remaining cells at `nice=100` on the
  same node. This is an explicit prioritisation of the treatment probe over the
  remainder of its own cohort, recorded here because it changes what finishes
  first and therefore what is reportable first.

## Arm

Identical to the E80-R1 `replay` arm — same bank, capacity 16, one scheduled
bank per update, uniform verified-likelihood replay at weight 0.10, live replay
derivative — plus the detached semantic advantage

    A_sem_i = eta * (min(s_i, C) - E[min(s, C)]) / C,   eta = 0.10, C = 5,

applied only to active, parseable, validator-positive rows, added after
Dr.GRPO's own task centering, with no second centering and no outer clamp. The
term is bit-identical in definition to E81, E82, and E83.

All other actuators are hard-disabled: novelty bonus, balance KL, bank entropy
shaping, adaptive coefficients, entropy controllers, token and sequence
entropy, SEED, counterfactual proposals, singleton escape, xDr, reference KL.

## Runtime

E87 runs from the E80-R1 snapshot with three files replaced: `args.py` and
`run_experiment.sh` as in E81, plus `learner/grpo.py` carrying the
canonical-action key repair. PantryPlan's semantic term is therefore live here
from the first update, rather than silently inert as it was in E81, E82, and
E83 before their E85 repair. The new branch is unreachable for any arm that
does not enable semantic MaxEnt, so both E80-R1 comparators remain
code-identical on every path they execute.

## Outcomes and estimands

At every registered half-pass checkpoint, per domain: greedy pass@1, sampled
mean@8, sampled pass@8, mean distinct correct modes@8, and excess multiplicity.

The primary comparison is the paired seed difference `semantic - replay` at
pass 8 for `distinct@8` and `pass@8`. Report the single seed as a single seed:
no mean over seeds, no range, no uncertainty interval, no pooling across
domains, and no checkpoint selection. The secondary comparison is
`semantic - control`.

## Registered interpretation

Fixed before the cells run:

1. Sign agrees with the completed five-seed 0.5B result in most domains: report
   as directional support at 3B and authorise the remaining four seeds.
2. Sign disagrees, or is mixed: report as such and authorise the remaining four
   seeds, because a one-seed disagreement cannot settle it either way.
3. Any cell fails its mechanism gate: the cell is void, not negative.

In no case is a one-seed result reported as an effect estimate.

## Mechanism gate

Before any E87 result is reported, every cell must show a parseable fraction
above 0.5, an eligible fraction above 0.1, and at least one update with a finite
nonzero applied semantic advantage. PantryPlan is included in this gate
precisely because it is the domain that failed it silently before.

## Integrity

All 5 cells are submitted held from one hash-bound snapshot and released only
after their scheduler environments pass an exact audit. A snapshot diverging
outside the three declared files, a non-terminal comparator, a duplicate run
directory, a non-finite loss, a traceback, or a missing pass-8 endpoint fails
closed.
