# E102: full open-bank MaxEnt replay at Qwen-0.5B

Date frozen: 2026-08-14, before submission of any E102 full cell.

## Question

Does adding retention-safe direct bank balancing and model-driven support
expansion to verified replay improve semantic-mode coverage without sacrificing
the correctness and retention delivered by replay alone?

E102 adds one arm to the already completed E78 Qwen2.5-0.5B experiment. It does
not rerun either E78 comparator. The primary comparison is E102 minus the E78
replay arm. The secondary comparison is E102 minus the E78 compute-control arm.
All 25 domain/seed cells are paired by the same source template, initialization,
data order, optimizer schedule, and evaluation draws.

## Frozen design

The five domains are Graph Coloring, Countdown, Python Factors, MathIR, and
PantryPlan. Seeds are 43--47. Every full cell uses 384 training prompts for
eight passes (3,072 optimizer steps), group size 16, one PPO epoch, learning
rate 2e-7, Dr.GRPO, beta zero, and the E78 half-pass evaluation grid.

The training loss is

\[
L = L_{\mathrm{DrGRPO,task}}
  + 0.10 L_{\mathrm{replay\ mass}}
  + 0.10^{\mathrm{safe}} L_{\mathrm{whole-bank\ balance}}.
\]

There is no semantic reward or semantic PPO advantage. The online canonical
bank advantage coefficient is zero. `L_replay mass` is verified-sequence
likelihood over one globally scheduled prompt bank per optimizer update.
`L_whole-bank balance` is the separate within-bank probability-balancing loss.
Its requested coefficient is 0.10, but it is scaled down independently within
each bank whenever necessary so that the combined replay-side derivative never
directly decreases the score of any verified sequence. Replay mass remains
uncapped.

## Explorer and admission

At every eligible update, the explorer makes one additional group-size-16
sample from the original task prompt at temperature 1.2. It uses no exhaustive
support, gold mode count, evaluation statistic, transformed answer, conditioned
prompt, or PPO semantic advantage. A candidate enters the replay bank only if
the ordinary task reward and the independent canonical validator are both
positive and its canonical outcome is absent from the bank. Proposal rows are
never appended to the PPO batch and proposal-only outcomes are kept out of the
neutral online advantage support.

For the four free-form domains this is an isolated actor request. PantryPlan's
frozen E78 treatment uses a finite six-bit learner policy, so its extra group is
sampled directly from that same fixed-shape learner policy with an isolated
seed and is discarded after validation/admission. This is an implementation
difference required by the inherited action interface, not a scientific
difference: both paths sample the unchanged policy on the original prompt and
send only verified novel exemplars to replay.

Exploration is not restricted to singleton banks: discovering further modes is
part of the treatment. Newly admitted modes receive four replay visits at a 4x
raw mass weight. Weights are renormalized within the bank so the total mass
budget remains equal to the bank size. Priority affects only replay mass; the
balance term always compares the unweighted whole bank. The priority queue is
FIFO and checkpointed.

The extra explorer samples are treatment compute rather than a compute-matched
ablation. Realized and charged proposal/replay tokens will therefore be
reported explicitly. Scientific conclusions concern the bundled mechanism and
must not attribute an effect uniquely to balance, safe capping, exploration, or
priority.

## Mechanism gates

Before releasing the full set, a Graph Coloring seed-43 smoke must complete and
show finite training, live replay, retention-safe balance telemetry, zero
proposal rows to PPO, zero proposal objective-support delta, no transform/gold/
desired-support/evaluation feedback, and at least one discovered mode that is
subsequently replay-prioritized. A smoke failure blocks release and is not a
license to tune on downstream evaluation.

The initial mechanism smoke used eight distinct prompts for one pass. It
completed safely but only two prompts produced a valid anchor, and both
proposal groups returned already-known modes (13 neutral validator positives,
9 proposal validator positives, zero novel outcomes). Thus it failed the
admission gate before any full cell was released. This was a structurally weak
test of the registered eight-pass design: it gave no prompt a second discovery
attempt. The repaired smoke freezes four Graph prompts for eight passes (32
updates), leaving every treatment coefficient and the one-group-per-update
explorer unchanged. The full campaign remains blocked unless this recurrence-
matched smoke produces an admission followed by priority replay. A separate
eight-step Pantry compatibility smoke already passed the same safety and
actuation checks for the canonical learner path; it is implementation evidence,
not a replacement for the free-form Graph gate.

For full runs, any non-finite value, proposal leakage into PPO, nonzero proposal
objective-support delta, enabled transform, positive applied score derivative
above 1e-7, or configuration drift is a mechanism failure. Admissions are a
realized stochastic quantity, not an outcome-tuned gate; they are reported by
domain and seed. The campaign-level mechanism check requires at least one
admission and subsequent priority actuation in every domain across its five
seeds. A domain that fails that check is reported as an inert explorer for that
domain, not silently pooled as evidence for support expansion.

## Outcomes

The registered task and semantic-coverage outcomes are exactly the E78
checkpoint outcomes: greedy accuracy, sampled any-correct@8, sampled mean@8,
distinct correct@8, and mode coverage@8. Primary summaries are paired E102
minus E78 replay changes at every half-pass and at pass 8, shown separately for
each domain and then averaged with equal domain weight. Correctness/retention
is a co-primary safety outcome: a coverage gain accompanied by a material loss
in sampled correctness is not called an improvement.

No arm is selected or stopped from intermediate evaluation results. All 25
held jobs must pass a scheduler/environment audit before their atomic release.
E102 runs on the `all` partition under the `mltheory` account on an explicit
healthy-node pool that excludes nodes 302 and 105. Hardware affects runtime,
not the registered optimizer-step comparisons.
