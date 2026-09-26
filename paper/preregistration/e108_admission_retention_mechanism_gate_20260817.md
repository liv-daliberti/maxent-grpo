# E108: proposal admission-to-retention mechanism gate

Frozen: 2026-08-17, before any E108 model update or outcome inspection.

## Motivation and status of prior evidence

E102 is length-normalized exemplar replay plus within-bank score balancing and
an original-prompt explorer. It is not exact semantic-mode MaxEnt, and its
score-gradient cap is not a guarantee that every stored mode's policy
probability survives shared-parameter updates. E103 changed only proposal
sampling through an explorer-starvation fallback. Although that fallback
produced 86 extra admissions over the terminal five-domain cohort, E103 did not
improve terminal breadth relative to E102. Admission is therefore not treated
as retention evidence.

E108 tests the missing link directly: after a proposal-derived exemplar is
admitted, does it reappear in later neutral rollouts, and does its
teacher-forced likelihood survive later updates? More proposal sampling is not
an E108 treatment.

## Frozen arms

Both arms use Qwen2.5-0.5B-Instruct, seed 43, the five static ModeBench domains,
8 training rows for 8 passes (64 optimizer updates), and the exact E102
learning stack:

- task-only Dr.GRPO PPO;
- split length-normalized replay mass and whole-bank score balance, each with
  coefficient 0.10 per rollout;
- retention-safe direct score-gradient capping;
- one original-prompt proposal attempt at temperature 1.20;
- proposal rows discarded before PPO and kept outside neutral on-policy count
  support;
- four initial replay-priority visits at multiplier 4.0;
- no explorer-starvation fallback and no semantic PPO advantage.

The paired arms are:

1. `retention_tracking`: add checkpointed measurement only.
2. `adaptive_retention`: add the same measurement plus bounded mass-replay
   priority refresh.

The adaptive controller refreshes an admitted exemplar to at most four
remaining priority visits after either (a) two later neutral prompt groups in a
row without verifier-positive reappearance, or (b) a mean-token replay log
likelihood drop greater than 0.5 nats from that exemplar's own first measured
post-admission value. Likelihood refresh has a two-observation cooldown.
Refresh changes only the existing within-bank normalized mass weights. It does
not change PPO rows, task advantages, proposal sampling, total replay-mass
normalization, or whole-bank balance support.

## Measurement

Each proposal-admitted stored exemplar receives a stable prompt/outcome record.
The neutral group that preceded its admission is excluded. Later opportunities
record verifier-positive row hits and whether the exemplar ever reappears.
Every time its replay group is teacher-forced, E108 records mean-token log
probability and full-sequence log probability. The first such score is the
exemplar's self-baseline; there is no cross-exemplar length comparison in the
controller.

Reported mechanism quantities are tracked admissions, rollout-eligible
admissions, on-policy conversion fraction, rollout row frequency,
score-follow-up admissions, score-retained fraction, joint retained fraction,
mean-token and sequence likelihood drops, refresh requests by trigger, and
priority visits actually added.

## Fail-closed mechanism gate

The gate covers all ten arm-domain cells. It fails if any cell does not reach
64 updates, emits non-finite telemetry, leaks a proposal row into PPO, changes
neutral objective support from a proposal, enables a proposal transform or
starvation fallback, consumes evaluation/gold/desired-support feedback, or
applies a positive direct derivative to a verified exemplar score.

Both arms must materialize tracking, at least one proposal admission, and at
least one replay-score observation in aggregate. The passive arm must emit zero
adaptive refresh requests and add zero adaptive priority visits. The adaptive
arm must emit at least one retention-triggered request and actually add at
least one bounded priority visit in aggregate. These are mechanism checks, not
outcome gates. Terminal breadth and correctness are reported regardless of
sign and do not determine whether a later full cohort is released.

## Information prohibition

The tracker and controller may use only verifier-positive training rollouts,
stored training exemplars, and current-policy teacher-forced scores. They may
not read fixed evaluation draws, terminal outcomes, exhaustive or gold
support, desired mode counts, or E102/E103 outcome differences. The three
forbidden-feedback telemetry fields must remain exact zero.
