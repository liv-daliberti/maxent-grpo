# E111: verified-support MaxEnt plus support discovery at three scales

Date frozen: 2026-08-18, before submission of any E111 job and before inspecting
any E105 or E109 endpoint outcome.

## Question

Does the repaired implementation realize the intended decomposition of
conditional semantic entropy with ReplayDr.GRPO at Qwen2.5-0.5B, Falcon-1B,
and Qwen2.5-3B?

1. a target-free sampler discovers validator-positive support;
2. ReplayDr.GRPO raises the likelihood of every discovered verified mode; and
3. the semantic score-function estimator redistributes probability across that
   verified support without assigning probability to a fictional unseen-success
   bucket.

E111 is an outcome-blind mechanism gate, not an efficacy experiment. PointMaze
is excluded. The still-running E105 v6 treatment is not evidence for this
repair: v6 centers only over successful rows in the current 16-way group and
therefore erases every singleton success. E105 and E109 had zero terminal cells
at the time of this freeze; no endpoint or evaluation outcome from either
campaign was inspected.

## Frozen estimator

For an active, parseable, verifier-positive sampled outcome y on prompt x, let
S be the union of:

- prior neutral-policy verified outcomes for x;
- current leave-one-out verifier-positive peers;
- validator-positive replay exemplars for x, including proposal-only support;
- the currently sampled verified outcome y.

Only the first two sources contribute empirical frequency counts. A replay or
proposal key contributes support membership with count zero. With pseudocount
one, q is normalized only over S. There is no structural unseen bucket. The
semantic advantage is

    A_sem(y) = eta * (c(y) - sum_z q(z)c(z)),
    c(z) = min(-log q(z), 5) / 5,
    eta = 0.10.

Failures, inactive rows, and unparseable rows receive exact zero and never
enter semantic history. A sole verified support element is an exact no-op. A
rare singleton remains live once another verified mode is known. Proposal-only
keys never become neutral-policy frequency observations.

The replay term is the existing ReplayDr.GRPO uniform verified-likelihood
objective with coefficient 0.10. Each scheduled prompt bank contributes the
unweighted mean negative sequence log likelihood over all stored verified
modes. Admission priority is disabled (zero visits, multiplier one), so the
replay target remains uniform. The task update remains ordinary Dr.GRPO, beta
zero, with no token-entropy, outcome-collision, UCPO, RLEP, bank-advantage,
RMS-controller, or balance loss.

## Frozen support discovery

At every eligible update, one isolated group of 16 is sampled from the original
unchanged prompt at temperature 1.2. It uses no gold support, desired mode,
evaluation statistic, transformed answer, or conditioned prompt. A response is
admitted only when ordinary task reward and the independent executable
canonical validator are both positive and its key is novel for that prompt.
Proposal rows never enter PPO. They contribute only verified support membership
and a replay exemplar. Retention tracking is diagnostic only; adaptive refresh
is disabled.

## Recurrence-matched three-scale gate

The gate contains all five natural domains at each of the three paper scales,
using one frozen seed per scale: Qwen-0.5B seed 43, Falcon-1B seed 55, and
Qwen-3B seed 70. Each cell uses the first eight training prompts for eight
passes, giving 64 optimizer updates with group size 16. This intentionally
matches the recurrence structure of the eight-pass full experiment. The prior
64-prompt/one-pass smoke could not test a persistent predictor because it never
revisited a prompt.

All 15 held jobs must pass a scheduler/environment audit before atomic release.
The runtime audit reads training telemetry and scheduler state only; evaluation
outcomes are forbidden. Every cell must:

- reach 64 optimizer steps with finite telemetry and no failure marker;
- report v7 verified-support mode active, replay-bank support inclusion active,
  v5/v6 inactive, and the RMS controller inactive;
- keep every semantic advantage within eta plus numerical tolerance;
- report proposal rows to PPO as zero, transformed proposals as zero, adaptive
  retention as zero, and uniform replay mass weights exactly one;
- preserve proposal-only support outside neutral frequency counts; and
- report a live applied ReplayDr gradient whenever a replay group exists.

In addition, each model scale must contain at least one cell that completes the
full causal chain: a novel proposal admission, external verified support visible
to v7, verified support size at least two on an eligible row, nonzero v7
semantic RMS, and a nonzero uniform ReplayDr applied gradient. This is a
mechanism criterion, not a task outcome. Failure blocks a full corrected cohort
and is reported without choosing a domain or coefficient from evaluation.

## Downstream decision

If E111 passes, E105 is permanently superseded without outcome release and a
new E112 full cohort is frozen prospectively: all three scales, all five natural
domains, five paired seeds, 384 prompts for eight passes. The already registered
E109 Python ReplayDr.GRPO cells remain valid matched comparators because their
objective contains neither v6 nor v7. If E111 fails, no corrected full campaign
is released and the failure mode is diagnosed from mechanism telemetry only.
