# E54: target-free uniform verified likelihood

**Status: FROZEN BEFORE E54 ENGINEERING SMOKE OR SENTINEL SUBMISSION — 2026-07-26**

## Structural failure being repaired

E53 supplies a persistent gradient path to validator-positive modes that have
left the current rollout. Its bank-conditioned reverse KL has score derivative

`d KL(U(B_x) || softmax(s_x)) / d s_(x,k) = q_(x,k) - 1/|B_x|`.

Those derivatives sum to zero within every prompt. The actuator can therefore
redistribute model score among retained correct modes, but it cannot raise
their common score against invalid outputs. This distinction is already
visible without a terminal behavioral decision: E53 can preserve multiple
graph-coloring modes while task-valid probability falls behind matched
Dr.GRPO.

E54 changes only that actuator. It does not select a stronger coefficient,
support target, desired entropy, or domain-specific setting from E53 outcomes.

## Discovery and bank

E54 inherits E53 without adjustment:

- direct conditional-content entropy base coefficient `0.000075`;
- direct-controller warmup 64 and EMA decay `0.90`;
- direct coefficient `0.000075 * H_ref / H_ema` after warmup;
- no lower or upper coefficient projection;
- validator-positive one-time discovery bonus `0.50`;
- sampled count-bank entropy coefficient `0`;
- one exact response-token exemplar per observed prompt/outcome;
- replay eligibility only after two distinct validator-positive modes;
- replay capacity 16, fixed from rollout width rather than semantic support;
- full discovery counts continue after retained replay capacity is reached.

The direct channel remains the only mechanism that can discover an outcome
that has never been produced. Replay uses no response other than an exact
validator-positive response previously sampled from the policy.

## Uniform verified-likelihood actuator

For observed prompt-local bank `B_x`, retain the E53 length-normalized
teacher-forced score

`s_(x,k) = |r_(x,k)|^(-1) sum_t log pi_theta(r_(x,k,t) | x, r_(x,k,<t))`.

E54 minimizes

`L_verified = mean_x mean_(k in B_x) -s_(x,k)`.

Every eligible prompt has equal weight and every observed mode within a prompt
has equal weight. The score derivative is

`d L_verified / d s_(x,k) = -1 / (number_of_eligible_prompts * |B_x|)`.

It sums to `-1` across the replay batch, supplying the common verified-mode
component absent from E53. In model-logit space, uniform teacher forcing
simultaneously raises validator-positive exemplar likelihood and continues to
give low-probability target tokens restorative gradients.

The E53 bank-conditioned reverse KL remains a detached diagnostic. It is not
added to the E54 loss. This avoids introducing a second coefficient selected
from behavioral outcomes.

## Unbounded model-entropy controller

The controller and sensor are identical to E53. Let

`q_theta(k | x) = softmax_k(s_(x,k))`

over retained observed modes and

`z_t = mean_x H(q_theta(. | x)) / log |B_x|`.

Ineligible updates are idle. For the first 64 eligible observations,
`alpha=0.10` and `z_ref` is the warmup mean. With EMA decay `0.90`, thereafter

`alpha_(t+1) = 0.10 * z_ref / z_ema,t`.

There is no lower or upper projection. E54 deliberately inherits `0.10`
rather than choosing a new value from E53 evaluation curves. The exact audit
must show a replay score-gradient sum of `-1`, no projection, and no
gold-support feedback.

## Information boundary

Training may use ordinary sampled executable reward, canonical keys and exact
tokens of validator-positive on-policy discoveries, current teacher-forced
model scores on retained exemplars, and each controller's own warmup history.

Training may not use a gold answer list, gold support count, reference-answer
multiplicity, evaluation distinct-correct/pass/mean metrics, a desired mode
count, a domain-specific entropy target, or a coefficient chosen from E53
behavioral results.

## Engineering smoke

Before the outcome sentinel, one graph-coloring run of at most 16 optimizer
updates may verify argument transport, exact `-1` score-gradient sum, finite
two-pass chunked backward, replay activation, checkpoint round-trip, memory,
and actor synchronization. Any evaluation emitted by shared infrastructure is
ignored. Smoke weights, model outputs, bank, controller state, and optimizer
state are discarded and cannot tune or authorize a coefficient change.

## Stage S: full-horizon cross-domain sentinel

Run E54 from the frozen Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775` with engineering seed `9010` on
graph coloring, Countdown easy3, and executable Python factors. Use the same
prompt/evaluation pools, row order, group size 16, learning rate `2e-7`, one
PPO epoch, `beta=0`, maximum norm 1, sampling temperature 1, top-p 1, response
limit 192, fixed quarter-pass evaluations, and exactly 50 complete prompt-pool
passes as E53.

The frozen E53 seed-9010 matched Dr.GRPO arms are the comparators; they may not
be relaunched or selected after observing E54. E54 therefore submits one fresh
treatment job per domain. Source regression tests must establish that the E53
`bank_balance` option retains its prior loss and gradient exactly.

All three E54 jobs must reach the exact boundary with finite task loss, direct
entropy, both unbounded alphas, replay actuator loss, detached balance
diagnostic, replay model entropy, gradient, response length, and evaluation
telemetry. Checkpoints must restore the exemplar bank and both controllers
exactly.

For each domain, over the final eight quarter-pass evaluations, E54 must:

- have higher mean distinct-correct@8 than frozen matched E53 Dr.GRPO;
- win distinct-correct@8 at least six of eight times;
- have higher mean `(distinct-correct@8 - pass@8)` than matched Dr.GRPO;
- win that excess-multiplicity quantity at least six of eight times;
- show positive excess multiplicity at least six of eight times;
- retain at least 75% of its own best rolling-eight distinct-correct mean;
- trail matched terminal pass@8 and mean@8 by no more than `0.03`.

Any runtime or domain failure rejects E54. Sentinel outcomes may reject this
mechanism but may not tune its coefficients, capacity, warmup, or behavioral
thresholds.

## Conditional replication

Only a machine-readable terminal three-domain approval bound to this protocol,
source snapshot, execution snapshot, launcher, auditor, and frozen E53 control
identity may authorize fresh seeds 43, 44, and 45. Every treatment seed and
the three-seed mean must independently pass the same stability, multiplicity,
and task-quality gates.
