# E55: per-rollout verified anchor with singleton cold start

**Status: FROZEN BEFORE E55 ENGINEERING SMOKE OR SENTINEL SUBMISSION — 2026-07-26**

## Failures being repaired

E53's bank-conditioned reverse KL preserves relative diversity but has
zero-sum score gradients and cannot raise the common mass of verified modes.
E54 replaces it with uniform verified-exemplar likelihood. Its bounded
16-update engineering smoke established, without using evaluation behavior,
that the new raw score gradient is exact and finite. It also exposed a measure
mismatch: the first two active auxiliary updates had pre-clip policy-gradient
norms 16.49 and 11.82, versus ordinary non-replay updates generally below one.
The global max-norm clip would therefore select almost entirely the auxiliary
direction.

E54 also remains idle on a singleton bank. That is unavoidable for a
*canonical entropy sensor*: normalized entropy on one observed mode is
undefined and contains no direction toward an undiscovered valid mode. It is
not unavoidable for a verified-likelihood anchor.

E55 fixes both engineering gaps without changing an entropy target or selecting
a coefficient from behavioral evaluation.

## Three distinct roles

1. **Support-independent discovery.** The E53 direct conditional-content
   entropy objective remains active even with no verified mode. Its base
   coefficient is `0.000075`, its first 64 observations define its own
   reference, EMA decay is `0.90`, and its post-warmup coefficient is
   `0.000075 * H_ref / H_ema`, without lower or upper projection.
2. **Verified correctness anchor.** As soon as one validator-positive mode has
   been sampled for a prompt, its retained exact token sequence participates
   in uniform verified likelihood. This prevents exploration or later updates
   from erasing the only known correct behavior.
3. **Multi-mode entropy restoration.** Once a prompt has at least two observed
   validator-positive modes, the model's normalized entropy over their current
   teacher-forced scores drives the same unbounded replay controller as E53.

The bank alone is never claimed to discover an unseen valid mode. Only
support-independent policy exploration can cross that information boundary.

## Per-rollout uniform verified likelihood

For every current prompt with observed bank `B_x`, including `|B_x|=1`, retain
the E54 length-normalized score

`s_(x,k) = |r_(x,k)|^(-1) sum_t log pi_theta(r_(x,k,t) | x, r_(x,k,<t))`

and raw auxiliary

`L_verified = mean_x mean_(k in B_x) -s_(x,k)`.

The raw score gradient sums to `-1` over replayed prompts. The ordinary policy
objective is estimated from 16 rollout candidates, whereas the replay bank is
one auxiliary pseudo-rollout at the optimizer boundary. E55 therefore applies
the fixed sampling-measure factor

`c_replay = 1 / num_samples = 1/16`.

The mathematical auxiliary is

`alpha_t * (15/16) * (1/16) * L_verified`.

`15/16` is the frozen leave-one-out reward-estimator correction inherited from
E53. `1/16` is not a tuned coefficient, entropy target, or gradient clip: it
counts one replay pseudo-observation in the same units as one member of the
16-sample rollout group. The raw and applied score-gradient sums must be logged
as `-1` and `-1/16`, respectively.

The replay alpha remains `0.10` during its first 64 *multi-mode* observations.
Afterward it remains exactly

`alpha_(t+1) = 0.10 * z_ref / z_ema,t`

with EMA decay `0.90` and no lower or upper projection. Thus alpha itself is
not bounded. A singleton replay update uses the current alpha but is ineligible
for the canonical-entropy EMA, warmup count, and reference. If no multi-mode
observation has occurred, the current alpha is the base `0.10`.

## Bank and information boundary

E55 retains E53's validator-positive one-time discovery bonus `0.50`, sampled
count-bank entropy coefficient `0`, one immutable exemplar per observed
prompt/outcome, capacity 16 fixed from rollout width, and full discovery
counters beyond retained capacity.

Training may use sampled executable reward, canonical keys and exact tokens of
validator-positive on-policy discoveries, the current model's teacher-forced
scores, rollout width, and each controller's own observations.

Training may not use gold answers or support counts, reference-answer
multiplicity, evaluation distinct-correct/pass/mean metrics, a desired mode
count, a domain-specific entropy target, or a coefficient selected from E53 or
E54 behavioral outcomes.

## Engineering smoke

Before a full sentinel, run at most 16 optimizer updates each on graph coloring
and executable Python factors from fresh initialization. Shared evaluation
output is ignored.

The smoke must establish:

- exact raw score-gradient sum `-1`;
- exact objective scale `1/16` and applied score-gradient sum `-1/16`;
- finite two-pass, one-row-chunk replay forward/backward;
- graph multi-mode replay advances the entropy controller only when eligible;
- Python or graph singleton replay can be actuator-active while reporting zero
  entropy-eligible groups and leaving the controller observation count
  unchanged;
- no repeated auxiliary-only domination of the global max-norm clip;
- finite generation, actor synchronization, and exact checkpoint round-trip.

Smoke outputs, weights, optimizer state, banks, controller state, and
evaluation values are discarded and cannot tune E55.

## Stage S: 50-pass three-domain sentinel

Run one E55 treatment from Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775` with engineering seed `9010` in
graph coloring, Countdown easy3, and executable Python factors. Use the same
prompt/evaluation pools, row order, group size 16, learning rate `2e-7`, one
PPO epoch, `beta=0`, maximum norm 1, temperature 1, top-p 1, response limit
192, fixed quarter-pass evaluation schedule, and exactly 50 complete
prompt-pool passes as E53.

The frozen E53 seed-9010 matched Dr.GRPO arms are the comparators. Only the
three fresh E55 treatment jobs are submitted.

All treatment jobs must reach the exact terminal boundary with finite task,
direct-entropy, unbounded-alpha, replay, gradient, response-length, and
evaluation telemetry. Controller equations, singleton idling, applied
`1/16` measure, no-projection flags, no-gold-feedback flags, and checkpoint
state must reproduce exactly.

For each domain, over the final eight quarter-pass evaluations, E55 must:

- have higher mean distinct-correct@8 than frozen matched E53 Dr.GRPO;
- win distinct-correct@8 at least six of eight times;
- have higher mean `(distinct-correct@8 - pass@8)` than matched Dr.GRPO;
- win that excess-multiplicity quantity at least six of eight times;
- show positive excess multiplicity at least six of eight times;
- retain at least 75% of its own best rolling-eight distinct-correct mean;
- trail matched terminal pass@8 and mean@8 by no more than `0.03`.

Any runtime or domain failure rejects E55. Sentinel behavior may reject the
mechanism but may not tune its coefficients, capacity, warmup, measure factor,
or behavioral thresholds.

## Conditional replication

Only a machine-readable terminal three-domain approval bound to this protocol,
source, execution snapshot, launcher, auditor, and frozen E53 control identity
may authorize fresh seeds 43, 44, and 45. Every treatment seed and the
three-seed mean must independently pass the same stability, multiplicity, and
task-quality gates.
