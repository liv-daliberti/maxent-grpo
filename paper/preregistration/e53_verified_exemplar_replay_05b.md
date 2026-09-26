# E53: target-free verified-exemplar replay

**Status: FROZEN BEFORE E53 SENTINEL SUBMISSION — 2026-07-26**

## Failure being repaired

E52 separated an always-live token-entropy discovery actuator from the
canonical count bank. Early cross-domain telemetry nevertheless exposes a
second actuator gap. The count-bank advantage can affect a verified mode only
when that mode is sampled in the current rollout. After a discovered mode
disappears from the on-policy group, its count remains in memory but it has no
gradient path back into the policy. Direct content-token entropy can raise
surface-form entropy without restoring the missing executable outcome.

E53 keeps E52's discovery bridge and replaces persistent count-entropy shaping
with differentiable replay of validator-confirmed exemplars.

## Discovery bridge

The direct conditional-content entropy objective and its inverse controller
are identical to E52:

- EOS is removed from each visited-state categorical distribution;
- active token positions are averaged within a response and responses receive
  equal weight;
- the sampled state distribution is detached;
- base coefficient `0.000075`;
- first 64 observations define the run's own entropy reference;
- EMA decay `0.90`;
- after warmup, `lambda_(t+1) = 0.000075 * H_ref / H_ema`;
- no lower or upper projection, target answer count, target mode count, or
  evaluation feedback.

This channel remains active before any correct mode exists and is responsible
for discovering candidate modes not already in the verified bank.

## Verified replay actuator

For each prompt, retain one exact response-token exemplar for each newly
observed validator-positive canonical outcome. Duplicate responses do not add
an exemplar. A prompt becomes replay-eligible only after at least two distinct
verified outcomes have been produced on policy.

Replay is compute-bounded at 16 retained modes per prompt, equal to the frozen
training rollout width. The full discovery counter continues past 16; only the
teacher-forcing workload is bounded. Capacity is fixed before training and is
not a claim about, or estimate of, the true number of valid outcomes.

For replay-eligible prompt `x`, let `r_(x,k)` be the retained token sequence
for observed mode `k`, and define its current length-normalized teacher-forced
model score

`s_(x,k) = |r_(x,k)|^(-1) sum_t log pi_theta(r_(x,k,t) | x, r_(x,k,<t))`.

Let `q_theta(k | x) = softmax_k(s_(x,k))` over retained modes only. The
restorative loss is

`L_replay = mean_x KL(U(B_x) || q_theta(. | x))`.

The reverse KL direction is deliberate: its score gradient is
`q_theta(k | x) - 1/|B_x|`, so a retained mode with near-zero current
probability continues to receive a finite restorative gradient. Replay never
uses an unobserved, validator-negative, reference-only, or manually supplied
answer.

The ordinary online bank entropy coefficient is `0`. The one-time
validator-positive discovery bonus remains `0.50`, with the existing atomic
group update and no post-discovery count-entropy impulse.

## Unbounded replay controller

The controller observes the model's own normalized entropy over retained mode
scores,

`z_t = mean_x H(q_theta(. | x)) / log |B_x|`.

Ineligible singleton/empty banks are idle and do not consume warmup. For the
first 64 eligible observations, replay alpha is `0.10` and
`z_ref = mean(z_1, ..., z_64)`. With EMA decay `0.90`, after warmup

`alpha_(t+1) = 0.10 * z_ref / z_ema,t`.

There is no lower or upper projection. The coefficient does not use the
bank's empirical count entropy, a gold support size, a desired distinctness
value, or any evaluation metric. Float64 is used only for the detached entropy
sensor to prevent float32 probability underflow; it is not an epsilon or a
coefficient bound.

Replay runs once per optimizer boundary. With gradient accumulation width
`A`, the replay backward scalar is multiplied by `A` solely to cancel
DeepSpeed's mechanical `1/A` loss scaling. The logged objective coefficient
and resulting mathematical gradient are independent of microbatch placement.

## Information boundary

Training may use:

- ordinary executable reward for sampled responses;
- canonical keys of validator-positive responses actually sampled on policy;
- exact token sequences of those same responses;
- the current model's teacher-forced scores and entropy on retained exemplars;
- each controller's own warmup observations.

Training may not use:

- a gold list or count of valid outcomes;
- evaluation distinct-correct, pass@K, mean@K, or coverage;
- reference-answer multiplicity;
- a domain-specific entropy or diversity target;
- coefficients selected from E52 evaluation outcomes.

The replay capacity derives from the frozen training group width, not the
evaluation draw count or any domain support.

## Engineering smoke

Before any E53 outcome sentinel, one short graph-coloring pipeline smoke may
verify only: argument plumbing, validator/exemplar admission, response-token
masking, finite replay forward/backward, accumulation scaling, telemetry,
checkpoint round-trip, memory, and actor synchronization. It uses at most one
16-prompt sweep and no mode-coverage evaluation. Its weights, optimizer state,
controller references, bank, and outcomes are discarded and cannot authorize
or tune E53.

## Stage S: full-horizon cross-domain sentinel

Use fresh engineering seed `9010` in graph coloring, Countdown easy3, and
executable Python factors. Each domain has:

1. matched Dr.GRPO with passive verified-discovery tracking;
2. E52 direct inverse conditional-content entropy;
3. the same direct controller plus E53 novelty and verified replay.

All arms start from Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775` with group size 16, learning rate
`2e-7`, one PPO epoch, `beta=0`, maximum norm 1, sampling temperature 1,
top-p 1, response limit 192, and exactly 50 complete prompt-pool passes.
Prompt pools and fixed quarter-pass evaluations are identical to E52.

All nine jobs must reach the exact boundary with finite task loss, direct
entropy, both unbounded alphas, replay loss, replay model entropy, gradient,
response length, and evaluation telemetry. Controller equations must reproduce
from their logged sensors; replay must be absent for support below two, present
for eligible banks, and report projection and gold-support feedback as zero.
Checkpoint state must restore the exemplar bank and both controllers exactly.

For each domain, over the final eight quarter-pass evaluations, the replay arm
must:

- have higher mean distinct-correct@8 than matched Dr.GRPO;
- win distinct-correct@8 at least six of eight times;
- have higher mean `(distinct-correct@8 - pass@8)` than matched Dr.GRPO;
- win that excess-multiplicity quantity at least six of eight times;
- show positive excess multiplicity at least six of eight times;
- retain at least 75% of its own best rolling-eight distinct-correct mean;
- trail matched terminal pass@8 and mean@8 by no more than `0.03`.

Any runtime or domain failure blocks replication. Sentinel outcomes may reject
the mechanism but may not tune its coefficients, replay capacity, warmup, or
behavioral thresholds.

## Conditional Stage A

Only a machine-readable terminal Stage-S approval bound to this protocol,
source, execution snapshot, launcher, and auditor may launch fresh seeds
43, 44, and 45 for every domain and arm. Every replay seed and the three-seed
mean must independently satisfy the same terminal, stability, multiplicity,
and task-quality gates. No smoke or sentinel state enters Stage A.
