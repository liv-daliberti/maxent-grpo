# E52: direct inverse-entropy bootstrap plus fixed canonical shaping

**Status: FROZEN BEFORE SENTINEL SUBMISSION — 2026-07-26**

## Motivation

E51 established a controller-design failure across graph coloring, Countdown,
and executable Python factors. Its sensor correctly detected collapse in the
model's token-policy entropy, but its coefficient multiplied a verified
canonical-bank advantage. That advantage is exactly zero before a second
verified outcome exists, remains zero on ineligible groups, and is not scaled
in the same units as token-policy entropy. The result was a dead actuator in
Python and delayed, high-amplitude canonical impulses in graph coloring and
Countdown.

E52 separates bootstrap exploration from verified-mode balancing:

1. a projection-free inverse controller multiplies a direct, differentiable
   policy-entropy objective that remains active before any correct mode has
   been discovered; and
2. the canonical bank keeps a fixed coefficient and may only balance or reward
   modes that the policy has actually generated and the executable validator
   has accepted.

No E51 checkpoint, optimizer state, controller state, bank, target, or partial
trajectory is resumed.

## Label-free direct entropy controller

At every sampled active response state, remove the EOS column and define the
model's content distribution conditional on continuing,

`q_theta(a | s, continue) = pi_theta(a | s) / (1 - pi_theta(EOS | s))`.

For response `i`, let

`H_i = mean_t H(q_theta(. | s_it))`

over its active sampled positions, and let `H_t` be the mean of `H_i` across
response rows. The sampled state distribution is detached. Every response has
equal weight regardless of length, EOS receives no direct entropy gradient,
and earlier actions receive no reward for reaching additional states.

The actor directly minimizes

`L_direct = -lambda_t * H_t`

in addition to its ordinary Dr.GRPO loss. This is not a sampled-completion
surprisal advantage. It differentiates every non-EOS vocabulary probability
at every visited prefix and therefore remains an actuator when all sampled
responses are identical, all task rewards are zero, or a canonical bank is a
singleton.

For the first 64 observations, `lambda_t = 0.000075` and

`H_ref = mean(H_1, ..., H_64)`.

The entropy EMA is

`M_t = 0.9 M_(t-1) + 0.1 H_t`,

initialized by the first observation. After warmup,

`lambda_(t+1) = 0.000075 * H_ref / M_t`.

There is no lower projection, upper projection, Haarnoja/SAC loss, Adam state,
target mode count, target answer count, or numerical epsilon. A nonpositive
post-warmup EMA fails closed because the unbounded inverse is undefined. The
controller observes exactly `maxent_conditional_token_entropy`, the quantity
multiplied by `lambda` in the actor loss.

The reference dose `0.000075` is inherited from the pre-existing E21
length-neutral engineering contract. It is not selected from E51 outcomes,
ModeBench answer catalogues, evaluation distinctness, or any domain's number
of valid modes.

## Canonical channel

The hybrid arm independently retains E44-OGS at a fixed dose:

- bank entropy coefficient `0.10`;
- one-time verified novelty coefficient `0.50`;
- pseudocount `1`;
- surprisal clip `5`;
- prompt-local support populated only by on-policy, validator-positive
  outcomes;
- leave-one-out scoring followed by an atomic group bank update;
- canonical advantage added once after Dr.GRPO task centering.

Neither policy entropy nor any evaluation metric changes the canonical
coefficient. The direct-only arm runs the same validators and passive
discovery tracker but has canonical entropy and novelty coefficients exactly
zero.

## Information boundary

Training-time controller calibration uses only the model's own conditional
token entropy during its first 64 optimizer rounds. The controller never
receives:

- a gold list or count of valid outcomes;
- evaluation pass@K, coverage, or distinct-correct values;
- canonical-bank size, entropy, or support;
- reference-answer multiplicity;
- a domain-specific entropy target.

Executable references remain available only to the ordinary reward validator
and to admission of outcomes the policy has actually produced. Evaluation
ground truth is used only after checkpoints are written, for reporting and
the prospectively frozen stage gate below.

## Stage S: full-horizon cross-domain sentinel

Use independent engineering seed `9009` in all three domains. Each domain has
three fresh arms:

1. `grpo`: matched Dr.GRPO with passive discovery tracking;
2. `maxent_inverse`: direct inverse conditional-token entropy only;
3. `maxent_inverse_canonical`: the same direct controller plus the fixed
   canonical channel.

All arms use Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`, group size 16, learning rate
`2e-7`, one PPO epoch, `beta=0`, maximum norm 1, temperature 1, top-p 1,
responses up to 192 tokens, and exactly 50 complete prompt-pool passes.

Prompt pools are graph coloring 192, Countdown easy3 384, and executable
Python factors 384. Evaluation uses the neutral `multi_answer` split at
initialization and every quarter pass: deterministic pass@1 plus four fixed
temperature-one K=8 draws with seeds `520100--520103`.

The sentinel must span the complete 50-pass horizon because E51's largest
graph-coloring excursion appeared only after approximately 39 passes.

### Runtime and actuator gate

All nine jobs must reach their exact terminal boundary with finite loss,
entropy, alpha, gradient, response-length, and evaluation telemetry.

For both inverse arms:

- controller arithmetic must reproduce the frozen equation at every
  post-warmup observation;
- the logged objective sensor and controller observation must match;
- direct entropy loss must be finite, negative, and nonzero on every
  post-warmup update;
- no projection may be reported;
- trailing-64 mean conditional-token entropy must be at least 50% of the
  run's own warmup reference;
- no-EOS count and mean response length must remain within the larger of
  1.5 times the matched control or the matched control plus one row / 32
  tokens, respectively;
- validator/key parity failures must remain zero.

For the hybrid arm, the canonical coefficient must remain exactly `0.10` on
every update, including when the direct inverse coefficient changes.

### Behavioral gate

For each domain, over the last eight quarter-pass evaluation boundaries:

- the hybrid arm's mean distinct-correct@8 must exceed matched Dr.GRPO;
- the hybrid arm must exceed matched Dr.GRPO at least six of eight boundaries;
- hybrid terminal pass@8 and mean@8 may each trail matched Dr.GRPO by no more
  than `0.03`;
- the direct-only arm is reported separately and cannot be substituted for a
  failed hybrid arm.

Failure in any one domain blocks the matched three-seed cohort and requires a
freshly named redesign. It does not authorize coefficient selection from the
observed evaluation curves.

## Conditional Stage A

Only a machine-readable positive Stage-S approval whose protocol, source, and
execution-surface hashes still match may authorize Stage A. Stage A repeats
the same three arms and settings at seeds 43, 44, and 45 across all three
domains. No Stage-S optimizer state, entropy reference, bank, or model
checkpoint enters Stage A.

The final objective is not merely absence of crashes: the hybrid must retain
consistently higher mean distinct-correct@8 across seeds and time in all three
domains while respecting the pass@8, mean@8, length, EOS, and validator
guardrails above.
