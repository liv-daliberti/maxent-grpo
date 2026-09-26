# E14 prospective engineering gate: canonical graph actions

**Status: EXPLORATORY ENGINEERING GATE, FROZEN BEFORE E14 OUTCOMES
(2026-07-18).** E11--E13 showed that free-text sequence entropy can be spent
on response length and EOS avoidance rather than on distinct graph-coloring
answers. E14 removes that confound. It changes the graph-coloring response
space itself to a finite canonical action space and asks whether direct MaxEnt
preserves useful answer diversity there. E14 is not evidence about the
free-text method, model scaling, or Countdown.

**Prospective amendment A1: behavior-policy interface (2026-07-18).** The
first cross-engine runtime probe (job 30010491) stopped before its first
optimizer update because selected-token log probabilities from vLLM and the
learner differed by more than the original `5e-3` equality tolerance. It
produced no E14 training outcome. That tolerance incorrectly treated two
numerical implementations of the same restricted policy as if they had to be
bitwise-near replicas. Before retrying the probe, A1 replaces that condition
with an explicit behavior-policy correction and a frozen full-distribution
overlap gate below. This is a prospective engineering amendment, not a
response to an E14 result.

**Prospective amendment A2: learner-side behavior sampling (2026-07-18).**
The A1 cross-engine probe (job 30010680) captured the complete vLLM behavior
distribution and used it in the learner denominators, but failed the frozen
overlap gate: its all-action `q_learner/q_b` range was
`[0.787985, 1.376587]`, outside `[0.8, 1.2]`. Its maximum TV (`0.041378`),
bidirectional KL maxima (`0.006143` and `0.006649` nats), and importance-weight
ESS passed their separate thresholds; those successes cannot override the
failed all-action condition. The probe performed only the prescribed
zero-learning update at learning rate zero and produced no learned E14
outcome. A2 therefore invokes A1's preregistered fallback before C0 or any
treatment is run. It does not relax or select among thresholds after seeing a
scientific result.

All subsequent E14 training rollouts must be sampled autoregressively from the
restricted three-action softmax of the Hugging Face learner itself, under
`torch.no_grad()`, with the complete behavior log-probability row captured at
each realized prefix before sampling the next action. The same learner model,
weights, tokenizer IDs, temperature, and restricted-softmax implementation
then evaluate those rows for PPO and the entropy estimator. vLLM may remain
allocated for the maintained evaluation path, but it is not the E14 training
behavior sampler. Metrics retain the `actor/` namespace for compatibility
with OAT's logging interface; `actor/canonical_sampler_learner=1` explicitly
means those rollout metrics came from the learner-side HF sampler.

Every preflight, C0, M01, and M05 run must set
`canonical_graph_learner_sampling=true`, record the exact training-launch log
field `learner_sampling=1`, and emit `actor/canonical_sampler_learner=1` on
every training telemetry row. A missing, false, or conflicting marker fails
closed. The A1 full-distribution normalization, ratio, TV, KL, ESS, and
actor-denominator conditions remain unchanged. If same-engine sampling fails
any of those conditions, E14 stops; cross-engine sampling is not restored and
the gates are not weakened.

**Prospective amendment A3: fixed-shape same-HF execution (2026-07-18).** The
A2 learner-side probe (job 30010722) used the same Hugging Face learner and
captured its complete restricted behavior distribution, but still failed the
unchanged all-action overlap gate. Its `q_learner/q_b` range was
`[0.939986, 1.227925]`, so the upper endpoint exceeded `1.2`. Its maximum TV
(`0.033191`), bidirectional KL maxima (`0.003293` and `0.003441` nats),
full-sequence ESS (`0.992579`), and minimum prefix ESS (`0.993004`) all passed
their separate thresholds; they cannot override the failed ratio condition.
The probe completed only the prescribed learning-rate-zero accounting update,
which left the model weights unchanged, and produced no learned E14 outcome.

The most direct remaining execution discrepancy was layout: behavior rows
were sampled with growing prefix lengths, whereas the learner-old rows were
recomputed by fixed-length teacher forcing. Even with identical weights and
software, those shapes can select different reduced-precision attention or
matrix kernels. A3 prospectively removes that numerical confound; the failed
probe by itself does not establish that layout caused the mismatch. A3 changes
neither the canonical policy, objective, random uniforms, model weights,
dataset, optimizer, nor any overlap threshold.

For all later E14 runs, the same HF learner must sample with a fixed total
sequence shape. Each rollout microbatch appends three copies of a fixed token
from the allowed support as causal placeholders. Its attention mask is one at
all prompt and action-slot positions. At action step `t`, the implementation
reads the restricted next-token distribution at the realized prefix, stores
the complete ordered three-way row, samples with the preregistered uniform,
and replaces only slot `t`; the causal mask prevents every future placeholder
from affecting that distribution. Zero-attention padding of the action slots
is prohibited because it can select a different SDPA path from teacher
forcing. Behavior sampling and learner-old scoring must both run in evaluation
mode and use identical microbatch shape, grouping, and row order.

Every subsequent preflight, C0, M01, and M05 run must additionally set
`canonical_graph_fixed_shape_sampling=true`, record launch field
`fixed_shape_sampling=1`, and emit
`actor/canonical_sampler_fixed_shape=1` on every training telemetry row. The
frozen runtime identity must record exactly
`canonical_training_sampler=learner_hf_restricted_inverse_cdf_fixed_shape_causal_placeholder`.
A missing or conflicting marker, a masked-out action placeholder, or failure
of any unchanged A1 overlap condition stops E14 before C0.

## Canonical policy and objective

Every E14 training and evaluation prompt has exactly three hidden graph nodes.
In their fixed prompt order, the action is

```text
A = (c_1, c_2, c_3) in {1, 2, 3}^3.
```

The model emits exactly the three bare digit tokens, with no explanation,
box, delimiter, or EOS action. Thus the syntactically valid action space has
`3^3 = 27` members. The ordinary graph grader reconstructs the full coloring
and assigns binary correctness; the semantically correct actions are a
prompt-dependent subset of those 27 actions.

The frozen 192-row training and 96-row `multi_answer` evaluation pools have
semantic content hash
`8e8bd8d4986920784cb067f99f4dc1b401f553b0d40a378dd4ada5dab29b48d6`.
The preflight validates every row and rejects any alternate content, even if
its shape and schema match.

Let `D={1,2,3}` and let `z_d` be the model logit of digit token `d`. Both the
behavior policy and the learner policy are restricted distributions of
the form

```text
q_theta(d | x, a_<t)
    = exp(z_d) / sum_{d' in D} exp(z_d'),       t in {1,2,3}.
```

This restriction is part of the policy, not a post-hoc mask on sampled
free-text responses. The sampler's realized three-way distribution is denoted
`q_b` and is the behavior policy actually responsible for each rollout. For
every realized prefix, the behavior sampler stores the complete ordered vector
`(q_b(1), q_b(2), q_b(3))`, represented in the trajectory by all three log
probabilities, not only the sampled token's log probability. The learner
evaluates `q_new` on the same prefixes and uses `q_b` in every
importance-ratio denominator. Thus the selected-action PPO ratio is
`q_new(A_t)/q_b(A_t)`, while the entropy estimator uses products of those
same behavior-corrected ratios. A separately recomputed learner-side old
policy may be logged as a diagnostic, but it is not allowed to replace the
actual behavior distribution in a denominator.

Behavior sampling, stored behavior probabilities, current-policy PPO
probabilities, and local categorical entropy all use a three-way softmax on
the same ordered token IDs. Logits outside `D` have neither probability mass
nor gradient through the E14 objective. Each digit must resolve to one
distinct tokenizer token before a run can start.

The installed vLLM 0.8.4 V1 engine reports requested log probabilities from
raw logits before `allowed_token_ids` and temperature, whereas its V0 engine
reports the logits actually used for sampling after those transformations.
E14 therefore pins and records `VLLM_USE_V1=0` before vLLM is imported. Using
V1, leaving the engine mode implicit, or changing vLLM versions invalidates
the implementation gate.

The tokenizer is likewise frozen to Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`; the preflight records and checks
its tokenizer-file and vocabulary hashes rather than following a mutable
cache `main` reference silently.

Training loads that exact local snapshot in offline mode. Its
`model.safetensors` SHA-256 is
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`;
both actor and learner therefore use the model identity validated before
submission rather than resolving a mutable Hub identifier at allocation time.

The E14 treatment objective is

```text
J(theta) = E_x [ E_{A ~ q_theta(. | x)} R(x,A)
                 + alpha H(q_theta(. | x)) ],
```

where

```text
H(q_theta(. | x))
    = E_{A ~ q_theta} sum_{t=1}^3
        H(q_theta(. | x, A_<t))
    = -sum_{a in D^3} q_theta(a | x) log q_theta(a | x).
```

Consequently `0 <= H(q) <= log(27) = 3.295836866004329` nats. There is no
response-length objective, EOS term, full-vocabulary entropy, answer-span
heuristic, or length controller in E14. A generation termination marked
`length` is expected after the third action token; any row with fewer or more
than three tokens, or with a token outside `D`, is an implementation failure
and must fail closed.

Rollouts come from `q_b`. For learner policy `q_new`, define the exclusive
prefix ratio and local restricted entropy

```text
W_(t-1) = q_new(A_<t | x) / q_b(A_<t | x),
h_t     = H(q_new(. | x, A_<t)).
```

The exact finite-horizon identity is

```text
H(q_new(. | x))
    = E_{A ~ q_b} sum_{t=1}^3 W_(t-1) h_t.
```

E14 uses the unclipped prefix ratios for this three-action entropy estimator;
reward retains the ordinary clipped PPO ratio. The historical Dr.GRPO outer
normalization constant remains `T_scale=192` solely to keep update magnitudes
comparable to the preceding graph experiments. It is not the E14 action
horizon. Reward and raw action entropy receive this shared outer normalization
exactly once. With group size `G` and `c=(G-1)/G`, the maintained actor loss is

```text
L_actor = L_Dr.GRPO - c * alpha * H_hat(q_new) / T_scale.
```

There is no division by the three-action horizon and no second division by
`T_scale`.

## Preflight and no-learning gate

No training arm may be submitted until maintained tests establish all of the
following:

- the graph training and `multi_answer` evaluation rows each contain exactly
  three hidden nodes, and the canonical prompt requests exactly three digits;
- the three actions tokenize one-to-one to distinct single token IDs;
- restricted log probabilities and entropy agree with direct three-way
  calculation, active labels outside the support fail, and disallowed logits
  have zero gradient;
- exact enumeration of a distinct-old/distinct-new autoregressive toy policy
  reproduces both the value and gradient of the exclusive-prefix entropy
  identity, including first-token ratio one and response masking;
- `alpha=0` makes the auxiliary loss and gradient exactly zero and reproduces
  the constrained Dr.GRPO learner loss;
- a short zero-learning runtime probe emits only three supported tokens and
  carries all three behavior probabilities for all `16*3=48` realized
  prefix rows into the learner;
- every behavior row has support exactly three and sums to one with
  maximum absolute normalization error
  `|logsumexp_d(log q_b(d))|` at most `1e-6`;
- before the update, every one of the 144 full-distribution action ratios
  `q_learner(d|prefix)/q_b(d|prefix)` lies in the reward PPO trust interval
  `[0.8, 1.2]`;
- the maximum per-row total-variation distance is at most `0.05`, and both
  maximum per-row KL divergences, `KL(q_b || q_learner)` and
  `KL(q_learner || q_b)`, are at most `0.01` nats;
- both the full-sequence importance weights and the exclusive-prefix weights
  retain an effective-sample-size fraction of at least `0.90`;
- the runtime records `learner_sampling=1`, `fixed_shape_sampling=1`,
  `actor/canonical_sampler_learner=1`, and
  `actor/canonical_sampler_fixed_shape=1`; vLLM 0.8.4 remains V0 for the
  maintained evaluation path, while V1 remains prohibited. The frozen runtime
  identity separately records
  `canonical_training_sampler=learner_hf_restricted_inverse_cdf_fixed_shape_causal_placeholder`
  and `vllm_role=evaluation_only` so an evaluation-engine identity cannot be
  mistaken for the behavior sampler;
- the fixed-shape sampler and learner-old scorer both use evaluation mode and
  exactly the same microbatch shape, grouping, and row order; the three action
  slots have an all-ones attention mask, future support-token placeholders are
  excluded only by the causal mask, and zero-attention action padding fails
  closed.

For the overlap diagnostics, `TV(p,q)=0.5 sum_d |p_d-q_d|`. The KL directions
are computed from the complete three-action rows, in float64, and reported as
both means and maxima. For weights `w_1,...,w_N`, the ESS fraction is
`(sum_i w_i)^2 / (N sum_i w_i^2)`. The sequence diagnostic uses
`W_3=prod_{t=1}^3 q_learner(A_t)/q_b(A_t)` across the 16 rollouts. The prefix
diagnostic is the minimum across prefix lengths zero, one, and two of the ESS
fraction of `W_0`, `W_1`, and `W_2`; these are exactly the exclusive-prefix
weights used by the entropy identity. The ratio, TV, KL, and ESS thresholds
are an initial same-weights overlap gate before any optimizer update. They do
not clip the entropy estimator or assert that the two inference engines are
numerically identical.

A1 specified that failure of this strict full-distribution overlap gate would
require learner-side sampling so sampling and probability evaluation occur in
the same engine. A2 records that this condition occurred and makes that
fallback mandatory. A3 records the remaining execution-shape failure and
requires fixed-shape causal-placeholder sampling before another preflight.
The tolerances must not be weakened post hoc to admit either failed probe.

The runtime probe must also record the resolved action token IDs, tokenizer
identity, dataset hash, source hash, action count, and maximum possible action
entropy. A failed identity or mismatch is an implementation failure, not a
scientific result.

## Sequential smoke

All E14 arms use graph coloring, Qwen2.5-0.5B-Instruct, the exact-answer
`multi_answer` data with three hidden nodes, seed 9005, group size 16,
four-row backward microbatches, one PPO epoch, reward PPO clip 0.2, 128
optimizer updates, and evaluations every 32 prompt updates. Each arm must use
an immutable source snapshot and the same tokenizer, prompt pool, evaluation
set, and outer normalization.

Automatic checkpoint resume and watchdog requeue are disabled for these
short gates. OAT checks its strict trajectory-query budget after an update;
resuming a checkpoint that has already crossed that budget could therefore
execute one unintended additional update. A failed allocation is resubmitted
from initialization under a new immutable stamp and is never spliced into a
nominal E14 run.

OAT's `max_queries` counter counts the 16 sampled trajectories, not prompt
updates, and stops only after the counter is strictly greater than its budget.
Accordingly the 128-update arms use `max_queries=(128-1)*16=2032` (and
`max_train>=2032` to prevent argument normalization from clipping that
budget). The one-update zero-learning probe uses budget 1.

The first training arm is **C0**, constrained Dr.GRPO with `alpha=0`. It is
run alone. C0 passes only if:

- it reaches the frozen step-128 endpoint and has contiguous finite telemetry
  for updates 97--128;
- all rollout rows contain exactly three supported actions, with zero invalid
  rows, and carry normalized three-action behavior distributions used in all
  importance-ratio denominators;
- the final-32 mean rollout reward is positive and final exact valid-action
  probability on the evaluation set exceeds `0.05`;
- the endpoint enumeration audit below passes for every evaluation prompt;
- no MaxEnt controller, length controller, EOS penalty, or free-text entropy
  telemetry is active.

Failure of C0 stops E14. In particular, the MaxEnt treatments cannot be used
to repair a broken constrained control.

Only a passing C0 authorizes two fixed-coefficient treatment smokes:

| Arm | Fixed `alpha` | Maximum per-prompt entropy bonus |
|---|---:|---:|
| M01 | `0.01` | `0.01 log(27) = 0.0329584` |
| M05 | `0.05` | `0.05 log(27) = 0.164792` |

M01 and M05 differ from C0 only in the fixed coefficient and active canonical
entropy loss. They are a single-seed engineering calibration, not comparative
evidence. Proportional feedback, Haarnoja dual control, and coefficient
adaptation of any kind are prohibited in E14.

## Exact 27-action endpoint audit

At optimizer update 128, OAT writes the audit target as
`saved_models/step_00128`. Its forced terminal save is named `step_00129` but
contains no additional optimizer update; it is a duplicate endpoint alias and
is excluded from updates 97--128 telemetry. The audit uses `step_00128`.
Every one of the 27 action vectors is teacher-forced for every evaluation
prompt under the restricted policy. The audit records each leaf's log
probability and grader result and computes

```text
P_valid(x) = sum_a q(a | x) R(x,a),
q_plus(a | x) = q(a | x) R(x,a) / P_valid(x),
H_valid(x) = -sum_{a:R(x,a)=1} q_plus(a | x) log q_plus(a | x),
N_eff_valid(x) = exp(H_valid(x)).
```

The data constructor must guarantee at least one correct action for every
prompt. Because the restricted softmax gives every finite-logit action
positive mass, this makes `P_valid>0` in exact arithmetic; the audit computes
the valid normalization with log-sum-exp and fails on a nonfinite result. For
every prompt, enumerated probabilities must sum to one within `1e-5`, leaf
entropy must agree with the conditional-entropy identity within `1e-5` nats,
and teacher-forced selected-action log probabilities must agree with the
same-learner tree calculation within `5e-3`. This same-engine enumeration
tolerance is distinct from the retired cross-engine actor/learner equality
gate. The audit reports evaluation means
for exact action entropy, `P_valid`, `H_valid`, and `N_eff_valid`, plus the
number of semantically correct actions. Sampled pass@1, pass@8, and
coverage@8 remain descriptive Monte Carlo diagnostics; they cannot override
the exact audit.

## Treatment decision rule

M01 or M05 is **runtime-valid** only if it meets every C0 integrity condition,
uses its assigned fixed coefficient on every update, has canonical action
entropy in `[0, log(27)]`, and contains no adaptive or length-controller
telemetry.

A runtime-valid treatment is **behaviorally safe** only if its endpoint mean
exact `P_valid` is both greater than `0.05` and at least 80% of C0's endpoint
mean exact `P_valid`, and its final-32 mean rollout reward is positive. It is
**diversity-effective** only if, relative to C0 on the same evaluation
prompts, both:

- mean exact action entropy increases by at least `log(1.25) = 0.2231436`
  nats; and
- mean exact valid-mode effective support `N_eff_valid` increases by at least
  25%.

Requiring valid-mode support prevents entropy spent only on incorrect actions
from counting as success. Peak entropy, sampled coverage, or a successful
subset of prompts cannot rescue a failed endpoint gate.

An arm is viable only if it is runtime-valid, behaviorally safe, and
diversity-effective. If both arms are viable, choose the arm with the larger
mean exact `N_eff_valid`; values within 5% are tied and select the smaller
coefficient. If neither arm is viable, E14 selects no canonical MaxEnt dose:
low entropy is an actuation failure, while higher entropy with failed
`P_valid` or valid-mode support is an allocation failure.

## Scope firewall

Before C0 and at least one fixed treatment pass their respective gates, no
E14 job may be launched for 3B, 7B, Countdown, another seed, an adaptive
controller, or the analytical campaign grid. A viable fixed treatment only
authorizes drafting and reviewing a separate three-seed 0.5B graph-coloring
replication protocol. It does not automatically submit that replication, and
even a successful replication would require a separately frozen protocol
before any scale or domain expansion.

## Retrospective outcome (written after all E14 endpoints)

This section is an outcome record, not a prospective amendment. It does not
alter the frozen gates above. C0, M01, and M05 completed 128 optimizer updates
at seed 9005, and their exact 27-action `step_00128` audits passed:

| Arm | Train / audit job | Exact $H(A)$ | Exact $P_{\mathrm{valid}}$ | Exact $H_{\mathrm{valid}}$ | Exact $N_{\mathrm{eff,valid}}$ | Final-32 reward |
|---|---|---:|---:|---:|---:|---:|
| C0 | 30010773 / 30010861 | 1.3450209681 | 0.3140740167 | 0.7472495017 | 2.3263799784 | 0.3300781250 |
| M01 | 30010871 / 30010874 | 1.4282535041 | 0.3100185289 | 0.7946474626 | 2.4422400374 | 0.3183593750 |
| M05 | 30010870 / 30010875 | 1.8170387683 | 0.2957824297 | 0.9317972903 | 2.7749407825 | 0.2929687500 |

Both treatments were runtime-valid and behaviorally safe. M01 retained
98.7087% of C0's mean exact valid probability, but its entropy gain
(`0.0832325360` nats) was below `log(1.25)` and its valid-support ratio
(`1.0498027236`) was below `1.25`. M05 retained 94.1760% of C0's mean exact
valid probability and passed the entropy gate with a gain of `0.4720178003`
nats. It missed only the valid-support gate: its ratio was `1.1928149349`, or
a 19.2815% gain, 5.7185 percentage points below the required 25%. M05 is thus
a near miss descriptively, but a failure under the frozen binary decision
rule.

The final decision is **no viable canonical MaxEnt dose**. No arm is selected.
Because this is a single-seed 0.5B engineering calibration, and because no arm
passed the complete rule, it does not authorize another seed, larger model,
Countdown run, adaptive controller, analytical-grid expansion, or main-paper
claim. The machine-readable retrospective record is
`paper/results/e14_canonical_graph_actions.json`.
