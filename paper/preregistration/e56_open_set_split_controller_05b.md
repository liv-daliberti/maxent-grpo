# E56 design draft: open-set discovery with split canonical control

**Status: FROZEN FOR ONE THREE-DOMAIN SENTINEL — 2026-07-26**

The frozen treatment is `open_set_split_canonical`, seed 9010, evaluated
against the already matched E53 Dr.GRPO seed-9010 controls. The sentinel runs
50 training passes on Countdown, graph coloring, and Python factors with
K=8, four fixed evaluation draws at each quarter-pass boundary. No E55
behavioral result was used to change a dose, warmup, controller rule, or
terminal gate.

## Why E55 is not the final controller

E55 repaired two real failures: direct conditional-token entropy remains
active before a verified mode exists, and verified likelihood can anchor a
singleton bank. Its live sentinel nevertheless exposes two remaining
identifiability mismatches.

First, token entropy is not semantic entropy. A policy can retain uncertain
tokens while repeatedly producing the same validator-positive outcome.
Second, E55 uses normalized entropy among known modes as the sensor for a
uniform verified-likelihood actuator. In score coordinates, the latter has
gradient `-1/K` for every retained mode. It raises common verified score but
does not directly change relative bank scores, whereas the bank-entropy
sensor contains only relative-score information.

E56 therefore assigns discovery, verified-mass retention, and known-mode
balance to separate sensor--actuator pairs. No pair uses evaluation behavior,
gold support, a desired number of modes, or a domain-specific entropy target.

## 1. Support-independent token discovery

Retain E55's conditional-content token-entropy objective and its projection-
free inverse coefficient:

`lambda_(t+1) = lambda_0 * H_token,ref / H_token,ema,t`.

Its warmup reference contains only the model's first 64 observations. This
channel can reach a first correct behavior but is not treated as evidence of
semantic diversity.

## 2. Open-set semantic discovery

For prompt `x`, count only prior and leave-one-out current outcomes that are
loss-active and validator-positive. Let the explicit observed support be
`B_x`. Add one reserved `UNSEEN` bucket with pseudocount one. This is an
open-set posterior-predictive model, not a catalogue of valid answers.

For every eligible row `i`, let `q_-i` be the resulting distribution over
`B_x union {UNSEEN}`. When `|B_x| >= 1`, define the controller observation

`z_i = H(q_-i) / log(|B_x| + 1)`.

The normalization depends only on support already produced by the policy plus
the single structural unseen bucket. It contains no gold support count.
Groups with no previously or peer-observed successful mode are ineligible for
this observation; the existing one-time verified-discovery credit remains
responsible for retaining a first success.

The first 64 eligible observations define `z_ref`. Thereafter the semantic
coefficient is

`beta_(t+1) = beta_0 * z_ref / z_ema,t`,

with no lower or upper projection. For an eligible sampled outcome `a_i`, the
detached semantic advantage is

`A_sem,i = beta_t * (min(-log q_-i(a_i), S) -`
`                    E_q[min(-log q, S)]) / S`.

Wrong, unparseable, and loss-inactive rows receive exactly zero semantic
advantage and cannot enter the predictor. A repeatedly sampled singleton gets
negative semantic pressure because the reserved unseen bucket keeps the
predictive expectation above the singleton's realized surprise. A newly
sampled valid mode gets positive pressure. Unlike the earlier fixed and
clipped success-conditioned advantage, E56 does not project `beta`; the
vanishing predictive-entropy error and its inverse coefficient are audited
together.

## 3. Verified-mass retention

For retained validator-positive sequence scores

`s_(x,k) = |r_(x,k)|^-1 sum_t log pi(r_(x,k,t) | x, r_(x,k,<t))`,

define

`L_mass = mean_x mean_(k in B_x) -s_(x,k)`.

Its model-only sensor is the same positive verified surprisal
`c_t = L_mass.detach()`. The first 64 singleton-or-multimode replay
observations define `c_ref`; subsequently

`mu_(t+1) = mu_0 * c_ema,t / c_ref`.

Thus a rise in the model's own verified surprisal strengthens the likelihood
anchor, while an improvement weakens it. `mu` has no projection and never
reads task evaluation. The score gradient of `L_mass` sums to exactly `-1`.

## 4. Known-mode balance

For replay groups with `K >= 2`, let

`p_B = softmax(s_B)`,

`L_balance = mean_x KL(U_B || p_B)`, and

`h_t = mean_x H(p_B) / log K`.

The first 64 eligible observations define `h_ref`; subsequently

`alpha_(t+1) = alpha_0 * h_ref / h_ema,t`.

`alpha` has no projection. Unlike E55, this bank-entropy sensor multiplies the
bank-balance actuator it actually describes. The score gradient of
`L_balance` is `p_B - U_B`, sums to zero, and directly raises weak known modes
relative to dominant known modes.

## Replay measure and combined objective

One replay group remains one auxiliary pseudo-rollout beside the standard
16-sample rollout group. The combined replay term is

`(15/16) * (1/16) * (mu_t L_mass + alpha_t L_balance)`,

where `L_balance=0` and its controller is idle for singleton groups. Splitting
the mathematical roles does not double the replay sampling measure.

Mandatory telemetry includes both raw score-gradient sums (`-1` and `0`),
the shared `1/16` replay measure, each observation/reference/EMA/coefficient,
eligibility, projection-active flags fixed to zero, and separate gradient
norms before their sum.

## Information and selection firewall

Training may use only:

- sampled model tokens and model log probabilities;
- executable task reward and validator-derived canonical keys;
- counts and exact exemplars of validator-positive outcomes already sampled;
- one structural unseen bucket;
- rollout width and each controller's own warmup observations.

Training may not use:

- gold answers or the number of valid answers;
- reference-answer multiplicity or maximum support;
- neutral-evaluation pass, mean, coverage, or distinct-correct metrics;
- a desired semantic-entropy ratio or desired mode count;
- a coefficient selected from E55 behavioral outcomes.

## Engineering evidence used to freeze

The fail-closed engineering audit
`var/artifacts/e56_smoke_audit_latest.json` terminally passed both frozen
smokes with zero violations:

- graph coloring completed 16 training steps, activated replay four times,
  and exercised the open-set semantic controller;
- Python factors completed 128 training steps, activated replay nine times,
  exercised the open-set semantic controller, and crossed the direct
  controller's 64-observation warmup;
- every observed raw mass score-gradient sum was `-1`, every raw balance
  score-gradient sum was zero within tolerance, both shared the single
  `1/16` replay measure, and every projection/gold-feedback flag remained
  zero.

The smoke was a mechanics gate only. Shared evaluation output was ignored.
The frozen sentinel keeps the existing target-free final-window
distinct-minus-pass, self-retention, task-quality, and generation-safety
gates. It additionally audits all four unprojected controller recurrences,
semantic positive/negative pressure, exact split replay geometry, and exact
terminal checkpoint state.

The sentinel identity is also bound to the exact Slurm job ID and
`comparative_jobs.tsv` hash for every treatment and matched control. Both the
auditor and live parser accept only that job's `debug_job<ID>` attempt and
fail closed if another attempt appears under the same run stamp. This prevents
a canceled or requeued process from winning a "furthest trajectory" heuristic
and silently mixing attempts. Administrative `TERM`/`INT` now suppresses the
training watchdog's automatic requeue; retry remains enabled only for an
unsignaled training failure.

The live chart is deliberately not capped at the 50-pass sentinel budget:
its parser ingests every persisted checkpoint and each x-axis grows with the
latest treatment point.
