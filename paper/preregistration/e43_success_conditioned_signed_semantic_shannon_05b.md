# E43: success-conditioned signed semantic-Shannon advantage at 0.5B

**Status: FROZEN BEFORE LAUNCH (2026-07-23).**

## Motivation and status

E41 preserves a signed semantic-Shannon advantage outside Dr.GRPO's empirical
task-reward centering, but its predictor and pressure include incorrect
answers. E42 conditions the predictor and history on verified successes and
gives wrong answers exactly zero pressure, but its positive-only transform
removes negative pressure from common successful outcomes. That safety rule
can make the semantic actuator too sparse to sustain breadth.

E43 isolates the next mechanism change. It retains E42's successful-outcome
predictor and zero pressure for every wrong, unparseable, or loss-inactive
row. Among eligible successful answers, it restores E41's signed predictive
advantage and clips it symmetrically to `[-0.05, 0.05]`. Rare successes can
receive positive pressure and overrepresented successes can receive negative
pressure, while arbitrary wrong MATH answers can receive neither.

Everything else matches E41 and E42: model, data, prompts, seeds, group size,
optimizer, ten-pass budget, resources, checkpoints, and neutral evaluations.
E43 is an exploratory mechanism ablation and launches one new treatment arm.
The frozen E37, E38, E39, E41, and E42 trajectories are reused.

## Frozen successful-outcome predictor

For prompt `x`, let `n_x(a)` count only parseable outcomes from completed
earlier rows that had positive task reward and were active in the policy loss.
For current row `i` in a rollout group of size `G=16`, let `m_-i(a)` count
only leave-one-out peers satisfying those same eligibility conditions.

The detached row-specific predictive distribution `q_i` has:

- explicit support equal to the union of eligible successful historical
  outcomes and eligible successful leave-one-out peer outcomes;
- numerator `n_x(a) + m_-i(a) + alpha` for each explicit outcome;
- one reserved unseen bucket with numerator `alpha`;
- pseudocount `alpha=1`; and
- denominator equal to the sum of those numerators.

Ineligible rows are not scored by the predictive mechanism and never enter
current support or persistent history. Eligible current outcomes are added to
history only after every row in the group has been scored. An all-wrong group
therefore produces exact-zero semantic advantages and leaves semantic history
unchanged.

The stable prompt identity is the SHA256 digest of unpadded prompt token IDs.
The successful-outcome history and signed-mode schema are stored in
optimizer-resumable checkpoints and restored exactly. A resume with a
different signed cap fails closed.

## Frozen signed semantic advantage

For row `i`, define clipped realized surprise

`s_i = min(-log q_i(a_i), 5.0)`

and detached predictive clipped-surprise expectation

`h_i = sum_a q_i(a) * min(-log q_i(a), 5.0)`,

where the sum includes every explicit successful outcome and the unseen
bucket. The raw semantic advantage is

`A_i_raw = (0.10 / 5.0) * (s_i - h_i)`.

Before eligibility and clipping, this is centered under the row's predictive
distribution: `E_{a ~ q_i}[A_i_raw]=0`.

Define

`e_i = 1[row i is loss-active, has positive task reward, and has a parseable answer key]`.

E43's effective semantic advantage is

`A_i_E43-sem = e_i * max(-0.05, min(A_i_raw, 0.05))`.

Consequently:

- wrong, unparseable, and loss-inactive rows receive exactly zero;
- rare eligible successes may receive at most `+0.05`;
- common eligible successes may receive at least `-0.05`; and
- the semantic mechanism cannot reward randomness among failures.

The symmetric clip is applied after raw predictive centering; E43 does not
claim the clipped effective values remain exactly predictive-mean-zero.

Ordinary Dr.GRPO task rewards are centered exactly as in E41 and E42:

`A_i_task = r_i - mean_j(r_j)`.

The policy-gradient advantage is

`A_i_E43 = A_i_task + A_i_E43-sem`.

No second empirical group centering is applied. The task rewards sent into
Dr.GRPO centering are unchanged. Because a correct-versus-incorrect
task-advantage difference is one and only the correct row can receive a
semantic term of magnitude at most `0.05`, semantic pressure cannot reverse
correctness ordering.

The new arm and `OAT_ZERO_VARIANT` are both
`success_conditioned_signed_semantic_shannon`. The launcher pins:

- `OAT_ZERO_SEMANTIC_SHANNON_COEF=0.10`;
- `OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP=5.0`;
- `OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT=1.0`;
- `OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE=1`;
- `OAT_ZERO_SEMANTIC_SHANNON_QUALITY_GATED_ADVANTAGE=0`;
- `OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_ADVANTAGE=1`; and
- `OAT_ZERO_SEMANTIC_SHANNON_SUCCESS_CONDITIONED_SIGNED_CAP=0.05`.

All other diversity mechanisms are disabled: outcome collision, DIAYN,
direct token entropy, fixed or controlled MaxEnt, SEED, xDr aggregation, and
canonical actions. Every older semantic-Shannon variant explicitly pins the
new signed mode off.

## Frozen domains and training contract

The nine fresh E43 runs comprise:

1. Countdown easy3, seeds `43, 44, 45`;
2. graph coloring `multi_answer`, seeds `43, 44, 45`; and
3. MATH12K-384 rows 0 through 383, seeds `43, 44, 45`.

All runs match E41 and E42 on:

- Qwen2.5-0.5B-Instruct revision
  `7ae557604adf67be50417f59c2c2f167def9a775`;
- group size 16 and ten complete prompt passes;
- learning rate `2e-7`, one PPO epoch, `beta=0`, and gradient norm 1;
- rollout temperature 1, `top_p=1`, and neutral free-form prompts;
- one GPU, colocated vLLM and learner, and learner microbatch one; and
- one optimizer-resumable checkpoint per prompt pass, newest two retained,
  watchdog requeue, and exact semantic-history restoration.

Countdown and graph coloring use the exact E37/E38/E41/E42 data and
32-GB/four-CPU A5000-class request. MATH uses the exact E39/E41/E42
MATH12K-384 materialization, full MATH-500 evaluation, and
64-GB/eight-CPU A5000-class request.

## Frozen neutral evaluations

Countdown and graph coloring use ordinary `qwen_boxed` requests, greedy
pass@1, and four independent K=8 temperature-one draws with seeds
`370100, 370101, 370102, 370103`, evaluated every quarter pass. Within each
task and seed, all E43 step-zero scalar metrics and raw outcomes must agree
exactly with E41 and E42.

MATH uses ordinary `qwen_math` requests over all 500 MATH-500 rows, greedy
pass@1, and one deterministic K=8 draw with seed `390100`. Evaluations occur
at initialization and passes 2, 4, 6, 8, and 10, without a duplicate terminal
evaluation. Within each seed, all E43 MATH step-zero scalar metrics and raw
outcomes must agree exactly with E39, E41, and E42.

## Identity, held audit, and release

The analytical prefixes are:

- `cde43_success_conditioned_signed_semantic_shannon_05b_v1`;
- `gce43_success_conditioned_signed_semantic_shannon_05b_v1`; and
- `mte43_math12k_384_success_conditioned_signed_semantic_shannon_05b_v1`.

The launcher audits and hashes the frozen E37, E38, E39, E41, and E42
identities and manifests. It audits the E39 MATH materialization, freezes
source and execution snapshots, and records protocol, launcher, source,
execution, data, and comparator hashes in
`e43_success_conditioned_signed_semantic_shannon_05b_v1_identity.json`.

All nine jobs are submitted held. Before any release, the launcher verifies
the three tasks times three seeds, exact E43 arm and variant, success
conditioning, symmetric cap, coefficient, surprise clip, pseudocount, every
disabled mechanism, data, prompts, lengths, evaluation schedules,
checkpoint/recovery fields, and task-specific placement. A partial or failed
cohort is cancelled and its manifests and identity are quarantined with a
failure timestamp. Only a complete audited nine-job cohort is released.

No E43 outcome may alter eligibility, sign, cap, coefficient, predictive
support, data, cadence, seed, endpoint, or evaluation request.

## Reporting and interpretation

Report E43 head-to-head with E41 and the retained Dr.GRPO, E37 collision, E38
predictive-Shannon, and E39 MATH trajectories. E42's frozen artifacts remain
available as provenance for the motivating negative ablation, but, by the
retirement decision made before E43 launch, E42 is excluded from the active
monitor and comparison figure. Primary quality endpoints are neutral pass@8
and mean@8, guarded by greedy pass@1. Countdown and graph coloring
additionally report semantic coverage and distinct-correct@8.

Mandatory mechanism telemetry includes eligible, gated, active,
reward-positive, and parseable fractions; raw-eligible and effective signed
semantic advantage mean/min/max/RMS and positive/negative/zero fractions;
positive-cap and negative-cap fractions; predictor normalization and
centering errors; history rows and groups updated/skipped; and
semantic-to-task advantage RMS.

On MATH, E43 is a safety and quality stress test. Distinct incorrect final
answers are excluded from both learning pressure and predictor history. A
positive MATH result cannot by itself establish diverse correct proof
strategies.
