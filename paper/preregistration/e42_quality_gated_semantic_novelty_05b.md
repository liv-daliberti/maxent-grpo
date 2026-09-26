# E42: quality-gated semantic novelty at 0.5B

**Status: FROZEN BEFORE LAUNCH (2026-07-23).**

## Motivation and status

E41 adds a separately centered predictive semantic-Shannon advantage after
ordinary Dr.GRPO task-reward centering. That placement preserves semantic
pressure when a sampled group collapses, but it also exposes a failure mode:
novel wrong answers can receive positive semantic advantage. This is
especially undesirable on single-answer MATH, where unconstrained incorrect
hypotheses are easy to make distinct.

E42 changes only the eligibility and ceiling of E41's semantic term. It uses
the same model, prompts, data, seeds, rollout group, optimizer, ten-pass
budget, checkpoints, resources, and neutral evaluations. The predictive
history and current leave-one-out support contain verified successful,
parseable, loss-active answers only. The separately computed semantic term
can only add a nonnegative, capped boost to an eligible answer. Wrong,
unparseable, or loss-inactive rows receive exactly zero semantic pressure and
cannot alter semantic history.

E42 is an exploratory mechanism ablation selected after inspecting E41. It
launches one new treatment arm and reuses the frozen E37, E38, E39, and E41
trajectories as comparators.

## Frozen successful-outcome predictor

For prompt `x`, let `n_x(a)` count only parseable outcomes from completed
earlier rows that had positive task reward and were active in the policy loss.
For current row `i` in a group of size `G=16`, let `m_-i(a)` count only
leave-one-out peers satisfying those same three eligibility conditions.

The detached row-specific predictive distribution `q_i` has:

- explicit support equal to the union of successful historical outcomes and
  successful eligible leave-one-out peer outcomes;
- numerator `n_x(a) + m_-i(a) + alpha` for each explicit outcome;
- one reserved unseen bucket with numerator `alpha`;
- pseudocount `alpha=1`; and
- denominator equal to the sum of those numerators.

An ineligible row is queried against this distribution only for diagnostic
telemetry. It is not added to support or history. After every row in a group
has been scored, eligible current outcomes are added to history. An all-wrong
group therefore produces exactly zero effective semantic advantages and
leaves semantic history unchanged.

The stable prompt identity is the SHA256 digest of the unpadded prompt token
IDs. Predictor history and its quality-gated schema are included in
optimizer-resumable checkpoints and restored exactly. A resume whose saved
quality-gate cap differs from the frozen cap fails closed.

## Frozen raw and effective semantic advantages

For row `i`, define clipped realized surprise

`s_i = min(-log q_i(a_i), 5.0)`

and the predictive clipped-surprise expectation

`h_i = sum_a q_i(a) * min(-log q_i(a), 5.0)`,

where the sum includes every explicit successful outcome and the unseen
bucket. The detached raw E41-style semantic advantage is

`A_i_raw = (0.10 / 5.0) * (s_i - h_i)`.

Define frozen eligibility

`e_i = 1[row i is loss-active, has positive task reward, and has a parseable answer key]`.

E42's effective semantic novelty advantage is

`A_i_E42-sem = e_i * min(max(A_i_raw, 0), 0.05)`.

Thus the effective semantic term is always in `[0, 0.05]`. Negative raw
surprise never penalizes a common correct answer. A novel failure, including
a parseable but wrong MATH answer, receives exactly zero. The cap limits the
largest possible novelty preference to five percent of the unit
correctness-reward scale.

Ordinary Dr.GRPO task rewards are centered exactly as in E41:

`A_i_task = r_i - mean_j(r_j)`.

The policy-gradient advantage is

`A_i_E42 = A_i_task + A_i_E42-sem`.

No second empirical group centering is applied. The task rewards sent to
Dr.GRPO centering are bitwise unchanged by the semantic mechanism.

The new arm and `OAT_ZERO_VARIANT` are both
`quality_gated_semantic_novelty`. The launcher pins:

- `OAT_ZERO_SEMANTIC_SHANNON_COEF=0.10`;
- `OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP=5.0`;
- `OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT=1.0`;
- `OAT_ZERO_SEMANTIC_SHANNON_SEPARATE_ADVANTAGE=1`;
- `OAT_ZERO_SEMANTIC_SHANNON_QUALITY_GATED_ADVANTAGE=1`; and
- `OAT_ZERO_SEMANTIC_SHANNON_QUALITY_GATED_CAP=0.05`.

All other diversity mechanisms are disabled: outcome collision, DIAYN,
direct token entropy, fixed or controlled MaxEnt, SEED, xDr aggregation, and
canonical actions. Older semantic-Shannon variants explicitly pin the
quality gate off.

## Frozen domains and training contract

The nine fresh E42 runs comprise:

1. Countdown easy3, seeds `43, 44, 45`;
2. graph coloring `multi_answer`, seeds `43, 44, 45`; and
3. MATH12K-384 rows 0 through 383, seeds `43, 44, 45`.

Every run matches E41 on:

- Qwen2.5-0.5B-Instruct revision
  `7ae557604adf67be50417f59c2c2f167def9a775`;
- group size 16 and ten complete prompt passes;
- learning rate `2e-7`, one PPO epoch, `beta=0`, and gradient norm 1;
- rollout temperature 1, `top_p=1`, and neutral free-form prompts;
- one GPU, colocated vLLM and learner, and learner microbatch one; and
- one optimizer-resumable checkpoint per prompt pass, newest two retained,
  watchdog requeue, and exact semantic-history restoration.

Countdown and graph coloring use E41's exact E37/E38 data and
32-GB/four-CPU A5000-class request. MATH uses the same E39 materialization,
MATH12K-384 training boundary, full MATH-500 evaluation, and
64-GB/eight-CPU A5000-class request.

## Frozen neutral evaluations

Countdown and graph coloring use ordinary `qwen_boxed` requests, greedy
pass@1, and four independent K=8 temperature-one draws with seeds
`370100, 370101, 370102, 370103`, evaluated every quarter pass. Within each
task and seed, all E42 step-zero scalar metrics and raw outcomes must agree
exactly with E41.

MATH uses ordinary `qwen_math` requests over all 500 MATH-500 rows, greedy
pass@1, and one deterministic K=8 draw with seed `390100`. Evaluations occur
at initialization and passes 2, 4, 6, 8, and 10, with no duplicate terminal
evaluation. Within each seed, all E42 MATH step-zero scalar metrics and raw
outcomes must agree exactly with E41 and E39.

## Identity, held audit, and release

The analytical prefixes are:

- `cde42_quality_gated_semantic_novelty_05b_v1`;
- `gce42_quality_gated_semantic_novelty_05b_v1`; and
- `mte42_math12k_384_quality_gated_semantic_novelty_05b_v1`.

The launcher audits and hashes the frozen E37, E38, E39, and E41 identities
and manifests. It audits the E39 MATH materialization; freezes source and
execution snapshots; and records protocol, launcher, source, execution,
data, and comparator hashes in
`e42_quality_gated_semantic_novelty_05b_v1_identity.json`.

All nine jobs are submitted held. Before any release, the launcher verifies
the three tasks times three seeds, the exact E42 arm and variant, gate and cap,
coefficient, surprise clip, pseudocount, all disabled mechanisms, data,
prompts, lengths, evaluation schedules, checkpoint/recovery fields, and
task-specific placement resources. A partial or failed cohort is cancelled
and its manifests and identity are quarantined with a failure timestamp.
Only a complete audited nine-job cohort is released.

No E42 outcome may alter eligibility, the positive-only transform, cap,
coefficient, predictive support, data, cadence, seed, endpoint, or evaluation
request.

## Reporting and interpretation

Report E42 head-to-head with E41 and the retained Dr.GRPO, E37 collision,
E38 predictive-Shannon, and E39 MATH trajectories. Primary quality endpoints
are neutral pass@8 and mean@8, guarded by greedy pass@1. Countdown and graph
coloring additionally report semantic coverage and distinct-correct@8.

Mandatory mechanism telemetry includes eligible, gated, active,
reward-positive, and parseable fractions; raw-all-row and effective semantic
advantage mean/min/max/RMS; positive-only-zeroed and cap fractions;
predictor normalization and centering errors; history rows added and groups
updated/skipped; and semantic-to-task advantage RMS.

On MATH, E42 tests safety rather than proof-strategy diversity. Distinct
incorrect final answers are not useful diversity and must never receive
effective semantic pressure. A positive MATH result can establish that
bounded novelty among verified successes does not harm neutral quality; it
cannot by itself establish diverse correct reasoning strategies.
