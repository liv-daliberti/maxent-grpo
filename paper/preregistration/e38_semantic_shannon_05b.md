# E38: free-form semantic Shannon entropy at 0.5B

**Status: FROZEN BEFORE LAUNCH (2026-07-23).**

## Motivation and relation to E37

E37 tests a dense, group-local collision penalty over canonical free-form
outcomes. That penalty estimates a quadratic/Tsallis-2 diversity objective,
but its resolution is limited by the 16 samples in the current rollout group:
all singletons receive the same shaping reward. E38 asks whether a bounded
predictive Shannon-surprisal reward can retain information across rollout
groups and distinguish the long tail more finely.

E38 is a prospectively frozen, treatment-only extension. It reuses the already
launched E37 Dr.GRPO and semantic-collision arms as historical comparators; it
does not resubmit either arm. The decision to test a Shannon successor follows
the early E37 read, so E38 is not independent confirmatory evidence for the
broad semantic-diversity hypothesis. No E38 outcome or dose sweep is used to
select its settings: the coefficient, estimator, smoothing, clipping, seeds,
evaluation requests, and endpoint below are fixed before any E38 job is
submitted.

## Question

Relative to the matched E37 Dr.GRPO and semantic-collision trajectories, does
bounded predictive Shannon surprise improve neutral-prompt pass@8 and
distinct-correct@8 without materially reducing greedy pass@1 or mean@8?

## Frozen treatment

The sole new arm is `semantic_shannon`. It uses the same free-form Dr.GRPO
update as E37, after adding a bounded semantic-surprise term to every
candidate's terminal task reward. There are no latent instructions, enumerated
valid-mode catalogue, learned classifier, token-entropy term, or correctness
filter.

For prompt `x`, let `n_x(a)` be the historical count of canonical outcome `a`
from all completed earlier rollout groups for that prompt and let
`N_x = sum_a n_x(a)`. In the current group of size `G=16`, let `m_-i(a)` be
the count among the other 15 candidates. For row `i`, the explicit support
`S_-i` is the union of historical keys and keys observed among those 15 peers,
with `K_-i = |S_-i|`. The predictive distribution also contains one unseen
bucket. Every observed outcome participates. Parse failures map to one shared
`INVALID` key.

Before the current group is added to history, define

`D_i = N_x + G - 1 + (K_-i + 1)`.

If `a_i` belongs to the explicit support, its leave-one-out
posterior-predictive probability is

`p_hat_i = (n_x(a_i) + m_-i(a_i) + 1) / D_i`.

If `a_i` has never appeared in history or among its current peers, it is
scored through the unseen bucket with `p_hat_i = 1 / D_i`.

Its clipped surprise and shaping reward are

`s_i = min(-log(p_hat_i), 5.0)`,

`b_i = 0.10 * (s_i / 5.0 - 1)`,

and the learner uses `R_i = r_i + b_i`. The full current-group counts are
added to history only after all candidates have been scored. Thus
`b_i in [-0.10, 0]`. Under the binary task rewards used here,
`R_correct >= 0.90 > 0 >= R_incorrect`; a correct response always has a
strictly greater shaped reward than an incorrect response.

The frozen coefficient is `0.10`, the surprise clip is `5.0` nats, and the
Dirichlet pseudocount is `1.0` for every explicit outcome and the unseen
bucket. There is no dose selection from E37 or E38 outcomes. The historical
state is keyed by a stable SHA256 digest of the unpadded prompt token IDs and
stores exact integer outcome counts. It is included in optimizer-resumable
checkpoints and restored exactly before continued training. Requeue/resume
must therefore reproduce the uninterrupted estimator.

## Frozen training contract

- Tasks: Countdown easy3 and graph coloring, `multi_answer` split.
- Model: `Qwen/Qwen2.5-0.5B-Instruct`, revision
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Prompt policy: neutral free-form `qwen_boxed`; no canonical-action or latent
  prompt modification.
- Training seeds: `43, 44, 45`, matching E37 exactly within each task.
- Group size: 16.
- Training duration: ten complete prompt passes.
- Optimizer: learning rate `2e-7`, one PPO epoch, `beta=0`.
- Excluded mechanisms: outcome-collision shaping, answer-option MI,
  token-MaxEnt, EMA/Haarnoja MaxEnt, SEED, xDr, KL, and canonical-action
  sampling.
- Execution: one 24 GB A5000-class GPU per run, ZeRO stage 2, learner
  microbatch 1.
- New cohort size: 6 fresh runs = 2 tasks x 1 arm x 3 seeds.
- Source, execution shell surface, launcher, protocol, model revision, and
  datasets are frozen by the E38 identity artifact before release.
- Jobs are submitted held, audited, and released only as one complete cohort.
- One rolling optimizer-resumable checkpoint is written per prompt pass; the
  two newest checkpoints are retained. Evaluation remains quarter-pass, and
  model export is terminal-only.

The treatment pins `OAT_ZERO_VARIANT=semantic_shannon`,
`OAT_ZERO_OUTCOME_COLLISION_COEF=0.0`,
`OAT_ZERO_SEMANTIC_SHANNON_COEF=0.10`,
`OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP=5.0`, and
`OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT=1.0`. Every other stochastic,
optimization, prompt, evaluation, placement, and recovery setting matches E37.

## Frozen neutral evaluation and comparison contract

- E38 uses the ordinary `qwen_boxed` prompt with no latent instruction.
- Dataset order, batching, request structure, sample count, temperatures, and
  evaluation seeds match E37 exactly.
- Record deterministic greedy pass@1 plus four independent K=8 draws at
  temperature 1.
- Draw seeds are `370100, 370101, 370102, 370103`.
- The four draws define pass@8, mean@8, semantic coverage@8, and
  distinct-correct@8. There is no latent-conditioned evaluation.
- Because E37 and E38 start from identical model weights and use identical
  neutral evaluation requests, every E38 step-zero evaluation metric and raw
  draw must agree exactly with the corresponding E37 `(task, seed)` rows.
  Any mismatch quarantines that seed before post-training comparison.
- Primary comparisons are paired by seed against both frozen E37 arms. Report
  all three trajectories and their seed mean; do not pool prompt rows as
  independent replicates.
- The E37 identity and comparison prefixes are recorded in the immutable E38
  identity artifact. No E37 job is relaunched or modified by E38.

## Mechanism telemetry and readout

The learner records original and augmented reward means; bonus mean/min/max;
raw, clipped, and normalized surprisal means; mean predictive Shannon entropy;
clip fraction; predictive-probability mean/min/max; unseen-outcome fraction;
maximum predictive-distribution normalization error; distinct-outcome count
and fraction; invalid and parseable fractions; mean historical count; and
numbers of tracked prompts and outcomes.

The primary quality estimands are neutral pass@8 and mean@8 at ten passes,
with greedy pass@1 as the quality guardrail. The primary breadth estimand is
distinct-correct@8, supported by semantic coverage@8. Mandatory safety
diagnostics are response length, invalid fraction, clip fraction, and the
distribution of predictive probabilities.

E38 may be interpreted as evidence for bounded semantic Shannon shaping only
if its checkpointed historical estimator is active, its step-zero pairing
passes, and its neutral quality/breadth curves improve relative to the frozen
E37 comparators. It does not test latent binding and does not establish that
unbounded surprise rewards are safe.
