# E37: free-form semantic outcome-collision MaxEnt at 0.5B

**Status: FROZEN BEFORE LAUNCH (2026-07-23).**

## Motivation and separation from E36

E36 attempts to discover diverse correct answers and bind them to a latent
option in one objective. Its leave-one-out answer-option estimator has produced
little usable binding signal because correct semantic answer repetitions are
sparse. E37 tests the preceding causal question directly: can a dense,
group-local penalty on repeated semantic outcomes preserve a broader answer
distribution in free-form training?

E37 has no latent variable, latent instruction, classifier, answer-mode
catalogue, direct token-entropy term, or MaxEnt controller. E36 remains a
separate mechanism experiment and no E36 result is pooled with E37.

## Question

Relative to matched free-form Dr.GRPO, does penalizing duplicate canonical
outcomes within each rollout group improve neutral-prompt pass@8 and semantic
answer coverage without materially reducing pass@1?

## Arms and frozen treatment

1. `grpo`: ordinary free-form Dr.GRPO.
2. `outcome_collision`: the identical update after adding a group-local
   duplicate penalty to each candidate's terminal task reward.

For candidate `i` in a group of size `G=16`, let `k_i` be the canonical answer
key extracted by the existing task grader. Every observed outcome participates,
whether correct or incorrect. Every parse failure maps to one shared `INVALID`
key, so unparsable responses collide rather than masquerading as diverse.

The frozen shaping term is

`b_i = -(0.10 / G) * sum_{j != i} 1[k_i = k_j]`,

and the learner uses `R_i = r_i + b_i`. Thus `b_i` is in
`[-0.09375, 0]`. Under the binary task rewards used here, a correct response
always has strictly greater shaped reward than an incorrect response:
`R_correct >= 0.90625 > 0 >= R_incorrect`.

The coefficient is fixed prospectively at `0.10`; there is no dose selection
from E37 outcomes. The control pins it to exactly `0.0`.

## Frozen training contract

- Tasks: Countdown easy3 and graph coloring, `multi_answer` split.
- Model: `Qwen/Qwen2.5-0.5B-Instruct`, revision
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Prompt policy: neutral free-form `qwen_boxed`; no canonical-action or latent
  prompt modification.
- Training seeds: `43, 44, 45`, paired exactly across arms within each task.
- Group size: 16.
- Training duration: ten complete prompt passes.
- Optimizer: learning rate `2e-7`, one PPO epoch, `beta=0`.
- Excluded mechanisms: answer-option MI, token-MaxEnt, EMA/Haarnoja MaxEnt,
  SEED, xDr, KL, and canonical-action sampling.
- Execution: one 24 GB A5000-class GPU per run, ZeRO stage 2, learner
  microbatch 1.
- Cohort size: 12 fresh runs = 2 tasks x 2 arms x 3 seeds.
- Source, execution shell surface, launcher, protocol, model revision, and
  datasets are frozen by the E37 identity artifact before release.
- Jobs are submitted held, audited, and released only as one complete cohort.
- One rolling optimizer-resumable checkpoint is written per prompt pass; the
  two newest checkpoints are retained. Evaluation remains quarter-pass, and
  model export is terminal-only.

Apart from the run/arm identifiers required to keep artifacts distinct, the
only method difference between a paired control and treatment is
`OAT_ZERO_VARIANT` plus the corresponding
`OAT_ZERO_OUTCOME_COLLISION_COEF=0.0` versus `0.10`. All stochastic,
optimization, prompt, evaluation, placement, and recovery settings are shared.

## Frozen neutral evaluation contract

- Both policies are evaluated with the ordinary `qwen_boxed` prompt and no
  latent instruction.
- Dataset order, batching, request structure, sample count, temperatures, and
  seeds are identical across arms.
- Record deterministic greedy pass@1 plus four independent K=8 draws at
  temperature 1.
- Draw seeds are `370100, 370101, 370102, 370103`.
- The four draws define pass@8, mean@8, semantic coverage@8, and
  distinct-correct@8. There is no latent-conditioned evaluation.
- Because matched arms have identical initial weights and neutral requests,
  every step-zero evaluation metric and raw draw must agree exactly within
  each `(task, seed)` pair. A mismatch quarantines that pair before any
  post-training comparison.
- Primary comparisons are paired by seed. Report all three trajectories and
  their mean; do not pool prompt rows as independent seed replicates.

## Mechanism telemetry and readout

The learner records, per update:

- original task-reward mean and augmented-reward mean;
- collision rate;
- collision-bonus mean, minimum, and maximum;
- number of distinct observed outcomes per group; and
- invalid-answer fraction.

The primary quality estimands are neutral pass@8 and mean@8 at ten passes,
with greedy pass@1 as the quality guardrail. The primary mechanism estimands
are all-outcome collision rate and valid/correct semantic coverage. Response
length and invalid fraction are mandatory safety diagnostics.

Interpretation is deliberately staged:

- If E37 does not improve semantic coverage, no latent-binding successor is
  justified.
- If E37 improves coverage without violating the quality guardrail, a later
  separately preregistered experiment may distill or route that discovered
  support with latent options. E37 itself makes no claim about latent binding.
