# E36: answer-option MI with separated quality and binding evaluation at 0.5B

**Status: FROZEN BEFORE LAUNCH (2026-07-23).**

## Motivation and separation from E35

E35 trained the intended leave-one-out answer-option MI mechanism, but its
evaluation automatically changed DIAYN's quality prompt to a latent-conditioned
prompt while leaving Dr.GRPO neutral.  It also reused one request seed across
the four latent requests.  Consequently E35's apparent step-zero difference and
its latent diversity readout are evaluation confounds.  E35 remains an
inadmissible mechanism/debugging cohort and no E35 outcome is pooled with E36.

E36 keeps the E35 training mechanism and replaces its evaluation with two
separate estimands.

## Question

Does leakage-safe leave-one-out conditional mutual information bind an option
latent to the semantic identity of a correct answer, while preserving
unconditioned free-form task quality relative to matched Dr.GRPO?

## Arms

1. `grpo`: ordinary free-form Dr.GRPO.
2. `diayn`: the same update with four balanced answer-option latents and a
   correct-only semantic MI reward.

Each training prompt group contains 16 candidates, four from every
`z in {0,1,2,3}`.  For eligible candidate `i`, the treatment adds

`0.10 * (log q_{-i}(z_i | x, c(a_i)) - log(1/4))`

to terminal reward.  The discriminator sees only prompt/reference identity and
the canonical semantic answer key.  It never receives response text,
formatting, rationale, or token length.

## Frozen evaluation contract

### 1. Neutral-prompt quality

- Evaluate both policies with the ordinary `qwen_boxed` prompt and no latent
  instruction.
- Use identical dataset order, batching, request shape, sample count,
  temperature, and seeds in the two arms.
- Record deterministic greedy pass@1 plus four K=8 draws at temperature 1.
- Draw seeds are `360100, 360101, 360102, 360103`.
- These draws exclusively define the comparative pass@8, mean@8, semantic
  coverage@8, and distinct-correct@8 curves.
- Because matched arms have identical initial weights, the step-zero neutral
  evaluations must agree exactly within each task and training seed.  A
  mismatch invalidates the cohort before interpreting training effects.

### 2. Latent-conditioned binding

- Evaluate only the DIAYN policy with the four frozen answer-option
  instructions.
- K=8 consists of two samples from each option.
- Every `(draw, prompt, z)` request has an independent deterministic seed:

  `request_seed = draw_seed * 1_000_000 + prompt_index * 4 + z`.

- The global evaluation-dataset prompt index is used, so seeds do not repeat
  across actor batches.
- Record the exact request seeds in the durable prompt-level JSONL sidecar.
- Compute latent-conditioned quality separately from neutral quality.
- Compute the answer-option MI lower bound and classifier accuracy by holding
  out each complete draw and fitting the semantic option table on the other
  three draws.  Failure-rate specialization is reported separately and does not
  count as answer binding.

## Frozen training contract

- Tasks: Countdown easy3 and graph coloring, `multi_answer` split.
- Model: `Qwen/Qwen2.5-0.5B-Instruct`, revision
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Prompt policy: free-form `qwen_boxed`; no canonical-action constraint.
- Training seeds: 3601, 3602, 3603.
- Group size: 16; four candidates per latent option in the DIAYN arm.
- Training duration: ten complete prompt passes.
- Optimizer: learning rate `2e-7`, one PPO epoch, `beta=0`, no token-MaxEnt,
  SEED, xDr, KL, or other entropy treatment.
- MI coefficient: `0.10`; history EMA decay `0.9`; additive smoothing `1.0`;
  correct answers only; leave-one-out estimator enabled.
- Execution: one 24 GB GPU per run, ZeRO stage 2, learner microbatch 1.
- Cohort size: 12 runs = 2 tasks x 2 arms x 3 seeds.

## Readout

The primary quality plot uses only neutral-prompt metrics.  A separate binding
plot reports latent-conditioned coverage, held-out conditional MI,
option-classification accuracy, correct-only eligibility, and option
correctness-rate range.  Report all three seed trajectories and the three-seed
mean; do not merge the two evaluations into one curve.
