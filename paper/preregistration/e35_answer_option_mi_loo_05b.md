# E35: leave-one-out answer-option mutual information at 0.5B

**Status: FROZEN BEFORE LAUNCH (2026-07-22).**

**Execution amendment (2026-07-22, before allocation or any outcome):** the
initial `v1` cohort (jobs 30062926--30062937) was cancelled while all twelve
jobs were still pending.  A post-submission runtime test found that the new
leave-one-out classifier-accuracy diagnostic referenced the retired local
variable name `counts` instead of `scoring_counts`, which would have raised on
the first eligible row.  `v2` changes only that diagnostic reference.  No v1
run directory, metric, checkpoint, or allocation exists; v1 is inadmissible.

## Motivation and separation from E34

E34-v2 used an exact-per-prompt EMA discriminator that scored a rollout only
from earlier visits to the same prompt.  A live audit found that this made the
MI reward identically zero for the entire first prompt pass and at most about
`0.018` thereafter.  E34-v2 is retained as an inadmissible mechanism pilot for
this corrected comparison; none of its metrics or checkpoints may be merged
with E35.

E35 changes the estimator, uses a fresh three-seed cohort, and retains the
same semantic answer representation, model, tasks, optimizer, latent prompt,
correct-only gate, and total candidate count.

## Question

Does a leakage-safe leave-one-out estimate of conditional mutual information
bind a latent option to the semantic identity of a correct answer on free-form
Countdown and graph coloring without degrading success relative to matched
Dr.GRPO?

## Arms

1. `grpo`: ordinary free-form Dr.GRPO.
2. `diayn`: the same update with four balanced answer-option latents and a
   correct-only semantic MI reward.

Each prompt group contains 16 candidates, four from every
`z in {0,1,2,3}`.  For eligible candidate `i`, the treatment adds

`0.10 * (log q_{-i}(z_i | x, c(a_i)) - log(1/4))`

to terminal reward.  `q_{-i}` combines the pre-batch EMA history with counts
from the other 15 candidates in the same prompt group.  Candidate `i` is
removed from both the numerator and denominator, so a singleton answer cannot
reward itself.  After every group is scored, the EMA history is updated.

The discriminator receives only the prompt/reference identity and canonical
semantic answer key `c(a)`.  It never receives response text, formatting,
rationale, or token length.

## Frozen contract

- Tasks: Countdown easy3 and graph coloring, `multi_answer` split.
- Model: `Qwen/Qwen2.5-0.5B-Instruct`, revision
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Prompt policy: free-form `qwen_boxed`; no canonical-action constraint.
- Training seeds: 3501, 3502, 3503.
- Group size: 16; four candidates per latent option.
- Training duration: five complete prompt passes.
- Optimizer: learning rate `2e-7`, one PPO epoch, `beta=0`, no token-MaxEnt,
  SEED, xDr, KL, or other entropy treatment.
- MI coefficient: `0.10`; history EMA decay `0.9`; additive smoothing `1.0`;
  correct answers only; leave-one-out estimator enabled.
- Sampled evaluation: four fixed draws of eight samples at temperature 1,
  plus deterministic greedy pass@1.  Evaluation seeds begin at 350100.
- Execution: one 24 GB GPU per run, ZeRO stage 2, learner microbatch 1.
- Cohort size: 12 runs = 2 tasks x 2 arms x 3 seeds.

## Readout

Report seed-level and three-seed mean trajectories for pass@1, mean@8,
pass@8, semantic coverage@8, and distinct correct modes@8.  For the treatment,
also report:

- draw-wise cross-fitted lower bound on `I(z; c(a) | x, correct)`;
- option-classification accuracy;
- option correctness-rate range;
- training MI lower bound, realized MI bonus range, correct-only eligibility,
  and leave-one-out support fraction.

A positive result requires positive held-out MI across seeds without material
loss in sampled success against the matched E35 Dr.GRPO controls.  Failure-rate
specialization alone is not semantic binding.
