# E34: answer-option mutual information pilot at 0.5B

**Status: FROZEN BEFORE LAUNCH (2026-07-22).**

**Execution amendment (2026-07-22, before any usable outcome):** the initial
`v1` quartet (jobs 30055795--30055798) was cancelled during startup after the
graph control OOMed on its first backward pass.  The launcher had set
`train_batch_size_per_device=16`, inconsistent with the established 24 GB
free-form recipe.  `v2` changes only the per-device microbatch to `1`; group
size remains 16 and every scientific setting below is unchanged.  No v1
metrics are admissible.

## Question

Can a latent option bind to the semantic identity of a correct answer, rather
than merely changing response style, on free-form Countdown and graph coloring?

## Arms

The pilot uses Qwen2.5-0.5B-Instruct and the existing free-form ModeBench
policy/evaluator contract.

1. `grpo`: ordinary free-form Dr.GRPO.
2. `diayn`: the same Dr.GRPO update with four balanced answer-option latents.

For the treatment, each 16-candidate prompt group contains four candidates for
each `z in {0,1,2,3}`.  The latent is inserted into the system message.  The
learner extracts the existing canonical semantic answer key `c(a)` and adds

`0.10 * 1[correct] * (log q(z | x, c(a)) - log(1/4))`

to terminal reward, clipped to five nats before multiplication by 0.10.
`q` is a smoothed EMA count discriminator.  It scores a rollout from state
available before that rollout and updates afterward, preventing self-scoring
leakage.  The prompt/reference identity namespaces every answer key, so the
objective is conditional MI `I(Z; A | X)`.  The discriminator never receives
the response text, token length, formatting, or rationale.

## Frozen pilot contract

- Tasks: Countdown easy3 and graph coloring, `multi_answer` split.
- Model: `Qwen/Qwen2.5-0.5B-Instruct`, revision
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Prompt policy: free-form `qwen_boxed`; no canonical-action constraint.
- Seed: 3401.
- Group size: 16; four samples per latent option.
- Optimizer: the matched free-form Dr.GRPO recipe, learning rate `2e-7`, one
  PPO epoch, `beta=0`, and no token-MaxEnt, SEED, xDr, or KL treatment.
- Training duration: five complete prompt passes.
- MI coefficient: `0.10`; EMA decay `0.9`; additive smoothing `1.0`; correct
  answers only.
- Sampled evaluation: four fixed draws of eight samples at temperature 1,
  plus greedy pass@1.

## Readout

The treatment is useful only if it improves held-out semantic option binding
without sacrificing answer quality.  Report:

- sampled pass@8 and mode coverage@8;
- a draw-wise cross-fitted lower bound on `I(z; c(a) | x, correct)` in nats
  (three fixed draws fit `q`; the fourth is scored, rotating the held-out draw);
- Bayes option-classification accuracy from canonical answer keys;
- the range of correctness rates across latent options;
- training MI lower bound and correct-only eligibility fraction.

A positive cross-fitted MI lower bound with a large option correctness-rate range is not
sufficient evidence: that can mean some latents specialize in failure.  Any
multi-seed follow-up requires finite training, nonzero held-out MI, and no
material degradation in sampled success relative to the matched control.
