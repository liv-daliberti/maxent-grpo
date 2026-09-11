# PantryPlan six-bit Dr.GRPO plumbing smoke v1

**Status: FROZEN BEFORE TRAINING — 2026-07-29**

## Antecedent and scope

The prospectively frozen six-bit viability gate passed and its independent
replay audit reproduced 62/64 prefix-success prompts, 63/64 multimode prompts,
and 1,496/4,096 verified completions. This smoke is the one Dr.GRPO plumbing
run authorized by that gate. It is development-only, is not one of the final
five seeds, and cannot authorize the five-seed comparison by itself.

## Frozen cell

- model: Qwen2.5-0.5B-Instruct snapshot
  `7ae557604adf67be50417f59c2c2f167def9a775`;
- data: first 32 prompts under the seed-76201 order of the frozen 384-row
  PantryPlan v2 training split; no development or evaluation output is used
  for optimization;
- arm: plain Dr.GRPO (`grpo`), with every MaxEnt, novelty, replay, and token
  entropy actuator disabled;
- seed: 76201; rollouts: 16 per prompt; exactly 32 optimizer updates;
- policy: exactly six binary action tokens, two allowed tokens at every
  position, all 64 masks unfiltered;
- actor and learner: learner-side fixed-shape sampling with the complete
  restricted behavior distribution recorded at every decision;
- learning rate 2e-7, one PPO epoch, no recovery or requeue;
- evaluation: frozen Pantry v2 evaluation split, greedy plus one deterministic
  temperature-one K=8 draw every eight prompt updates. Evaluation does not
  affect training or stopping.

The environment projects only the emitted support through the public local
quantity constraints. Certified support lists, reference allocations, desired
mode counts, target entropy, development outcomes, and evaluation outcomes are
absent from the model context and optimizer.

## Frozen pass criteria

The smoke passes only if the unique job completes all 32 updates and every
update satisfies all of the following:

- six actions and 16 length-terminated rollouts, with no invalid or unexpected
  termination;
- actor restricted support min=max=2 and normalization error at most 1e-5;
- learner restricted normalization error at most 1e-5, positive density ratios,
  and positive sequence and prefix ESS;
- exact finite-policy leaf mass 1.0, leaf count 64, prefix-row count 63, and
  entropy within `[0, log(64)]`;
- finite loss, KL, entropy, and gradient diagnostics.

At least one update must have positive verified reward, and the sampled
evaluation stream must contain at least one prompt with two verifier-distinct
supports. A pass establishes only that current actor/learner/reward plumbing is
trainable. A separate prospective paired MaxEnt-mechanism smoke is required
before any five-seed Pantry Stage-B jobs.
