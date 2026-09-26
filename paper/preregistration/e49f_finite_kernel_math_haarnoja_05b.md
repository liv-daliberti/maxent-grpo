# E49F: finite-kernel executed-strategy MATH with normalized Haarnoja control

**Status: FROZEN BEFORE POLICY TRAINING — 2026-07-24**

## Question

Can the successful E46 normalized canonical-bank Haarnoja mechanism transfer
to hard free-response MATH when the canonical outcomes are a small,
validator-bound set of executable solution strategies rather than text
clusters?

## Canonical outcome

Every prompt contains a frozen finite menu with at most twelve action
definitions and one or more retained strategy combos. Actions carry one code
from the frozen operation enum and strategies carry one kernel from the frozen
route enum. The policy must:

1. declare one listed strategy ID;
2. declare its exact ordered action combo;
3. emit exactly one nonempty action-step block for every action in that order;
4. execute the stated operation in each block; and
5. produce a task-validator-correct boxed answer.

Missing, extra, duplicated, reordered, or mismatched steps deterministically
fail before semantic admission. Two independently seeded Qwen2.5-72B
integrity passes then verify that the written mathematics genuinely executes
the declared combo. Any invalid, ambiguous, truncated, or disagreeing
assessment receives no key and task reward zero. An accepted outcome key is
the exact frozen `(menu hash, strategy ID, action combo)`; runtime prose cannot
create a new strategy.

## Offline bank calibration

The toy bank uses the frozen E49E raw trace evidence, conservative answer
surface normalization, finite-kernel answer-bound augmentation, and
singleton-gap repair. Every retained route receives two action-by-action
soundness audits. Every retained pair receives two equivalence attacks;
different labels, notation, units, reordered algebra, redundant checks, or
representation-only changes are equivalent.

Before training, a blinded manual packet contains every retained distinct
pair plus three equivalent-route controls. Advancement requires:

- all 100 toy prompts certified;
- at least 20 multi-route prompts overall and at least 10/50 in evaluation;
- all five invalid-route controls rejected;
- all three equivalent-route controls rejected as new;
- zero manually identified false-new retained pairs;
- all three blinded equivalent controls recognized; and
- every rendered prompt at most 2048 tokens.

Auditor-only reference answers and gold derivations never reach the policy
prompt. Local validation rejects literal answers and numeric intermediates
not present in the original problem, except structural `0`, `1`, and `2`.

## Matched toy experiment

Both arms use the exact same:

- Qwen2.5-0.5B-Instruct initialization;
- 50 hard-MATH training prompts and 50 held-out hard-MATH prompts;
- calibrated finite menus and execution-gated task reward;
- seed `45`, group size `16`, learning rate `2e-7`;
- three prompt epochs and one PPO epoch;
- prompt/output limits `2048/1024`;
- sampling temperature/top-p `1/1`;
- runtime Qwen2.5-72B admission checks;
- one A100 GPU; and
- evaluation cadence after each 50-prompt epoch.

The control is ordinary Dr.GRPO. It passively performs the same strategy
tracking and reward gating but has zero canonical objective influence.

The treatment is the unchanged E46 normalized canonical-bank Haarnoja arm:

- initial alpha `0.10`;
- normalized target
  `rho_x = H(q_x) / log |B_x^+| = 0.80`;
- alpha bounds `[0.10, 0.50]`;
- log-alpha learning rate `0.003`;
- normalized-entropy EMA decay `0.90`;
- bank pseudocount `1.0`;
- novelty coefficient `0.50`; and
- surprisal clip `5.0`.

## Toy advancement

Both arms must complete all three epochs with exact source/data identities and
zero contract-audit failures. The treatment must show a nonzero canonical
learning signal on eligible multi-route prompts, finite normalized-entropy and
alpha telemetry, and no terminal normalized-support collapse. Evaluation
accuracy must not be materially worse than matched Dr.GRPO; a five-point
absolute tolerance is the engineering stop boundary for the 50-example toy.

The mechanism report compares the direction and persistence of:

- normalized canonical entropy;
- effective support and distinct valid strategies;
- controller alpha and target error;
- first discoveries and rediscoveries;
- execution-contract acceptance; and
- task accuracy

against the successful E46 Countdown and graph-coloring trajectories. No
claim of equivalence is made from endpoint accuracy alone.

## Full experiment

Only after the toy gate passes, the same calibrated finite-kernel contract is
materialized for the exact OAT cohort:

- 384 MATH12K training prompts;
- all 500 MATH-500 evaluation prompts; and
- three prompt epochs.

The matched Dr.GRPO and E46-Haarnoja arms retain the toy hyperparameters and
one-A100 allocation. The full stage reports task accuracy, pass@8,
execution-gated acceptance, strategy coverage, normalized entropy,
controller behavior, and comparison with Countdown/graph coloring.
