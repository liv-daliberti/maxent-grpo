# E69 Gate 2 compute-matched screen execution contract

Date frozen: 2026-07-28, after the registered free-form-MATH abstention and
before any Gate 2 job was submitted.

This document instantiates Gate 2 of
`e69_verified_route_successor_protocol_20260728.md`. It does not change the
algorithm or inspect MATH-500.

## Cohort

The screen uses Qwen2.5-0.5B-Instruct, training seed 43, 16 neutral rollouts
per prompt, learning rate `2e-7`, one PPO epoch, and exactly six deterministic
passes through each training population.

Four physical arms run on Graph, Countdown, Python factors, and MathIR:

1. `grpo_compute_matched`, reported as Dr.GRPO;
2. `verified_first_global_replay_canonical`, reported as E66;
3. `verified_entropy_gated_singleton_escape_canonical`, reported as E68; and
4. `verified_route_successor`, reported as E69.

On MATH12K-384/MATH12K-route-dev-128, Gate 1 selected abstention. There are
therefore only two scientifically unique physical arms: compute-matched
Dr.GRPO and endpoint-only replay. The endpoint run is reported identically
under the E66, E68, and E69 labels. Duplicate optimization runs are forbidden:
they add no independent evidence and would falsely inflate the sample size.

## Frozen data and prompting

- Graph: `var/data/exact_answer_mode_probe`, 192 train / 96 development;
- Countdown: `var/data/exact_countdown_easy3_probe`, 384 / 128;
- Python factors: `var/data/python_factor_modebench_v1`, 384 / 128;
- MathIR: `var/data/mathir_action_menu_v1`, 384 / 128;
- free-form MATH: `var/data/math12k_384_route_dev128_v1`, 384 / 128.

The four executable domains use `qwen_boxed` and their exact validators.
Free-form MATH uses the ordinary `qwen_math` prompt and `math_verify`. It
never uses the failed trace prompts, route novelty, route proposals, or
cross-prompt route replay. MATH-500 remains absent from the evaluation root.

## Compute contract

Every physical arm uses replicated free-form sampling and local actor weight
synchronization. For every training prompt it issues:

- one neutral request of 16 rows; and
- three independently seeded, proposal-shaped requests of 16 rows each.

Thus every arm is charged 64 sampled rows per training prompt and the same
maximum response-token budget. Sampling-only controls are verified by the
ordinary task verifier and discarded before banks, replay, and PPO. E68/E69
may inspect the already-generated proposal groups; E66 and Dr.GRPO discard
all three. Early proposal success never reduces the charged budget.

Every arm also enables one global replay group per optimizer update with
capacity 16. The runtime reports both realized teacher-forced prompt/response
tokens and the common conservative charged response-token cap
`16 * generate_max_length` per update. Realized sequence lengths may differ
because model outputs differ; the group count, capacity, two score passes,
backward traversal, and charged cap do not. Dr.GRPO replaces the replay score
derivative by exact zeros immediately before the same chunked backward path.
Its ordinary task gradient is unchanged.

Training prompts, neutral rollouts, proposal-control rows, charged sampling
and replay caps, verifier calls, optimizer updates, evaluation cadence, and
terminal checkpoint are therefore fixed by construction. Realized tokens are
reported and must remain within their common charged caps.

## Evaluation and checkpoint choice

Evaluation occurs at initialization and after every full prompt pass. Each
checkpoint has one deterministic greedy evaluation and one fixed-seed
temperature-1 sample of eight responses per development prompt. The fixed
reporting passes are `0, 1, 2, 3, 4, 5, 6`; terminal pass 6 is the only Gate
2 checkpoint. No best-checkpoint selection is allowed.

Report greedy verified accuracy, mean@8, pass@8, and verified distinct@8
(where route/mode identity is defined). Free-form MATH distinctness is not a
strategy metric.

## Additional audit and persistence gates

The parent Gate 2 thresholds apply to E69 minus compute-matched Dr.GRPO. In
addition:

- every parent quality non-inferiority condition must hold at both passes 5
  and 6;
- at least three executable domains must have positive distinct@8 deltas at
  both passes 5 and 6;
- MathIR pass@8 and distinct@8 deltas must both be positive at passes 5 and 6;
- on MATH route-dev, greedy and mean@8 must each be no worse than `-0.01`
  versus control and at least one must be positive, in addition to the parent
  greedy/pass@8 gate;
- at least three executable domains must record a neutral route on a target
  prompt only after that route had been replayed there from another prompt;
- every Dr.GRPO replay record must report compute-only mode, zero applied
  replay loss, and zero applied replay-gradient norm;
- every arm must report three fixed control groups, 48 control rows, zero
  control rows to PPO, one replay-group budget, and no forbidden gold or
  evaluation feedback.

The post-replay reproduction counter is a temporal mechanism diagnostic, not
by itself a causal estimate. The arm-level task and support differences
remain the causal screen.

If any integrity check fails, the screen fails. If the outcome thresholds
fail, E69 does not advance and no hyperparameter is changed under this
experiment identifier.
