# E60: finite verified bootstrap, then prompt-local canonical replay

**Status: FROZEN BEFORE E60 PYTHON PILOT SUBMISSION — 2026-07-27**

## Long-horizon failure being repaired

E58 made sparse executable Python rewards learnable by replaying one
model-discovered verified bank on every optimizer update. It remained finite
and crash-free, but the cross-prompt singleton actuator became an unintended
mode-selection mechanism. At matched step 1,111:

- E57 prompt-local replay had discovered 156 outcomes over 82 prompts
  (mean support 1.90);
- E58 global replay had discovered 95 outcomes over 94 prompts
  (mean support 1.01);
- E57's open-set entropy EMA was 0.66 with coefficient 0.062;
- E58's open-set entropy EMA was 0.19 with coefficient 0.179.

E58's inverse controller therefore detected the collapse and increased its
unbounded coefficient. The failure was actuator timing: every global
singleton likelihood update reinforced the only known correct response before
another valid response had been proposed. The known-mode balance loss is
identically zero for a singleton bank, while repelling the sole successful
on-policy response can move probability into invalid strings. Continued
global singleton replay consequently opposed the discovery mechanism it was
meant to bootstrap.

## Frozen repair

E60 changes only the replay schedule.

1. Before any validator-positive model output, every added objective is
   inactive.
2. After the first discovery, replay selects one verified prompt bank per
   optimizer update using E58's checkpointed prompt-hash round robin.
3. Exactly 64 **non-empty** global replay updates are allowed. This is the
   verified-mass controller's already-frozen self-calibration window.
4. After update 64, replay permanently returns to the prompt in the current
   rollout. The global update count and phase are checkpointed.

The transition is a finite cold-start compute budget, not a semantic target.
It never reads an evaluation, exhaustive support, a gold mode count, desired
entropy, desired success rate, or controller coefficient. Empty pre-discovery
updates do not consume the budget. Zero retains the old prompt-local or
unlimited-global configurations, so E60 is an explicit new arm rather than a
silent change to E57 or E58.

All E58 objectives remain:

- success-conditioned open-set semantic pressure;
- uniform verified-mass likelihood;
- reverse-KL balance over at least two model-discovered verified modes;
- one-time validator-positive novelty.

All three adaptive coefficients remain projection-free and have neither an
upper nor lower bound. Direct token entropy remains disabled.

## Python causal pilot

The first E60 run uses Qwen2.5-0.5B-Instruct, seed 9010, executable Python
factors, group size 16, learning rate `2e-7`, one PPO epoch, temperature 1,
top-p 1, response limit 192, and five complete passes over the 384-prompt
training pool. Evaluation uses the already-frozen 128 prompts, eight draws,
temperature 1, and quarter-pass cadence.

The pilot is inspected at and beyond matched step 1,056, where E57 had
mean discovered support 1.77 and E58 remained at 1.01. It passes the causal
gate only if:

- the global phase records exactly 64 non-empty updates and then stays off;
- prompt-local phase telemetry stays on thereafter, including after resume;
- direct token entropy, coefficient projections, gold-support feedback, and
  evaluation feedback remain absent;
- all losses, gradients, coefficients, and controller state remain finite;
- there is no traceback, OOM, worker death, or no-EOS/length failure;
- at least six of the final eight quarter-pass evaluations have positive
  `distinct-correct@8 - pass@8`;
- the final-eight mean distinct-correct@8 exceeds matched Dr.GRPO and the
  final-eight mean excess multiplicity is positive;
- final-eight pass@8 and mean@8 trail matched Dr.GRPO by no more than 0.03;
- final-eight mean distinct-correct@8 retains at least 75% of the pilot's best
  rolling-eight value.

These are relative and self-retention checks. No absolute desired number of
correct modes is used by training, controller adaptation, or pilot selection.

## Three-domain sentinel and replication

Only a clean Python causal pilot may authorize fresh seed-9010 E60 runs in
Countdown easy3, graph coloring, and executable Python factors. Each domain
must pass the same final-eight multiplicity, matched-quality, self-retention,
runtime, checkpoint, and information-firewall gates.

Only a terminal three-domain sentinel may authorize seeds 43, 44, and 45.
Completion requires stable three-seed evidence in all three domains; a
single-seed sentinel is not the final result.
