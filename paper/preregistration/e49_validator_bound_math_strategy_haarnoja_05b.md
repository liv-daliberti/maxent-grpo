# E49: validator-bound MATH strategy Haarnoja at 0.5B

**Status: FROZEN BEFORE LAUNCH — 2026-07-24**

## Question

Can the successful graph-coloring/Countdown E46 mechanism transfer to hard,
free-form MATH when the growing support consists of validated mathematical
strategies rather than answer strings?

E49 has a hard-toy advancement stage followed by the exact existing
MATH12K-384/MATH-500 OAT split. Both stages are three prompt epochs and compare
the treatment with a contemporaneous, compute-matched Dr.GRPO control.

## Canonicalization and validation

The policy is never trusted to name its own strategy. The ordinary full
`math_verify` task validator runs first. Validator-negative, inactive,
unparseable, and no-EOS rows receive no canonical key and can never enter the
bank.

For each prompt group, only validator-positive solutions plus one stored
representative of each previously admitted prompt-local strategy are sent to
the frozen E47J Qwen2.5-72B-AWQ judge. The E47 system prompt, strategy
definition, JSON schema, temperature zero decoding, and permutation seeds
`470721` and `470722` are unchanged. The server constrains each response to the
E47J membership-only JSON schema; free-text strategy descriptions are neither
generated nor consumed by the objective.

Non-ambiguous candidates form components in the union of the two candidate
partitions. A component touching exactly one stored representative in either
pass reuses that key. A component touching multiple representatives is
rejected. A component touching none receives a new key only when every stored
representative was observed and both passes separate the component from all of
them. This creates a new boundary only with unanimous separation, without
letting one conflicting candidate invalidate an unrelated existing-key match.

Failure is closed: it produces no key and no exploration reward. An otherwise
valid response that omits a candidate treats that candidate as ambiguous. An
omitted stored representative blocks new-key components but does not block a
component that safely reuses exactly one known key. Existing strategies retain
stable checkpointed IDs; a genuinely new consensus component receives one new
stable ID. The strategy state and the E46 count bank are checkpointed
separately. A judge transport, parse, or structural error terminates the update
rather than silently changing the objective.

The matched Dr.GRPO arm performs the same validation, two judge calls, strategy
tracking, checkpointing, and logging, but its strategy entropy and novelty
coefficients are exactly zero.

## Frozen E46 method

No policy-token or verbal uncertainty controller is used. The treatment is the
existing normalized canonical-bank Haarnoja method:

- group size `16`;
- `rho_x = H(q_x) / log |B_x^+|`, eligible only for `|B_x^+| >= 2`;
- target `rho*=0.80`;
- `alpha_0=alpha_min=0.10`, `alpha_max=0.50`;
- log-alpha Adam learning rate `0.003`, betas `(0.9,0.999)`, epsilon `1e-8`;
- entropy EMA decay `0.90`;
- novelty coefficient `0.50`;
- pseudocount `1`, surprisal clip `5`;
- immutable pre-group snapshots, row-order-independent commit, exploration
  advantage added after ordinary Dr.GRPO task centering, and controller
  observation after the bank update.

All other diversity objectives are zero.

## Model and optimizer

Both arms use the immutable local
`Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`,
neutral `qwen_math` prompting, learning rate `2e-7`, constant schedule,
Dr.GRPO, one PPO epoch, beta zero, max norm one, temperature one, top-p one,
and prompt/response/context limits `1024/1024/2048`. Each arm uses one A100.
The only treatment difference is the detached E46 strategy-bank advantage and
its alpha controller.

## Stage A: hard toy

Training contains the exact 50 E47 level-five MATH12K problems. Evaluation is
a deterministic held-out set of 50 difficulty-five MATH-500 problems, sorted
by `sha256("e49-hard-eval-v1" || problem || answer)`. The frozen materialized
artifact is `var/data/e49_math_strategy_toy`; its train and eval tree hashes
are respectively
`9c085fdcc5ed3bbf7032b254f1d98a4bbe5fc829ed0341c176a64022668ca731`
and
`01016cfd332cfd763f74d44e6d2e13dfd6ae080d7a2b15ebe4f65f4bb3e66f6b`.

Seed 45 runs for exactly three prompt epochs (150 prompt groups). Greedy
pass@1 and stochastic pass@8 are evaluated at initialization and after each
complete prompt epoch.

Stage A may launch only when bounded E47J-CAL clears its frozen gate. Stage B
may launch only when Stage A shows:

- no validator-negative strategy admission and no structural judge failure;
- nonzero canonicalizable-correct coverage and at least one prompt with
  support two or larger;
- finite nonzero exploration advantage in the treatment and exactly zero
  objective influence in matched Dr.GRPO;
- the normalized-entropy sensor and alpha move in the prescribed direction
  (alpha rises when the eligible entropy EMA is below `0.80`, and falls when
  above it), with no late non-finite or support-collapse failure;
- treatment terminal greedy accuracy is not more than two percentage points
  below matched Dr.GRPO; and
- either treatment greedy accuracy improves over initialization or its
  pass@8 improves while matched Dr.GRPO does not improve more.

The last rule is a small-pilot safety/learning gate, not a confirmatory claim.

## Stage B: exact OAT MATH split

Training and evaluation reuse, byte for byte:

- `var/data/math12k_384_math500/train`: the exact E39 MATH12K first 384 rows,
  Arrow SHA-256
  `359defbf82b6e05a1fdddb3479ed689f8a607dc727814e73ebfe69b2ffdff8b8`;
- `var/data/math12k_384_math500/eval`: full held-out MATH-500, Arrow SHA-256
  `2104f8f8eef09ce0bfc929e255f0f04c59311f1f3395bd293f1d03051c482cf7`.

Seed 45 runs for exactly three prompt epochs (1,152 prompt groups).
Full-MATH-500 greedy pass@1 and stochastic pass@8 are evaluated at
initialization and after each epoch. Checkpoints are written after each epoch;
the terminal model is exported.

## Required report and transfer criteria

The paired report includes validation reward, greedy pass@1, pass@8, accepted
strategy coverage, ambiguity/disagreement rejection, cumulative strategies,
support per solved prompt, normalized entropy, eligible fraction, alpha,
novelty and entropy advantage RMS, and exploration/task RMS ratio.

“Works like graph coloring and Countdown” requires the same mechanism pattern:
validated support grows, entropy pressure remains live after support reaches
two, alpha responds to normalized entropy rather than epoch number, and
quality is retained or improved relative to matched Dr.GRPO. E49 is successful
only if the treatment finishes all three full-data epochs, has no correctness
or bank-integrity violation, improves full-MATH-500 pass@1 or pass@8 from
initialization, and is not more than two percentage points below matched
Dr.GRPO on terminal pass@1. Any change after observing Stage A is a
prospectively named successor and may not overwrite E49.
