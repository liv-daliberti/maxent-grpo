# E49T — menu-inferred hard-MATH Haarnoja versus matched Dr.GRPO

**Status: PREREGISTERED BEFORE TRAINING — 2026-07-26**

## Motivation

E49S certified a 100-problem hard-MATH toy with 20 two-route problems, but
the matched three-epoch run provided no learning signal: Qwen2.5-0.5B ignored
the XML action-trace syntax, all answer-positive rollouts failed the exact
parser, the gated reward was zero, and the E46 controller received no
observations. E49T removes that formatting bottleneck without weakening the
mathematical route contract.

## Frozen task contract

E49T uses the same 100 E49S problems, answers, finite action menus, and
certified support. Only the response-format instruction changes. The model
chooses one listed strategy and writes a natural derivation. An ordinary MATH
validator must first accept the final answer. Then two independently
permuted, temperature-zero Qwen2.5-72B audits must unanimously find that the
written mathematics correctly and sufficiently executes every action of the
same one listed combo. Any error, omission, mixed/unlisted route, ambiguity,
disagreement, or malformed audit gates the task reward to zero. The
canonicalizer cannot emit a strategy outside the prompt's frozen menu.

Training may start only after both frozen E49T calibrations pass: the main
route-confusion calibration's false-route, duplicate, negative-control,
recall, dual-support, and schema gates, and the supplemental declaration
mismatch veto proving that a claimed strategy/combo must match the route
actually executed.

## Matched comparison

Both arms use Qwen2.5-0.5B-Instruct, seed 45, the same initial checkpoint,
data order, 16 samples per prompt, verifier, 72B route gate, optimizer,
learning rate 2e-7, generation limits, one A100 each, 50 train prompts per
epoch, and exactly three prompt epochs.

- `grpo`: ordinary Dr.GRPO after the shared task/route gate.
- `online_canonical_haarnoja`: the current E46 normalized canonical-bank
  treatment, with alpha 0.10 initially, novelty beta 0.50, normalized target
  `H(q_x)/log|B_x^+| = 0.80`, Haarnoja log-alpha Adam learning rate 0.003,
  EMA 0.90, and alpha projected to [0.10, 0.50].

No policy-entropy adaptation or later controller variant is enabled.

## Advancement criteria

The toy establishes viability only if both arms complete all 150 updates and:

1. the shared gate admits validator-positive natural derivations at nonzero
   rate;
2. the treatment observes at least one eligible two-strategy prompt and its
   canonical bank reaches support two;
3. the Haarnoja controller takes nonzero observations and optimizer steps;
4. task accuracy does not collapse relative to matched Dr.GRPO;
5. treatment strategy coverage/support exceeds or preserves the matched
   control in the prespecified evaluation summaries; and
6. the mechanism telemetry shows the same qualitative chain as successful
   Countdown/graph-coloring runs: validated discoveries grow support,
   normalized entropy is measured on that support, and alpha responds to the
   signed normalized-entropy error.

Only a passing toy treatment may be extended to the exact OAT
384-train/MATH-500-eval, three-epoch comparison.
