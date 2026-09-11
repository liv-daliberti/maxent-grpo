# E49V — exact OAT hard-MATH natural-menu extension

**Status: PREREGISTERED BEFORE PREPROCESSING — 2026-07-26**

## Scope and advancement gate

E49V is the full-data extension of E49T.  It may be materialized or trained
only after the matched E49T toy completes all 150 updates, passes its
answer-plus-route gate, reaches canonical support two, produces nonzero
normalized-entropy observations and Haarnoja optimizer steps, preserves task
quality and finite-menu route coverage versus matched Dr.GRPO, and receives a
positive frozen advancement decision.

## Exact data identity

- Training rows are exactly the 384 rows, in their frozen order, from
  `var/data/math12k_384_math500/train`.
- Evaluation rows are exactly all 500 MATH-500 rows, in their frozen order,
  from `var/data/math12k_384_math500/eval`.
- The 50 E49T train and 50 E49T evaluation menus are overlaid only when their
  original source identities match exactly.  This preserves all 20 certified
  dual-route problems.
- Every remaining source row receives one finite singleton action combo.  A
  singleton is generated without seeing the reference answer, then two
  independent temperature-zero Qwen2.5-72B auditors must literally execute
  all of its actions, derive the auditor-only reference answer, find no hidden
  decisive step, and find that the menu does not reveal the final answer.
  Any disagreement, ambiguity, malformed output, incompleteness, or answer
  mismatch fails closed and is retried without answer-bearing feedback.

The final artifact must contain 384 train and 500 eval rows, all 884 menus
must parse and bind to their source problems and answers, exactly 100 menus
must be inherited from E49T, exactly 20 menus must retain dual support, no row
may have zero support, and every formatted prompt must fit the frozen
2,048-token prompt limit.

## Matched three-epoch comparison

After materialization, run Qwen2.5-0.5B-Instruct seed 45 for exactly three
epochs (1,152 optimizer updates), 16 samples per prompt, learning rate 2e-7,
one A100 per arm, and full MATH-500 evaluation at steps 0, 384, 768, and
1,152.

- Control: ordinary Dr.GRPO behind the shared answer-plus-route gate.
- Treatment: the current E46 normalized canonical-bank Haarnoja method:
  novelty beta 0.50, alpha initialized at 0.10 and projected to
  [0.10, 0.50], normalized target `H(q_x)/log|B_x^+| = 0.80`, log-alpha Adam
  learning rate 0.003, and EMA 0.90.

Policy-entropy adaptation and later controller variants remain disabled.
The full result must report task quality, route-gate acceptance, canonical
support, normalized entropy, signed alpha-control telemetry, and route-aware
K=8 coverage beside the successful graph-coloring and Countdown mechanism
chain.
