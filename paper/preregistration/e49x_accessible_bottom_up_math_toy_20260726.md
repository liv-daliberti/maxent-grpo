# E49X — accessible bottom-up hard-MATH Haarnoja toy

**Status: PREREGISTERED BEFORE MATERIALIZATION — 2026-07-26**

## Advancement gate

E49X may be materialized only if E49W passes its frozen requirement of ten
bidirectionally executable level-5 training problems.  It is the next matched
toy iteration required by E49U; it does not relabel or reuse any failed E49T
dual route.

## Frozen data construction

Training contains exactly 50 hard MATH prompts:

- retain, byte-for-byte at the finite-menu level, the 40 E49T training
  problems that already have singleton certified menus, in original order;
- discard the ten inaccessible E49T dual training menus; and
- append the deterministic first ten bidirectionally executable E49W level-5
  problems in E49W's frozen candidate order, using their double-audited menus
  and exact source rows from `math12k_384_math500`.

Evaluation retains the same 50 held-out E49T MATH-500 problems.  Existing
singleton menus remain unchanged.  A dual menu remains dual only when E49U
proved it bidirectionally executable.  Every other dual menu is pruned to the
route with the most E49U fully gated successes; ties choose S1.  Pruning
renumbers only the retained strategy/actions and changes no mathematical
operation.  The expected support is therefore ten dual training menus and
one dual evaluation menu.  All prompts must parse, bind to their original
answers, and fit the 2,048-token prompt limit.

## Matched comparison

Run Qwen2.5-0.5B-Instruct seed 45 for exactly three prompt epochs, 16 samples
per problem, and learning rate 2e-7.

- control: ordinary Dr.GRPO behind the shared answer-plus-route gate;
- treatment: the unchanged E46 normalized canonical-bank Haarnoja method
  with novelty beta 0.50, alpha 0.10 projected to [0.10, 0.50], normalized
  target 0.80, log-alpha Adam learning rate 0.003, and EMA 0.90.

Policy-entropy adaptation remains disabled.  Advancement requires exact
matched initialization, nonzero gated reward in both arms, treatment support
at least two, nonzero normalized-entropy observations and controller steps,
no greater than five-point terminal task-quality loss versus control, and
terminal 72B route coverage that is preserved or improved.
