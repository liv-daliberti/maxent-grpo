# E131 preregistration: extended-horizon Dr.GRPO control on the Qwen2.5-0.5B panel

Date frozen: 2026-09-22, before submission of any E131 cell.

## The question this answers

App. "What Verified Replay Costs" states the gap plainly: the factorial matches
optimizer updates and fresh rollouts, both arms run the replay traversal, and
the control's traversal is switched off at the derivative rather than
reallocated to further learning. No cohort has yet spent the control's share on
more training. Until one does, a reader may hold that replay's reported gain is
a compute artifact --- more gradient work per run, arriving under a different
name.

E131 spends it. It is the E128 control objective, unmodified, trained for
twelve prompt passes instead of eight: 4,608 optimizer updates against 3,072,
with 50% more fresh rollouts, on the same prompts, seeds, placement, runtime,
and evaluator. Nothing about the objective moves.

Twelve passes is deliberately generous rather than exactly matched. The
measured replay likelihood-pass subtotal is 0.19 to 1.64 times a single
reference forward per replay visit (App. "Token-based cost accounting"), which
is a small fraction of an update's rollout-plus-backward cost; 50% more of
everything is therefore strictly more additional computation than replay
consumes, and by a wide margin. An exactly matched control is not available,
because total training FLOPs were never measured; a control that overshoots in
the direction that favors it is available, and is what this cohort is. The
comparison it supports is one-sided, and is registered as one-sided.

## Registered predictions

1. **The gap does not close.** Terminal `pass@8` at pass 12 for E131, averaged
   over the five domains within seed, remains below terminal Re:Dr `pass@8` at
   pass 8.
2. **Diversity does not recover.** Terminal PCMD for E131 at pass 12 does not
   differ systematically from the same control at pass 8. Longer training under
   an objective indifferent to which correct mode is produced is predicted to
   concentrate further or hold, not to broaden.
3. **`PythonFactors` is the sharpest case.** In the domain where 98.5% of
   control updates carry no fresh task gradient, the additional 1,536 updates
   are predicted to change terminal `pass@8` by less than the additional passes
   would suggest, because a zero task gradient cannot be made nonzero by
   granting more of them.

Disconfirmation of (1) --- an extended control that reaches Re:Dr's accuracy ---
would establish that the accuracy half of the result is a compute effect, and
the paper would have to report it as such. Prediction (2) is separable: the
extended control could close some of the accuracy gap and none of the diversity
gap, and that outcome is informative rather than a failure.

Trajectories are recorded at every half pass, so the pass-8 endpoint of each
E131 cell is also observed. It is a within-cohort consistency check against
E128 and not an additional independent sample.

## Frozen cohort

- Model and initialization: Qwen2.5-0.5B-Instruct, exactly as E128.
- Objective: `e78.fixed_objective("control")` with exactly two keys moved, both
  of them the horizon: `NUM_PROMPT_EPOCH` and `MAX_PROMPT_EPOCHS` to 12. The
  launcher refuses any other drift, so "only the horizon differs" is
  machine-checked.
- Domains: Graph Coloring, Countdown, PythonFactors, MathIR, PantryPlan.
- Seeds: 43, 44, 45, 46, 47 (25 scientific cells).
- Training: 384 prompts, 12 passes (4,608 updates), 16 rollouts per prompt,
  learning rate 2e-7, beta=0, one PPO epoch, checkpoints and evaluations every
  192 updates as elsewhere.
- Runtime: derived from `e76_tuned_scale_970e16dc21f47834`, the same base
  E126, E127, and E128 derive from, with the same eight patched files. The
  resulting tree differs from the snapshot those three ran under,
  `diversity_comparators_31f405a798a91bd3`, by exactly one hunk: the
  capacity-floor relaxation declared below in `args.py`. Every other
  patched file is byte-identical, verified by comparison before
  submission. The relaxation is unreachable for any capacity of two or
  more, so it cannot change a control cell, and the two cohorts are
  runtime-matched everywhere the code can execute.
  E131 itself requests a capacity of sixteen, so it never reaches the
  relaxed branch at all; it is named here only because the snapshot digest
  differs from E128's and that difference must be accounted for rather
  than noticed later.
- Placement: `cs` partition, `allcs` account, A5000, node203 or node204 ---
  the same two nodes E128 ran on. Partition and account are audited from the
  scheduler's own record before release.
- Comparators: E128 at pass 8 for the horizon effect within the control, and
  the published Re:Dr terminal endpoints for the claim under test. The second
  comparison crosses runtime and placement in the way the E128 preregistration
  describes, and must be reported with that stated.

## What this does not establish

E131 is not an equal-FLOP comparison and is not reported as one. It shows what
the control does when given more of the resource it is accused of being denied,
in a direction that favors the control. It does not measure rollout generation
cost, wall-clock time, or the exact computational price of replay, none of
which this campaign has instrumented end to end.

A longer horizon is also not the only way to spend a budget. Larger fresh
groups and additional PPO epochs over the same rollouts are separate
alternatives, and remain unrun.

## Failure rules

Submission is held until the snapshot and every scheduler export, including the
twelve-pass horizon keys, are audited from the scheduler's own record. A failed
32-query Graph/s43 smoke blocks all 25 scientific jobs. A scientific cell may be
retried only for an infrastructure failure under the same frozen configuration.
If E131 cannot be completed, App. "The missing comparison at equal total
compute" stands as written.
