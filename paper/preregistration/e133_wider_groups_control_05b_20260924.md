# E133 preregistration: the same extra budget, spent on wider fresh groups

Date frozen: 2026-09-24, before submission of any E133 cell.

## The question this answers

App. "The control at a longer horizon" names three distinct ways a control
could spend replay's share of the budget on learning instead: additional
task-gradient passes over existing fresh groups, larger fresh-rollout groups,
and more fresh-sampling updates beyond the eight-pass horizon. E131 measured
the third. It gave the control twelve passes against eight and closed neither
gap.

The second is untested and is the one that threatens the diversity result
rather than the accuracy result. Extra passes give the policy more updates on
groups of the same width; wider groups change what a group can contain. With
sixteen rollouts a prompt discovers some number of canonical modes per visit,
and replay's claim is that retaining those across visits is what preserves
breadth. If simply drawing twenty-four rollouts per visit recovers the same
breadth, then the mechanism the paper credits is a sampling artifact, and
"sample wider" is a simpler explanation than "retain and rehearse". That is a
more threatening story than "train longer" and the paper cannot currently
rebut it.

E133 gives the control the **same extra budget E131 received, allocated
differently**: 24 fresh rollouts per prompt in place of 16, at the unchanged
eight-pass horizon. Both arms therefore sit at approximately $1.5\times$ the
control's training compute, one having spent it on more updates and the other
on wider groups, and both can be read against the same matched control.

The budget is generous by a wide margin and is known to be. Measured on
matched hardware, Re:Dr's own overhead over this control is $+2.4\%$ of total
learner compute and $+1.1\%$ of wall clock
(App.~\ref{app:compute-accounting}); $+50\%$ is roughly nineteen times that
margin.

## What moves

Three coupled keys, and nothing else:

- `OAT_ZERO_NUM_SAMPLES`: 16 to 24.
- `OAT_ZERO_TRAIN_BATCH_SIZE`: 16 to 24, so one optimizer update still consumes
  exactly one prompt's group.
- `OAT_ZERO_PI_BUFFER_MAXLEN_PER_DEVICE`: 16 to 24, matching the group.

The update count is therefore unchanged at 3,072, which is the point: E131
held group width and raised update count, E133 holds update count and raises
group width. The replay traversal stays inert (`COMPUTE_ONLY=1`), as in every
control arm on this panel. These are declared through the launcher's override
mechanism so the drift guard records them; an undeclared fourth key is
refused.

## Registered predictions

1. **Diversity does not recover.** Terminal PCMD for E133 stays below half of
   the matched Re:Dr effect over this control, that is below $.19$ on the
   five-domain mean against Re:Dr's $.372$ and the control's $.004$. This is
   the prediction the cohort exists to test. Disconfirmation --- E133 reaching
   Re:Dr's breadth --- would mean the breadth result is reproducible by
   sampling width alone, and the mechanism claim in Sec. 5 would have to be
   rewritten around group size rather than retention.
2. **Correctness improves.** Terminal `pass@8` for E133 exceeds its matched
   control on the five-domain mean. Wider groups raise the chance a group
   contains both a correct and an incorrect response, so a real task gradient
   arrives on more updates; a null accuracy effect is not predicted and would
   not be needed for the comparison to be informative.
3. **The mechanism for (2) is visible.** The fraction of updates whose fresh
   group carries no task gradient falls relative to the control in every
   domain, and falls most in `PythonFactors`, where it is $98.5\%$ at sixteen
   rollouts (App.~\ref{app:degenerate-groups}). This is registered so that an
   accuracy gain is attributed to the degeneracy it removes rather than to
   breadth.

Predictions 1 and 3 can each overturn something already written. Prediction 2
is registered so an accuracy gain is not read as a failure of the control.

## Frozen cohort

- Model and initialization: Qwen2.5-0.5B-Instruct, exactly as E128.
- Objective: `e78.fixed_objective("control")` with the three keys above and no
  others.
- Domains: Graph Coloring, Countdown, PythonFactors, MathIR, PantryPlan.
- Seeds: 43, 44, 45, 46, 47 (25 scientific cells).
- Training: 384 prompts, 8 passes, **24** rollouts per prompt, learning rate
  2e-7, beta=0, one PPO epoch, 3,072 optimizer updates, and the E78
  decoding/evaluation schedule. Evaluation is untouched: mode-coverage draws
  are independent of training group width, so the estimator is the one every
  other cell uses.
- Runtime: the snapshot E130 and E132 ran under,
  `diversity_comparators_40628bb13a366baa`.
- Placement: `cs` partition, `allcs` account, A5000, node203 or node204 ---
  the same nodes E128, E130, E131 and E132 used.
- Control: the completed E128 cells, paired within domain and seed. The
  treatment reference is E132, the within-cohort Re:Dr.

## No source change

E133 declares no source change. Group width is an existing training parameter.
