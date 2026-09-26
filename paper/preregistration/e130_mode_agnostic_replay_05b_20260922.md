# E130 preregistration: mode-agnostic replay (capacity one) on the Qwen2.5-0.5B panel

Date frozen: 2026-09-22, before submission of any E130 cell.

## The question this answers

Re:Dr adds two things to Dr.GRPO at once: a supervised likelihood term on the
policy's own past verified outputs, and the rule that the term is keyed by
canonical mode and rehearsed uniformly across the distinct modes a prompt has
discovered. A reader can accept the second and still ask whether the first
alone explains the result, and the sharpest form of that objection is
quantitative. On Qwen2.5-0.5B `PythonFactors`, only 1.5% of Dr.GRPO rollout
groups contain both a correct and an incorrect response
(App. "Fresh-gradient degeneracy"), so 98.5% of that arm's updates carry no
fresh task gradient at all. In that regime almost any likelihood term on a
past success would move the policy, and the accuracy gain could be an artifact
of supervision arriving where the task objective supplies none.

E130 separates the two. It is the Re:Dr objective with bank capacity set to
one. Every other quantity is held: the same replay loss, the same
`verified_likelihood_per_rollout` objective, the same coefficient
`alpha = 0.10`, the same one scheduled bank per optimizer update, the same
admission rule, the same schedule, and the same evaluation. The bank simply
retains the first canonical mode each prompt discovers and no other, so the
replay loss becomes a length-normalized likelihood term on exactly one stored
success per prompt, with nothing to balance across.

This is a different ablation from E120-R1. E120-R1 kept one exemplar per
discovered mode and changed the weights within the bank from uniform to
fresh-observation frequency. E130 changes what the bank contains: with one
slot, mode identity can no longer enter the objective at all.

## Registered predictions

1. **Accuracy.** Terminal `pass@8` for E130 is above its matched control and
   below Re:Dr. A one-mode likelihood term is expected to supply some of the
   missing gradient in sparse-signal domains, so a null accuracy effect is not
   predicted and would not be needed for the comparison to be informative.
2. **Diversity.** Terminal PCMD for E130 does not differ systematically from
   its matched control. With one slot the objective never sees a second key,
   so the mechanism the paper credits for breadth is absent by construction.
3. **Separation.** The E130-minus-control PCMD effect is smaller than the
   Re:Dr-minus-control PCMD effect in at least four of the five domains.

Disconfirmation of (2) or (3) --- in particular an E130 PCMD gain comparable to
Re:Dr's --- would show that uniform rehearsal across discovered modes is not
what produces the breadth result, and the paper's mechanism claim would have to
be rewritten rather than defended. Prediction (1) is registered so that a large
E130 accuracy gain is reported as a finding about supervision in sparse-reward
regimes, not treated as a failure of the ablation.

## Frozen cohort

- Model and initialization: Qwen2.5-0.5B-Instruct, exactly as E128.
- Objective: `e78.fixed_objective("control")` with exactly three keys moved ---
  the variant to `verified_first_replay_rehearsal_only`, the replay derivative
  on (`COMPUTE_ONLY=0`), and the capacity to `1`. The launcher refuses any
  further drift, so "only the bank size differs from Re:Dr" is machine-checked.
- Domains: Graph Coloring, Countdown, PythonFactors, MathIR, PantryPlan.
- Seeds: 43, 44, 45, 46, 47 (25 scientific cells).
- Training: 384 prompts, 8 passes, 16 rollouts per prompt, learning rate 2e-7,
  beta=0, one PPO epoch, and the E78 decoding/evaluation schedule.
- Runtime: derived from `e76_tuned_scale_970e16dc21f47834`, the same base
  E126, E127, and E128 derive from, with the same eight patched files. The
  resulting tree differs from the snapshot those three ran under,
  `diversity_comparators_31f405a798a91bd3`, by exactly one hunk: the
  capacity-floor relaxation declared below in `args.py`. Every other
  patched file is byte-identical, verified by comparison before
  submission. The relaxation is unreachable for any capacity of two or
  more, so it cannot change a control cell, and the two cohorts are
  runtime-matched everywhere the code can execute.
- Placement: `cs` partition, `allcs` account, A5000, node203 or node204 ---
  the same two nodes E128 ran on. Partition and account are audited from the
  scheduler's own record before release.
- Control: the completed E128 cells, paired within domain and seed. E78 is not
  the control here, for the runtime, placement, and evaluator reasons the E128
  preregistration records.

## The one declared source change

`validate_zero_math_args` refused `online_canonical_replay_capacity < 2`. That
floor belongs to the legacy `bank_balance` objective, whose
`KL(U_g || softmax(scores_g))` term is undefined on a one-mode group and whose
loss function raises on one. The objective this campaign trains under,
`verified_likelihood_per_rollout`, computes the mass term alone, which already
scores one-mode groups in ordinary training: any prompt that has discovered
exactly one mode is replayed as a singleton on every visit. The floor is
therefore made objective-dependent --- two for `bank_balance`, one for the mass
objectives --- rather than removed. `tests/test_args.py` covers both sides.

No learner, replay, admission, or scheduling code is touched. A capacity of one
exercises paths the existing arms already exercise.

## Outcomes and reporting

Report terminal `pass@8` and PCMD on all five seeds at the registered K=8
evaluation, paired against E128 within domain and seed, under the same
20-prompt paired support threshold the alternatives table uses. Report the
Re:Dr effect beside it, and state that Re:Dr's control is E78 while E130's is
E128, exactly as the GAPO and SetPO rows already do.

Mean selected-bank occupancy is recorded per run. It is expected to be 1.00 by
construction; a value above 1.00 is a mechanism failure, not a result.

## Failure rules

Submission is held until the snapshot and every scheduler export are audited.
A failed 32-query Graph/s43 smoke blocks all 25 scientific jobs. A scientific
cell may be retried only for an infrastructure failure under the same frozen
configuration. If E130 cannot be completed, the appendix states the question as
open and does not substitute E120-R1, which answers a different one.
