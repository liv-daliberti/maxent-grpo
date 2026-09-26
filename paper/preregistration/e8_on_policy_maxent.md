# E8 prospective analysis plan: direct on-policy MaxEnt-GRPO

**Status: EXPLORATORY, FROZEN AFTER THE E7 AUDIT AND BEFORE ANY E8 TRAINING
OUTCOME (2026-07-17).** The direct objective, estimator, controller direction,
smoke gate, and matched grid below were fixed after cancelling the invalid E7
candidate-projection grid. No E8 smoke or analytical trajectory had been
generated when this plan was written.

## Why E7 was retired

E7 removed E6's candidate-local length bias but did not implement the desired
policy-MaxEnt treatment. For on-policy samples, its detached weighted-cloning
target converges to

```text
p_target(y) proportional to pi_old(y) * exp(u(y) / tau),
```

which is a KL-regularized policy-improvement target, not the optimizer of
`E_pi[u] + tau H(pi)`. In addition, both E7 controllers reused xDr's inverse
aggregation-temperature law: when observed policy entropy fell, they lowered
the positive entropy coefficient `tau_target`, sharpening the target in the
opposite direction from MaxEnt control. The controller observed token entropy
while acting on rollout-slot target entropy, so its sensor and constrained
quantity were also different.

Jobs 30006358--30006411 were cancelled after this audit. Partial traces are
retained as a superseded diagnostic and excluded from E8 analysis. E6 and E7
remain separately documented; neither is silently relabeled as E8.

## Direct objective and estimator

For prompt `x`, completion `Y ~ pi_theta(.|x)`, terminal verifier reward `R`,
and fixed response budget `T_max`, E8 targets

```text
J(theta) = E[R] + alpha * H(pi_theta(.|x)) / T_max.
```

The coefficient `alpha` directly multiplies completion-policy entropy. At
rollout policy `pi_t`, define the normalized sequence-surprisal sample

```text
s_i = -log pi_t(Y_i | x) / T_max.
```

For each group of independent on-policy samples, use the leave-one-out baseline

```text
b_i = mean_{j != i} s_j,
A_soft_i = A_DrGRPO_i + alpha * (s_i - b_i).
```

The other samples' mean is independent of row `i`'s score function, so this is
an unbiased prompt-local variance-reduction baseline. `s_i`, `b_i`, and
`A_soft_i` are detached; the ordinary Dr.GRPO/PPO likelihood ratio supplies
the score-function gradient. Every analytical E8 run uses one PPO epoch per
fresh rollout group, synchronizes the updated policy to the generator, and
then collects the next group. The shared `T_max` scale is constant across
candidates and cannot reproduce E6's candidate-local EOS preference.

All-equal reward groups retain a generally nonzero entropy advantage and are
not masked. No candidate target, forward-KL cloning loss, semantic label,
answer-mode key, reference-policy tilt, replay buffer, or critic is added.

## Three methods

1. `maxent`: fixed direct on-policy MaxEnt with `alpha = 0.05`.
2. `maxent_control`: the same objective with a one-sided proportional
   controller. The first 64 sequence-entropy observations use `alpha = 0.05`
   and set `H_target = 0.8 * mean(H_warmup)`. Thereafter
   `alpha = clip(0.05 * exp(20 * max(H_target - H_ema, 0)), 0.05, 0.5)`,
   with EMA decay 0.9. Low entropy therefore raises the entropy coefficient.
3. `maxent_dual`: the same objective with Haarnoja-style dual descent on
   `J(alpha) = alpha * (H_observed - H_target)`. It learns `log(alpha)` using
   Adam at 0.003 with betas (0.9, 0.999), bounds `alpha in [0.005, 0.5]`, the
   same 64-observation warmup, and the same 0.8 target ratio. The learned dual
   is the actor's entropy coefficient directly; there is no inverse mapping.

Both controllers observe `H(pi)/T_max`, exactly the quantity whose coefficient
they change. The existing xDr proportional and inverse-temperature controller
arms remain separate treatments and are not modified by E8.

## Matched grid and five-pass ceiling

Run all three methods with seeds 43, 44, and 45 in Countdown and graph coloring
at Qwen2.5-0.5B-Instruct, Qwen2.5-3B-Instruct, and
Qwen2.5-7B-Instruct: 3 methods x 3 seeds x 2 environments x 3 scales = 54 E8
runs. Existing Dr.GRPO, fixed signed-surrogate xDr, and E4/E5 controller runs
are reused rather than resubmitted.

Every cell inherits the maintained compute-divergence recipe: identical prompt
pools, group sizes, seeds, data order, optimizer, learning rate, rollout
temperature 1.0, verifier, response budget, prompt-unit evaluation cadence,
and a hard ceiling of five complete prompt-pool passes. No model checkpoints
are retained; inline metrics preserve the trajectories.

## Mandatory smoke gate

Before the analytical grid is submitted, all three methods must complete a
128-update 0.5B graph-coloring smoke under stamp
`gce8_onpolicy_maxent_smoke_v1`, seed 9004. It passes only if every arm has:

- finite loss, gradient norm, token entropy, sequence entropy, entropy
  advantage, and alpha telemetry;
- positive `alpha` and positive normalized sequence entropy;
- terminal mean response length greater than two tokens;
- a nonzero rollout reward in the trailing 32 updates;
- no convergence to E6's two-token state; and
- controller telemetry crossing the 64-observation warmup for both adaptive
  arms.

Failure of any arm blocks all larger submissions. The smoke is operational and
excluded from outcome analysis.

## Outcomes and interpretation

The primary descriptive contrast is fixed direct MaxEnt minus Dr.GRPO at the
last common evaluation at or before five passes, paired by seed within each
environment and scale, on coverage@8. Controller branches are compared with
fixed direct MaxEnt under the same endpoint rule. Report all seed trajectories.

Secondary outcomes are pass@1, pass@8, mean@8, distinct@8, rollout reward,
normalized sequence entropy, logged token entropy, mixed/all-correct/all-zero
group fractions, entropy advantage magnitude, alpha, response length, and
gradient norm. Pass@1 and mean@8 remain accuracy guardrails.

An entropy or coverage gain with materially worse accuracy is a tradeoff. A
controller that raises alpha without stabilizing sequence entropy is actuator
failure. Because sequence entropy includes surface-form diversity, improved
sequence entropy alone does not establish semantic-mode preservation; exact
answer coverage supplies the behavioral endpoint.

## Maintained entry points

```bash
bash ops/exp_scaling/launch_on_policy_maxent_extension.sh smoke
bash ops/exp_scaling/launch_on_policy_maxent_extension.sh all
python ops/exp_scaling/refresh_on_policy_maxent_curves.py
python ops/exp_scaling/plot_divergence.py
```
