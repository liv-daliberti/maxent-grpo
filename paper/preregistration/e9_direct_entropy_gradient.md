# E9 prospective analysis plan: direct-gradient on-policy MaxEnt-GRPO

**Status: EXPLORATORY, FROZEN AFTER THE E8 ACTUATOR AUDIT AND BEFORE ANY E9
TRAINING OUTCOME (2026-07-17).** The objective, causal entropy-gradient
estimator, controller sensor, smoke gate, and matched grid below were fixed
before submitting an E9 job. E8 metrics are retained as a failed estimator
diagnostic and are not pooled with E9.

## Why E8 was retired

E8 used the correct on-policy score identity in expectation, but carried
normalized sequence surprisal through a leave-one-out completion advantage.
That is a poor finite-sample actuator in the regime the method is meant to
prevent. As sampled groups concentrated, their centered-surprisal dispersion
collapsed: graph-coloring dual control moved `alpha` from about `0.05` to its
`0.5` ceiling while the carrier dispersion fell from about `0.095` to
`0.0011`; Countdown showed the same failure (`0.029` to `0.0034`). Raising a
coefficient cannot recover support absent from the sampled group.

Jobs 30006799--30006852 were cancelled after this audit. The 18 allocated
0.5B runs retain approximately 25 minutes of partial telemetry; the 36 queued
3B/7B jobs were cancelled before allocation. The completed E8 smoke and all
partial analytical traces remain under their original `*e8*` stamps.

## Objective and direct causal gradient

E9 preserves the appendix objective rather than changing its strength:

```text
J(theta) = E[R] + alpha * H(pi_theta(. | x)) / T_max.
```

For a sampled autoregressive completion, let
`h_t = H(pi_theta(. | x, y_<t))` be the full categorical entropy at a visited
prefix. The sequence-entropy identity and its causal derivative are

```text
H(Y | x) = E[sum_t h_t],

grad H(Y | x)
  = E[sum_t grad h_t
      + sum_t grad log pi(y_t | x, y_<t) * sum_{k>t} h_k].
```

The trainer therefore adds the differentiable per-trajectory surrogate

```text
S_H = sum_t [h_t + log pi(y_t | x, y_<t) * stopgrad(sum_{k>t} h_k)].
```

The first term sees every vocabulary probability at every sampled prefix and
retains a gradient even if all sampled completions are identical. The second
term accounts for how earlier actions change later prefix visitation. Entropy
is no longer added to the reward advantage and is not passed through the PPO
clip. One fresh PPO epoch per rollout group limits state-distribution lag.

Dr.GRPO's shared loss normalizer is `1/T_max`. E9 adds
`-alpha * S_H / T_max^2`, preserving the stated `E[R] + alpha H/T_max`
tradeoff up to that shared global scale. The maintained self-including group
reward baseline attenuates its expected reward gradient by `(G-1)/G`; E9
applies the same factor to the entropy term so `alpha` retains its stated
relative units without changing Dr.GRPO itself.

## Three methods and controller units

1. `maxent`: fixed direct MaxEnt with `alpha = 0.05`.
2. `maxent_control`: the same objective with the previously specified
   one-sided proportional rule, warmup 64, target ratio 0.8, and bounds
   `[0.05, 0.5]`.
3. `maxent_dual`: the same objective with Haarnoja-style Adam descent on
   `log(alpha)`, warmup 64, target ratio 0.8, learning rate 0.003, and bounds
   `[0.005, 0.5]`.

Both adaptive branches now observe

```text
mean_Y [sum_t H(pi(. | x, Y_<t)) / T_max],
```

the on-policy categorical estimate of the exact normalized sequence-entropy
quantity in the objective. Controller checkpoints record the unit tag
`sequence_nats_per_tmax`. They never observe centered-surprisal dispersion,
answer modes, rollout-slot entropy, or xDr aggregation weights.

## Matched grid and five-pass ceiling

After the smoke passes, run all three methods with seeds 43, 44, and 45 in
Countdown and graph coloring at Qwen2.5-0.5B-Instruct,
Qwen2.5-3B-Instruct, and Qwen2.5-7B-Instruct: 54 E9 runs. Every cell inherits
the maintained compute-divergence recipe and a hard ceiling of five complete
prompt-pool passes. E8 checkpoints are not resumed and E8 metrics are not
included in E9 curves.

## Mandatory smoke gate

Before any analytical E9 job is submitted, all three methods must complete a
128-update 0.5B graph-coloring smoke under stamp
`gce9_direct_maxent_smoke_v1`, seed 9005. Every arm must have:

- finite reward loss, gradient norm, token entropy, normalized categorical
  sequence entropy, causal entropy surrogate, causal score term, direct
  entropy loss, and alpha telemetry;
- the expected `(G-1)/G` reward-estimator scale and a nonzero entropy loss;
- positive alpha and positive normalized sequence entropy;
- terminal mean response length greater than two tokens;
- a nonzero rollout reward in the trailing 32 updates; and
- controller telemetry beyond the 64-observation warmup for adaptive arms.

Any failure blocks the analytical grid. The smoke is operational and excluded
from outcome analysis.

### Post-smoke fail-closed amendment (2026-07-17)

This amendment was written after observing the E9 smoke and is explicitly not
prospective evidence. The original finite-loss/EOS/reward checker passed, but
that gate was too weak for an adaptive entropy experiment: over the final 16
updates, proportional control retained only 0.01273 normalized entropy versus
its 0.03244 target (39.2%), and dual control retained 0.00630 versus 0.03668
(17.2%). Terminal response lengths were 4.00 and 4.12 tokens. The fixed arm
ended at 0.00565 entropy and 4.06 tokens. All arms still produced reward, so
this is actuator failure rather than E6's two-token zero-reward failure.

The maintained gate is therefore fail-closed before larger compute: each
adaptive arm must retain at least half of its own frozen target on average over
the final 16 smoke updates. The 50% floor is a post-smoke operational safeguard,
not a confirmatory threshold or a relabeling of the observed run. E9 fails this
amended actuator guard, and none of its 54 analytical jobs is released.

The first smoke allocation also established that full-vocabulary categorical
entropy does not fit a 16-row backward microbatch on a 24 GB GPU. Those attempts
failed before writing a training row. The replacement uses four-row
microbatches with gradient accumulation to preserve the same effective train
batch; this is an operational memory repair, not an objective or data change.

## Outcomes and interpretation

The endpoint, pairing, environments, seeds, scales, coverage@8 primary
outcome, accuracy guardrails, and secondary behavioral metrics are unchanged
from E8. E9 additionally reports the direct entropy loss and causal score
term. A coefficient that rises while categorical sequence entropy continues
to collapse is controller failure; unlike E8, it cannot be explained by loss
of between-completion surprisal dispersion alone.

## Maintained entry points

```bash
bash ops/exp_scaling/launch_on_policy_maxent_extension.sh smoke
bash ops/exp_scaling/launch_on_policy_maxent_extension.sh all
python ops/exp_scaling/refresh_on_policy_maxent_curves.py
python ops/exp_scaling/plot_divergence.py
```
