# E7 prospective repair: fixed-scale Candidate MaxEnt projection

**Status: EXPLORATORY, FROZEN AFTER THE E6 FAILURE AND BEFORE ANY E7 TRAINING
OUTCOME (2026-07-17).** E6 exposed a structural error in the appendix
projection rather than a temperature-controller failure. This plan defines
the correction and keeps all E6 traces under their original stamps.

**Dated operational amendment (after the smoke, 2026-07-17): GATE PASSED.**
Jobs 30006354--30006356 reached 128 updates with exit 0. Terminal response
lengths were 3.88/15.75/4.00 and maximum trailing-window rewards were
0.8125/0.9375/0.875 for fixed/proportional/dual. All finiteness and controller
criteria passed, so the pre-authorized 54-run grid was submitted as jobs
30006358--30006411. This records gate execution; it does not alter outcomes or
the analysis plan below.

## Failure diagnosis and correction

E6 optimized

```text
-sum_i w*_xi log pi(y_xi | x) / T_xi.
```

For an on-policy uniform target, the expected unscaled score gradient is zero,
`E[grad log pi(Y)] = 0`. Candidate-local length normalization instead gives
`E[grad log pi(Y) / T(Y)] = grad E[1 / T(Y)]`; minimizing the negative loss
therefore rewards short samples even when every reward is identical. The
observed two-token/EOS collapse is the predicted consequence.

E7 uses the forward-KL projection with one shared constant:

```text
L_projection = -mean_x sum_i stopgrad(w*_xi)
                         * log pi_theta(y_xi | x) / T_max.
```

The positive `1/T_max` factor only rescales the objective and cannot change its
minimizer or introduce candidate-length preferences. The active prompt set is
restricted to informative mixed-reward groups (`max_i |A_xi| > 0`). Constant-
reward groups have no group-relative preference, so excluding them removes
finite-sample self-distillation without discarding a reward contrast.

The target is otherwise unchanged:

```text
u_xi = A_xi * T_xi / T_max
q_x = softmax(u_x / 1.0)
w*_x proportional to q_x ** (1 / tau_target)
```

At fixed `tau_target=0.05`, `w* = softmax(u/0.05)`, exactly matching fixed
xDr's candidate weights on every informative group. Reference tilt remains
disabled (`beta=0`), rollout temperature remains 1, and exact-answer or
semantic labels never enter training.

## Methods and grid

The three method names remain `xdr_maxent`, `xdr_maxent_tau_control`, and
`xdr_maxent_sac_dual`; E7 stamps, not method names, identify the repaired
objective. Fixed, proportional, and Haarnoja-dual temperature settings match
E6 exactly. The intended analytical grid remains three methods x three seeds x
two environments x three scales = 54 runs, capped at five prompt-pool passes.

## Mandatory staged launch

Before the 54-run grid can be submitted, all three methods must complete a
128-step graph-coloring 0.5B smoke. This smoke is operational and excluded from
endpoint analysis. It passes only if, at the terminal step, every arm has:

- finite loss, entropy, gradient norm, and controller telemetry;
- mean response length greater than 2 tokens;
- nonzero rollout reward in the trailing 32 steps; and
- no monotone convergence to the two-token all-zero state seen in E6.

If any criterion fails, no 3B or 7B E7 job is launched. If the smoke passes,
the full grid uses stamps beginning `cde7_maxent_fixedscale_` and
`gce7_maxent_fixedscale_`. Primary and secondary comparisons, controller
settings, matched seeds, evaluation cadence, accuracy guardrails, and
five-pass endpoint rules are inherited unchanged from E6.

E7 is a post-failure repair and must never be presented as the originally
pre-specified E6 objective. E6 is a negative pilot; E7 outcomes, if launched,
are a distinct exploratory experiment.
