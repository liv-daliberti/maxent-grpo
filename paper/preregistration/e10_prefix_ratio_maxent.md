# E10 prospective engineering plan: prefix-ratio direct MaxEnt

**Status: EXPLORATORY ENGINEERING GATE, FROZEN BEFORE E10 OUTCOMES
(2026-07-17).** E10 preserves all E9/E9b traces and the appendix objective.
It changes only the finite-step PPO surrogate used to estimate new-policy
sequence entropy from old-policy rollout prefixes.

## Estimator

Rollouts are sampled from `pi_old`, while the direct objective contains
`H(pi_new)`. For local categorical entropy `h_t = H(pi_new(. | s_t))`, define
the exclusive prefix ratio

```text
W_{t-1} = pi_new(y_<t | x) / pi_old(y_<t | x)
        = product_{j<t} exp(log pi_new(y_j|s_j) - log pi_old(y_j|s_j)).
```

Then the exact finite-horizon importance-sampled identity is

```text
H(pi_new) = E_{trajectory ~ pi_old} [sum_t W_{t-1} h_t].
```

Differentiating `W_{t-1}` supplies the causal state-visitation derivative;
differentiating `h_t` supplies the full-categorical local derivative. The
active PPO surrogate replaces each positive term `W h` by
`min(W h, clip(W, 1-epsilon, 1+epsilon) h)` with the same epsilon as the
reward PPO update. The first-token prefix ratio is exactly one. No current-
action ratio is multiplied into `h_t`, because the state entropy is defined
before action `y_t` is sampled.

Before training, exact enumeration under distinct old and new two-step
autoregressive Bernoulli policies must verify both the value and gradient of
the *unclipped* identity. Separate tests must verify exclusive-prefix indexing,
masking, and the upper PPO clipping branch.

## Smoke design

All smokes use graph coloring, Qwen2.5-0.5B-Instruct, seed 9005, group size 16,
four-row backward microbatches, one PPO epoch, 128 updates, and PPO clip 0.2.
Because `W=1` at the rollout policy, E10 reuses E9b's independently measured
frozen-policy target `H_target=0.0519168465631`.

The comparative smoke contains:

1. fixed direct MaxEnt at alpha 0.05;
2. E9b relative-deficit proportional control, base alpha 0.05 and maximum 0.5;
3. E9b immediate Haarnoja dual control, base alpha 0.05, maximum 0.5, and
   alpha learning rate 0.03.

A separate fixed-alpha 0.50 stress arm tests whether prefix-ratio clipping
prevents E9b's entropy/length runaway.

Every arm must complete 128 updates with finite prefix-ratio, clipping,
entropy, loss, gradient, reward, and response telemetry; produce a positive
rollout reward in the trailing 32 updates; end with mean response length in
`(2, 64]`; have at most 2 of 16 terminal rollouts without EOS; and retain
final average evaluation accuracy above 0.05. Each adaptive arm must retain at
least 50% of the frozen target over its final 16 updates. The alpha-0.50 stress
arm is a guardrail and is not an analytical method.

Any failure keeps all 54 analytical direct-MaxEnt rows held. Passing authorizes
design review, not automatic analytical-grid submission.

## Observed outcome (appended after all four jobs completed)

Jobs 30007613--30007616 completed 128 updates on node023 with exit code zero.
All new prefix-ratio, clipping, entropy, loss, gradient, reward, and response
telemetry remained finite. Exact enumeration and the maintained test suite had
already passed before submission.

The fixed, proportional, and dual comparative arms retained 24.7%, 17.7%,
and 11.7% of the frozen target over their final 16 updates. Proportional and
dual therefore fail the prospectively specified 50% entropy-retention gate.
Their terminal mean lengths were 4.00, 4.12, and 4.00; terminal evaluation
accuracies were 0.188, 0.172, and 0.182; and all retained positive trailing
reward. The failure is not numerical instability or terminal reward collapse.

The ratio-corrected gradient did produce large transient entropy and length
excursions. Peak normalized entropy was 0.273, 0.169, and 0.173 for fixed,
proportional, and dual, with peak mean lengths 40.25, 38.94, and 39.00. The
controllers did not stabilize those excursions: proportional and dual ended
at alpha 0.321 and 0.317 while normalized entropy had returned to 0.00618 and
0.00412.

The alpha-0.50 stress arm failed its guardrail decisively. Its final normalized
entropy was 0.752 (tail mean 0.631, 12.15 times target), terminal mean length
was 168.75 of 192 tokens, 14/16 rollouts lacked EOS, and final evaluation
accuracy was zero. Its terminal prefix ratio was one, so prefix-ratio clipping
cannot bound the local categorical entropy/length incentive when behavior and
learner policies are aligned.

E10 therefore repairs E9's finite-step off-policy entropy gradient but fails
the scientific control gate. All 54 analytical direct-MaxEnt rows remain held.
