# E9b engineering plan: direct-MaxEnt actuator calibration

**Status: EXPLORATORY ENGINEERING GATE, FROZEN BEFORE E9B OUTCOMES
(2026-07-17).** E9b does not alter or pool the failed E9 smoke. It keeps the
E9 direct causal entropy estimator and appendix objective fixed, and tests
whether the entropy coefficient can actuate that objective on the 0.5B graph-
coloring smoke before any analytical grid is released.

## Diagnosis motivating the gate

E9 used normalized sequence entropy of about `0.03`. Its proportional rule
applied gain to the absolute entropy deficit, so even zero entropy could only
move `alpha` from `0.05` to about `0.096`; the configured `0.5` ceiling was
unreachable. Its dual controller remained frozen for 64 updates, then an Adam
learning rate of `0.003` could move log-alpha by only about `0.19` over the
remaining smoke. These are actuator-timescale failures, not changes to the E9
entropy estimator.

## Stage A: frozen-policy target and fixed-alpha doses

All runs use graph coloring, Qwen2.5-0.5B-Instruct, seed 9005, group size 16,
the E9 causal estimator, and four-row backward microbatches.

1. Estimate the reference entropy with 64 rollout/update cycles at learning
   rate zero. Define one shared fixed target
   `H_target = 0.8 * mean(maxent_sequence_entropy)` over those 64 observations.
2. Independently run 128-update fixed-alpha smokes at alpha in
   `{0.05, 0.10, 0.20, 0.50}`.

A dose is actuator-viable when all required telemetry is finite, final mean
response length is greater than two, at least one rollout reward is positive
in the trailing 32 updates, and trailing-16 normalized sequence entropy is at
least 50% of the frozen-policy target. Stage B remains blocked unless at least
one dose is viable. The fixed `alpha=0.05` branch remains the scientific fixed
treatment regardless of which larger diagnostic dose is viable.

## Stage B: repaired adaptive controllers

Both arms use the fixed target from Stage A and begin control on their first
observation; there is no learning-policy target warmup.

- Proportional control uses relative deficit
  `d = max(1 - H_ema / H_target, 0)` and
  `alpha = alpha_0 * exp(log(alpha_max / alpha_0) * d)`, clipped to
  `[alpha_0, alpha_max]`, with `alpha_0=0.05`, `alpha_max=0.5`, and entropy EMA
  decay `0.9`.
- Haarnoja dual control keeps the E9 signed log-alpha update, starts at
  `alpha=0.05`, uses bounds `[0.005, 0.5]`, and changes Adam learning rate from
  `0.003` to `0.03`.

Each arm runs 128 updates. It must satisfy the same finite telemetry, response-
length, and tail-reward checks as Stage A, and retain at least 50% of the fixed
target over its trailing 16 updates. Failure keeps all analytical direct-
MaxEnt rows held. Passing this engineering gate authorizes configuration
review, not automatic submission of the 54-run analytical grid.

## Observed outcome (appended after both stages completed)

The zero-learning-rate calibration produced the frozen target
`H_target=0.05191685` from its first 64 observations. OAT did not stop the
allocation at the requested 64 rows, so job 30007143 was cancelled after 84
training rows and its watchdog requeue was cancelled; only the prospectively
specified first 64 rows enter the target.

Fixed-dose jobs 30007144--30007147 completed 128 updates. Tail-16 target
retention was 11.5%, 21.4%, 25.2%, and 1223.9% for alpha 0.05, 0.10, 0.20,
and 0.50. The 0.50 arm therefore demonstrated actuator authority, but did so
pathologically: terminal mean response length was 145 tokens, 12/16 terminal
rollouts hit the length cap without EOS, and final evaluation accuracy was
zero. The lower three doses retained reward and four-token responses but did
not hold entropy.

Adaptive jobs 30007197 and 30007198 completed 128 updates. Proportional
control peaked at alpha 0.3266 but retained only 0.01516 tail entropy, or 29.2%
of target. Dual control initially reduced alpha on above-target batches,
reversed after collapse, and reached the 0.5 ceiling only near the endpoint;
it retained 0.00692 tail entropy, or 13.3% of target. Both arms retained
nonzero tail reward and ended with four-token responses. E9b therefore fails
its frozen adaptive gate. The analytical grid remains held; these traces are
engineering diagnostics, not paper outcome curves.
