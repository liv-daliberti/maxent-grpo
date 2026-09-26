# E11 prospective engineering gate: standard sequence MaxEnt

**Status: EXPLORATORY ENGINEERING GATE, FROZEN BEFORE E11 OUTCOMES
(2026-07-17).** E11 preserves E9--E10 metrics as normalized-objective
provenance and changes the active objective to

```text
J(theta) = E[R] + alpha H(pi_theta).
```

The exclusive old/new prefix-ratio estimator from E10 is retained. Its
unclipped finite-horizon value is raw sequence entropy

```text
H(pi_new) = E_{Y ~ pi_old} sum_t W_(t-1) h_t.
```

Dr.GRPO's single shared `1/T_max` update normalization remains outside both
reward and entropy. There is no second, objective-level division of entropy by
`T_max`. Controllers observe raw sequence nats. A separately named
`maxent_sequence_entropy_per_tmax` diagnostic preserves comparison to E9/E10,
and old `sequence_nats_per_tmax` controller checkpoints are incompatible.

## Sequential smoke

The literal-coefficient gate runs first: graph coloring,
Qwen2.5-0.5B-Instruct, seed 9005, group size 16, four-row backward
microbatches, one PPO epoch, PPO clip 0.2, 32 updates, and fixed standard
`alpha=0.05`. This coefficient is intentionally not rescaled: it tests the
objective and units requested for the maintained fixed treatment. The run
must complete with finite raw and normalized entropy telemetry, preserve the
identity `raw / 192 = per_tmax`, produce positive reward in its trailing 16
updates, end with mean response length in `(2, 64]`, have at most 2/16 no-EOS
rollouts, and retain final evaluation accuracy above 0.05.

Failure stops the sequence. Passing authorizes, but does not automatically
submit, a 128-update fixed/proportional/Haarnoja comparison. That comparison
uses raw target `9.9680345401152`, exactly 192 times E9b's independently
measured normalized target, base `alpha=0.05`, maximum `0.5`, proportional
relative-deficit gain 1, and dual alpha learning rate 0.03. Both adaptive arms
must retain at least 50% of the raw target over their final 16 updates and
meet the same response, reward, accuracy, and telemetry guardrails.

Only a complete pass of both stages authorizes review of the 54-cell E11
analytical grid. No stage automatically submits that grid.

## Observed literal-gate outcome (appended after the run)

Job 30007745 started from the frozen E11 source snapshot on node023. OAT
continued beyond the requested 32-row ceiling, so the allocation was cancelled
after step 38 and its deferred watchdog requeue was cancelled before
allocation; only the prospectively specified step-32 endpoint enters the gate.
The raw and normalized telemetry agreed throughout. At step 32, raw
sequence entropy was 62.4276 nats and the separate normalized diagnostic was
0.325144, satisfying `62.4276 / 192 = 0.325144`.

The literal `alpha=0.05` treatment failed the behavioral guard: terminal mean
response length was 87 tokens and 7/16 rollouts lacked EOS. Trailing reward
remained positive (step-32 batch reward 0.4375), and final evaluation accuracy
was 0.05208, barely above the accuracy floor. Thus this is entropy/length
runaway, not a missing gradient or zero-reward short-output collapse.

The adaptive comparison was not submitted. All 54 analytical E11 rows remain
held. The standard objective stays implemented, but the literal coefficient
is not an experimentally viable fixed treatment for this reward/horizon scale.
