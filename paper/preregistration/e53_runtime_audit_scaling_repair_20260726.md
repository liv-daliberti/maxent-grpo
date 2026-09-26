# E53 runtime-audit scaling-field repair

**Status: FROZEN BEFORE PASS 0.5, AFTER ONE QUARTER-PASS GRAPH EVALUATION, AND BEFORE ANY TERMINAL WINDOW — 2026-07-26**

The original E53 auditor interpreted
`train/canonical_replay_backward_scale` as the complete scalar multiplying the
analytic score-gradient surrogate:

`alpha * ((N - 1) / N) * A`.

This mismatch was exposed by runtime telemetry after the first graph
quarter-pass evaluation had already landed. The correction below is fully
determined by the frozen implementation, engineering-smoke expectation, and
the algebraically redundant logged fields; it does not use that evaluation's
values or alter any behavioral criterion.

The frozen implementation and engineering-smoke contract instead log that
field as the mechanical DeepSpeed accumulation correction `A`. The other
factors are already logged independently as
`canonical_replay_alpha_used` and
`canonical_replay_reward_estimator_scale`. For E53, the exact expected tuple
is therefore:

- `canonical_replay_backward_scale = A = 16`;
- `canonical_replay_reward_estimator_scale = 15 / 16`;
- `canonical_replay_weighted_loss = replay_KL * alpha * 15 / 16`;
- `canonical_replay_chunk_size = 1`;
- `canonical_replay_score_passes = 2`.

The actual backward implementation still multiplies the analytic score
gradient by all three factors. This amendment repairs only the fail-closed
auditor's interpretation of an already frozen telemetry field. It does not
change source, execution snapshots, jobs, coefficients, capacity, behavioral
gates, evaluation, or Stage-A authorization criteria. The original auditor
remains immutable and identity-bound; the repaired auditor binds both the
original auditor and this amendment.
