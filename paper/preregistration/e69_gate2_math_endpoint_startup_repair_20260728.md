# E69 Gate 2 free-form-MATH endpoint startup repair

Date frozen: 2026-07-28, before the replacement job was submitted and before
any Gate 2 terminal outcome was available.

## Failure

Gate 2 job `30159730`, the seed-43 free-form-MATH endpoint-only arm, never
entered optimization. It failed argument validation on every startup attempt:

`semantic_shannon_separate_advantage requires a positive semantic_shannon_coef`

The job had no run directory, no optimizer metric record, no checkpoint, and no
evaluation result. Its repeated starts were watchdog retries of the same
pre-optimizer validation failure, not outcome-bearing attempts. It was
cancelled after five recorded retries.

## Cause and single correction

The Gate 2 domain launcher set the MATH semantic coefficient to zero, while the
frozen `verified_first_global_replay_canonical` variant enables its separated
semantic advantage by construction. That combination is invalid. The
prospectively named comparison arm is E66 endpoint-only replay; its frozen
coefficient in the parent E66 implementation is `0.10`.

The replacement changes only
`OAT_ZERO_SEMANTIC_SHANNON_COEF` from `0` to `0.10` for the endpoint-only MATH
job. This restores the already-defined E66 arm and satisfies the typed runtime
contract. It does not enable route prompting, verified-route identity, route
novelty, counterfactual proposals, or cross-prompt route replay.

All model, data, seed, six-pass stopping, neutral rollouts, three fixed
sampling-control groups, replay capacity, optimizer, verifier, evaluation,
checkpoint, source snapshot, execution snapshot, placement, and charged
compute settings remain exactly those in the Gate 2 identity.

## Attempt selection

Job `30159730` is permanently excluded as a pre-optimizer infrastructure
failure. Exactly one replacement job is submitted held, audited, and recorded
in `e69_gate2_math_endpoint_repair_identity.json`. The Gate 2 analyzer replaces
only that physical identity cell; it never combines the invalid and replacement
attempts. No other Gate 2 job is changed.

The replacement is an infrastructure/configuration repair under the original
scientific arm, not a new algorithm or outcome-dependent hyperparameter
choice. Any replacement that does not use the exact frozen snapshots and the
single correction above is ineligible.
