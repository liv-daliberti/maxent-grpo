# E101 zero-step fresh-sample repair

Recorded: 2026-08-14 before any E101 GPU allocation or metric. Jobs 30579498,
30579499, and 30579500 are held at zero steps and will be superseded.

A CPU audit found that Countdown's validator accepts deterministic sign rewrites
that the dataset's exact enumerator does not include. For numbers [2, 3, 4] and
target 10, the enumerated support has two keys and contains `2*3+4`, while the
proposal transformer also validates `(-2)*(-3)+4` and `2*3-(-4)` as distinct
keys outside that support. Training these rows could inflate distinct-mode
metrics through a verifier/support mismatch.

The clean replacement adds a default-on transform-proposal switch, preserving
every historical variant. E101 alone sets it off. The open arm now makes at
most one isolated sample from the untouched original prompt at temperature 1,
then independently validates, canonicalizes, and admits a genuinely sampled
new key to replay-only support. Proposal rows remain absent from PPO and neutral
counts. All other objective, data, seed, evaluation, and wall-time settings are
unchanged. Replacement jobs must use a new immutable source snapshot and new
job IDs; the initial ledger is preserved as superseded provenance.
