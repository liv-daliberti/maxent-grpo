# E49E finite-kernel proposal cache-replay amendment

**Status: FROZEN BEFORE ANY FINITE-KERNEL AUGMENTATION REQUEST OR POLICY TRAINING — 2026-07-24**

Every cached finite-kernel proposal is deterministically replayed before use.
Replay verifies the fixed seed and contract version, completed response
identity, content hash, strict structured payload, finite kernel and operation
enums, unique kernel and operation-code combos, action budget, answer/numeric
leakage gates, and exact canonical menu hash. A stored `pass` bit cannot make
a changed proposal eligible.

The final repair/materialization stage independently rechecks that every
selected augmented candidate menu remains non-leaking before replaying its two
soundness audits and two equivalence attacks. Any mismatch stops the stage and
does not trigger resampling.
