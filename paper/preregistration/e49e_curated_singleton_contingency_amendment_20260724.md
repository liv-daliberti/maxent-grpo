# E49E fixed curated-singleton coverage contingency

**Status: FROZEN BEFORE ANY ANSWER-BLIND V2 PROPOSAL OR CURATED-CONTRACT AUDIT REQUEST — 2026-07-24**

This contingency guarantees that the 100-row toy calibration can receive
coverage without weakening validation if the fixed answer-blind V2 proposer
still leaves zero-support rows.

The file
`ops/math_strategy_calibration/e49e_curated_singleton_contracts_toy.json`
contains one fixed problem-specific symbolic action contract for each of the
11 gaps identified before V1.  It is frozen before any V2 request and cannot
be edited in response to V2 outcomes.  No contract states a final answer or
an evaluated derived numeric value.  Each is checked by the same digit and
number-word leakage gates as V2.

After V2 is terminal, every successful V1/V2 repair is replay-validated and
carried without resampling.  A still-unresolved row may use only its
precommitted contract.  That contract must pass both unchanged Qwen2.5-72B
literal execution/soundness auditors and the independent exact-answer
validator.  A failed audit fails the stage; no human relabel, contract edit,
or alternate contract is allowed within this version.

Curated contracts are coverage-only singletons.  Their provenance is
recorded explicitly, and they cannot contribute novelty support, pairwise
strategy diversity, or normalized-entropy observations.  The blinded manual
false-new audit for multi-route rows remains unchanged.
