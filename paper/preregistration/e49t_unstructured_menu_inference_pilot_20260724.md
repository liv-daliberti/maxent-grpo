# E49T — finite-menu inference bootstrap pilot

**Status: FROZEN BEFORE PILOT CALLS — 2026-07-24**

E49S v1 produced an exact matched step-zero evaluation, but its first
answer-positive training rows were all rejected before semantic auditing
because Qwen2.5-0.5B ignored the XML trace syntax. This pilot measures a
strictly finite fallback on the already stored, immutable E49S Dr.GRPO
step-zero greedy responses.

The cohort is every response whose frozen ordinary MATH verifier score is
positive. For each such response, two independently permuted,
temperature-zero Qwen2.5-72B audits receive the problem's already audited
finite action menu and the natural derivation. Each audit must decide that the
written mathematics correctly and sufficiently executes every action in
exactly one listed combo. The response is admitted only when both audits
return `valid` and the same existing strategy ID. Invalid, ambiguous,
malformed, disagreeing, unlisted, or mixed-route responses receive no key.
The fallback cannot create a new strategy.

The report records strict-parser acceptance, unanimous inferred acceptance,
rejection counts, exact source/evaluator/endpoint identities, and hashes of
all per-row decisions. A matched training successor is eligible only if this
frozen cohort has nonzero unanimous acceptance and no runtime/schema failure.
