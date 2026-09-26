# E47I-CAL: finite-schema MATH strategy calibration

**Status: FROZEN BEFORE LAUNCH — 2026-07-24**

E47H verified that the vLLM JSON-schema response format is accepted, but its
first guided calls exposed an avoidable performance boundary before completing
any problem: JSON arrays without explicit maxima may continue emitting
redundant legal objects toward the legacy 8,192-token cap. That is unsafe for
the thousands of calls in online training even though strict parsing would
reject duplicates. E47H is preserved as a performance-failed partial and
cannot advance E49.

E47I keeps the E47H membership-only schema and adds finite serialization
bounds derived only from the request size `n`: clusters, each member list, and
the ambiguity list have `maxItems=n`; member lists are nonempty; and cluster
IDs are selected from the finite enum `c1` through `cn`. The completion cap is
4,096 tokens. These constraints cannot change a valid partition of the
supplied IDs. They only exclude redundant, empty, or unbounded
representations that the strict executable parser would reject anyway.

All data, validator decisions, prompts, judge identity, two permutation
passes, transitive union rule, fail-closed representative handling, audit
construction, and numerical gates remain those frozen for E47H. The
persistent schema is `math_strategy_canonicalizer_two_pass_union_v7`.
Calibration and training import the identical implementation.

E49 may launch only from a complete
`var/artifacts/e47i_bounded_math_strategy_calibration_v1` artifact whose
unchanged frozen gates all pass.
