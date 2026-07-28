# E47H-CAL: schema-bound MATH strategy calibration

**Status: FROZEN BEFORE LAUNCH — 2026-07-24**

E47G exposed a second production-blocking structural issue before completing
calibration or producing manual labels. On problem 21, Qwen72 placed an
unescaped LaTeX backslash in the optional free-text `strategy` description,
making its nominal JSON response invalid. Strict parsing rejected the output
and no key was admitted. At thousands of online calls, leaving an unused prose
field unconstrained would make eventual job failure likely.

E47H removes that non-operative field. The requested and server-constrained
response contains only:

```
{"clusters":[{"cluster_id":"c1","member_ids":["opaque id"]}],
 "ambiguous_ids":[]}
```

Each request uses vLLM's JSON-schema response format. The schema forbids extra
fields, restricts member and ambiguous values to the exact opaque IDs in that
request, and guarantees syntactically valid JSON. The existing strict parser
still requires every ID exactly once across clusters and ambiguity, rejects
duplicates or omissions according to the frozen omission rule, and rejects
any semantic conflict among stored representatives. Guided structure changes
only serialization: the two Qwen72 passes still decide all memberships using
the same prompt definition, item permutations, temperature zero, and seeds
`470721` and `470722`.

All E47G data, bounded online schedule, validator-positive gate, transitive
union-component admission rule, audit construction, and numerical gates are
unchanged. The persistent state schema is
`math_strategy_canonicalizer_two_pass_union_v6`. Calibration and training
import the same implementation. E47G is preserved as a failed partial and
cannot advance E49.

E49 may launch only from a complete
`var/artifacts/e47h_bounded_math_strategy_calibration_v1` artifact whose
unchanged frozen gates all pass.
