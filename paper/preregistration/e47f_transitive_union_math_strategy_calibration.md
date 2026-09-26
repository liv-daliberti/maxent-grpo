# E47F-CAL: transitive two-pass-union MATH strategy calibration

**Status: FROZEN BEFORE LAUNCH — 2026-07-24**

## Reason for the prospective successor

E47E's partial diagnostic began while its executable used a local
representative-consensus implementation. Before looking at any E47E manual
audit labels or downstream learning result, one remaining edge case was
identified: pairwise disagreement can form a multi-hop chain across existing
representatives and new candidates. E47F freezes a single graph construction
that handles this edge case transitively. E47E remains an immutable partial
diagnostic and cannot advance training.

## Frozen executable rule

For every prompt-local online round, only solutions already accepted by the
ordinary full `math_verify` validator are candidates. One stored solution for
each previously admitted strategy is included as an existing representative.
Qwen2.5-72B-Instruct-AWQ partitions the randomly permuted set twice at
temperature zero, using seeds `470721` and `470722`.

An ID omitted by a pass is ambiguous for that pass. Any candidate ambiguous in
either pass receives no canonical key. If either pass marks an existing
representative ambiguous, or either pass merges two distinct existing
representatives, the complete round is rejected and the bank is unchanged.
Malformed transport or JSON is an unrecoverable error rather than a fallback.

For all remaining representatives and candidates, construct an undirected
union graph: two vertices have an edge whenever either judge pass places them
in the same cluster. Use its full connected components, including transitive
paths.

- A component with no existing representative receives one new stable key.
- A component with exactly one existing representative reuses that key.
- A component with more than one existing representative is rejected.

The union graph coarsens both partitions. Consequently, vertices in distinct
admitted components were separated by both passes. Judge disagreement can
merge support or reject a round, but cannot split a possibly equivalent pair
and cannot manufacture novelty reward.

The persistent bank schema is
`math_strategy_canonicalizer_two_pass_union_v4`. Calibration and training
import the same implementation.

## Frozen data, schedule, and gates

E47F reuses the immutable E47 source artifact: 50 level-five MATH12K problems,
64 Qwen2.5-0.5B policy samples per problem, the full validator decisions, and
four blinded positive controls per problem (anchor, exact duplicate, formatting
variant, and lexical paraphrase). It reuses the bounded online schedule of at
most 16 new candidates per round, the Qwen72 identity, judge prompt, two
permutation seeds, 4,000-character item cap, and fail-closed parsing.

The blinded audit packet samples up to 25 judge-same and 25 judge-different
policy-solution pairs. Different-key pairs are enumerated across every pair of
admitted keys before deterministic sampling, rather than limiting each
problem to its first two keys. Audit provenance and labels are stored in a
separate sidecar.

The calibration gate is unchanged:

- exact-duplicate false-new count equals zero;
- all injected false-new rate is at most 5%;
- lexical-paraphrase false-new rate is at most 10%;
- at least 80% of validator-positive policy solutions receive a key;
- audited same-pair false-new rate is at most 5%;
- audited different-pair false-merge rate is at most 20%; and
- there is no structural failure.

E49 may launch only from a complete E47F artifact with every frozen check
passing.
