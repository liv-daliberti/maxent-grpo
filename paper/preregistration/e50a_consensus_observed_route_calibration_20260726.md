# E50A — consensus observed-route calibration on frozen hard-MATH samples

**Status: PREREGISTERED BEFORE E50A ROUTE INFERENCE — 2026-07-26**

## Motivation and frozen input

E49Y generated the intended natural Qwen2.5-0.5B corpus, but its global
partition interface failed closed on 23/74 eligible rows because structured
output omitted at least one response.  Another 19/74 rows were rejected
because the two partitions disagreed anywhere in the full response set,
even when a pair of response groups might still have been mutually stable.
E50A repairs those two false-negative mechanisms without weakening the
criterion used to declare two routes different.

E50A reuses the exact E49Y natural-response corpus and performs no new
natural sampling.  Its frozen private corpus SHA-256 is
`ab6c5c081919caaf358c65ff63e394b0fc6587dc98fb69318a4c77b5aa76401f`;
the public sampling-summary SHA-256 is
`e6203f7dec2a4b751d6d47e090c3c9884974f9dc3535e2dcfc61fef7edac7fc3`.
The E49Y terminal decision SHA-256 is
`4f87a4d232ba1bf70cdbeb1e98a6624295b7499f4c6d4803016db93dc1ab305b`.
No individual response or problem-specific partition was inspected to
choose this repair.

## Exact two-pass assignment

For each of the same 74 eligible rows, two temperature-zero, independently
order-permuted calls to the frozen E49T Qwen2.5-72B endpoint partition at
most 16 complete answer-validated responses.  The response schema contains
one required assignment property for every supplied response ID.  Local
validation requires every ID exactly once and requires a nonempty strategy
description for every used cluster label.

Let response \(i\) receive labels \((c_i^{(1)},c_i^{(2)})\) in the two
partitions.  A consensus group contains exactly the responses sharing the
same ordered label pair.  A candidate route needs at least two members.
Two consensus groups can form S1/S2 only when their first-pass labels differ
and their second-pass labels differ.  Thus both independent partitions must
agree that every within-route pair is the same and every cross-route pair is
different.  Disagreement involving responses outside the retained groups
cannot create a route and is ignored.

Candidate group pairs are ranked by decreasing smaller-group size, then
decreasing combined size, then lexicographic member IDs.  The first pair is
used.  This rule is fixed for every row.

## Semantic and executable gates

The retained observed exemplars are converted to a finite S1/S2 action menu.
Proposed routes remain forbidden.  The same two independent frozen-72B
auditors must literally execute both combos, bind them to the cited natural
exemplars, independently derive the answer, find no hidden decisive step or
answer leakage, and unanimously classify the routes as genuinely distinct.

Double-audited candidates are ordered by decreasing smaller consensus-group
size, decreasing combined size, then source index.  Qwen2.5-0.5B receives
each route as a forced declaration and produces eight fresh samples per
route.  A route success requires both exact-answer validation and assignment
to that exact route by the frozen E49T canonicalizer.  Wrong-route solutions
receive no credit.  A problem is bidirectionally executable only if both
routes have at least one success.

Advance only with at least ten bidirectionally executable problems.  Select
the deterministic first ten passing problems.  This stage performs no policy
update.  Route inference uses two workers and a 1,200-second request timeout
to avoid overloading the frozen endpoint.
