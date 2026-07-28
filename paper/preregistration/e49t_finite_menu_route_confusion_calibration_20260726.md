# E49T — finite-menu route-confusion calibration

**Status: FROZEN BEFORE 72B SCORING — 2026-07-26**

E49T may map an answer-positive, unstructured derivation only to one strategy
already present in that problem's E49S finite menu. It can never emit an
open-set strategy. The remaining novelty risk is therefore a *false
difference*: two executions of the same route could be assigned to different
existing menu entries, or an execution of one route could be assigned to its
other entry.

This calibration uses the twelve E49R pairs that a blinded manual audit found
to contain two sound, self-contained, genuinely distinct routes. Every route
has two independently generated, answer-validated execution traces. Their
strategy IDs and action labels are removed from the text shown to the E49T
auditor, yielding 48 positive natural derivations: 12 problems × 2 routes × 2
renderings. One answer-only negative control is injected per problem. The
60-item cohort and its private labels are hashed before any scoring call.

Two independently permuted, temperature-zero Qwen2.5-72B audits must agree on
`valid` and on the same frozen strategy ID. Anything else is rejected. The
checkpoint revision, endpoint record, cohort, labels, source, and protocol are
recorded in the result.

The calibration advances only when all of the following hold:

1. no returned key lies outside its frozen finite menu;
2. no accepted positive is assigned to the wrong route;
3. no two accepted renderings of one route receive different keys;
4. no answer-only negative is accepted;
5. at least 75% of the 48 positive derivations are accepted;
6. both routes are correctly admitted for at least 10 of 12 problems; and
7. there are no judge schema/format failures.

These gates are assessed before launching the matched E49T toy training pair.
