# E47J-CAL: component-incidence MATH strategy calibration

**Status: FROZEN BEFORE LAUNCH — 2026-07-24**

E47I validated the bounded membership-only serialization and completed every
previously problematic structural case. It nevertheless made its frozen 80%
policy-coverage gate mathematically unreachable after 39/50 problems:
187/254 completed policy-positive solutions received keys, and even perfect
admission on all 21 remaining positives could reach only 208/275 = 75.64%.
The failure was localized to whole-round rejection when one judge pass
coarsened two previously distinct representatives. E47I is preserved as an
early-stopped coverage failure.

E47J replaces that over-broad rejection with the maximal conservative
component-incidence rule. After removing candidates ambiguous in either pass,
form connected components of the candidate-only graph whose edges are
same-cluster judgments in either pass. For each component `C`, let `R(C)` be
the set of stored representatives co-clustered with any member of `C` in
either pass.

- If `|R(C)| = 1`, reuse that existing key.
- If `|R(C)| > 1`, reject only `C`.
- If `|R(C)| = 0` and every stored representative was observed in both
  passes, create one new key for `C`.
- If `|R(C)| = 0` but any stored representative was ambiguous or omitted,
  reject `C`; an unknown relationship may not earn novelty.

A representative that is ambiguous does not force rejection of a component
that already has exactly one known existing key, because reusing a key cannot
create support or novelty. Likewise, a judge pass that merges existing
representatives does not invalidate an unrelated candidate component.

This retains the false-new proof. The union graph makes distinct candidate
components separated in both passes. A new component is additionally
separated from every fully observed stored representative in both passes.
Every uncertain bridge is either folded into one existing key or rejected;
none can create a rewarded boundary.

All E47I data, validator decisions, prompts, finite JSON schema, judge
identity, decoding, bounded online schedule, blinded audit construction, and
numerical gates are unchanged. The persistent schema is
`math_strategy_canonicalizer_component_incidence_v8`. Calibration and training
import the same implementation.

E49 may launch only from a complete
`var/artifacts/e47j_bounded_math_strategy_calibration_v1` artifact whose
unchanged frozen gates all pass.
