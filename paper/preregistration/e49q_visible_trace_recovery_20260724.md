# E49Q visible-trace recovery under calibrated MathIR

**Status: FROZEN BEFORE ANY E49Q 72B REQUEST — 2026-07-24**

## Motivation

The strict curated cohorts required both execution judges to pass each route.
This exposed many useful false negatives: a route could have one exact,
answer-validated execution while the other judge returned a JSON pointer,
repeated an action, or made a calculation error. Rejecting such pairs before
the independently calibrated canonicalizer and blinded manual audit discards
support without reducing the measured false-new risk.

E49Q does not alter or relabel any prior result. It is a separately frozen
recovery analysis over the exact E49H, E49K, E49N, E49P, and E49L evidence
trees. Their frozen identity SHA-256 values are, respectively:

- `f58266dba1242c37d1a3308d3a02a180c0cc67fcf9a3d526c443ff06aee1577f`;
- `43eec09a20b6bd7d75653bd814e9f5221bd425003b7bbb053d4008bf6e75927d`;
- `6f17399340bcd6e0bcacb777ca345bcb9f3c60f68f60f7875d7fb63deba20404`;
- `50baffe2417831d48003a3ddf2b111318b4d27032b22d3d92b5091ed7abc821d`;
- `8fb606d7ef2ebdaa55239eefe8c69047fc81649f2d3291ff47ed63c794434f2c`.

## Frozen recovery rule

A previously rejected candidate is eligible exactly when:

1. all four execution responses are complete and schema-valid;
2. each route has at least one execution that exactly uses its declared
   action combo and passes the independent answer validator; and
3. the candidate did not already pass its original strict gate.

All four traces—including a judge's failing trace—are shown to every E49J
parser. Nothing is hidden or selectively repaired. The pair is recovered
only if all four role/order-swapped restricted-MathIR comparisons vote
distinct. A malformed, incomplete, unsound, empty, or disagreeing parse is
rejection.

The frozen preflight contains 53 source candidates, 14 eligible candidate
contracts, 11 unique evaluation rows, 3 unique training rows, and exactly 56
new MathIR requests.

E49Q uses the unchanged E49J report with SHA-256
`3400db77d6da7572395fe4d40b03ec59b792e2682c18c49ce861d32206bab3db`.
That calibration had 0/25 false-new, 4/4 true-distinct recall, and 116/116
complete requests. The novelty threshold remains four-way unanimity.

## Manual and training gate

Recovery is necessary but insufficient. Every recovered pair enters the same
sealed randomized manual packet as strict survivors, with hidden equivalent
controls. Manual false-new must be exactly zero and both routes must be
manually sound and self-contained. Support is counted by unique row, not by
candidate contract. Training remains blocked until at least ten unique rows
per split survive, all 100 rows retain nonzero support, and all prompts fit
the frozen context limit.
