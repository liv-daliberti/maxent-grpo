# Proof reference re-pin, 2026-09-15

Blocks: 20 before, 20 after. Identical blocks: 18.

Two formal blocks changed wording. Both edits were already in the working tree;
they were not made by the re-pin. Recorded here so the change is reviewable
rather than absorbed silently by moving the hash.

## [Fresh-group starvation] \label{lem:replay-gradient-availability
- insert: `` -> `for `
- replace: ` and has probability` -> `, an event whose probability is`
- insert: `` -> ` at every correctness level`

##  The all-correct and all-incorrect events are disjoint and have probab
- replace: `; the union bound gives` -> `, so the union bound supplies each of`
- replace: `bounds` -> `stated bounds,`
- insert: `` -> `, and the minimum of the pair is therefore also a bound`

Reference sha256: `1c4c8d816c8fb5a57aeb1c6d627c678b423b253dd2d2cb90eeac8b2d48ddd158`
