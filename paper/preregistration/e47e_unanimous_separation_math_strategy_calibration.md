# E47E-CAL: unanimous-separation MATH strategy calibration

**Status: FROZEN BEFORE LAUNCH — 2026-07-24**

E47D requires unanimous equality for an existing representative and identical
pairwise partitions for new candidates. Its partial diagnostic remains
precision-safe but rejects many rows when the two passes disagree. E47E
implements the direct conservative novelty rule proposed for E49:

> A new strategy boundary exists only when both independent judge passes
> separate the solutions.

Operationally:

- candidate omissions are ambiguous and receive no key;
- if either pass matches a candidate to one existing representative and
  neither pass suggests another representative, the existing key is reused;
  this cannot create novelty;
- conflicting possible existing representatives reject the candidate;
- among candidates separated from every existing representative in both
  passes, an edge is placed when either pass calls a pair the same;
- connected components of this union graph receive one key each.

The union graph coarsens both partitions. Therefore two new components are
distinct only when every cross-component pair is separated by both passes.
Judge disagreement can merge strategies or remove entropy signal, but it
cannot split a possibly identical pair and cannot earn false novelty.
Blinded manual false-merge measurement guards against excessive coarsening.

All E47C data, bounded schedules, Qwen72 identity, prompt, decoding seeds,
omission handling, audit construction, and frozen gates are retained:
exact-duplicate false-new zero, overall false-new at most 5%, lexical
false-new at most 10%, validator-positive coverage at least 80%, manual
same-pair false-new at most 5%, manual different-pair false-merge at most 20%,
and no unrecoverable structural failure.

E47E is a new immutable artifact; earlier calibration outputs are preserved.
