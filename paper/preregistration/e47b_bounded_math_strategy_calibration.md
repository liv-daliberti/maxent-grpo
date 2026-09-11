# E47B-CAL: bounded online MATH strategy calibration

**Status: FROZEN BEFORE LAUNCH — 2026-07-24**

## Reason for the prospective successor

Frozen E47 attempted one all-at-once partition per problem. Ninety of 100
temperature-zero partitions completed, but ten partitions from large
validator-positive pools repeatedly omitted IDs. The largest request contained
55 solutions. E47 therefore fails structurally and is not overwritten.

E47B tests the actual online operation required by E49: at most one policy
group's validator-positive candidates at a time, with stable representatives
from earlier rounds carried forward. It reuses E47's already frozen problems,
0.5B samples, full-validator decisions, blinded injections, Qwen72 checkpoint,
semantic-equivalence prompt, temperature, and permutation seeds. No 0.5B
sample is regenerated.

## Bounded stream

For each problem, candidates are processed in deterministic batches of at most
16, matching E46's rollout group size. The anchor and exact duplicate occur in
round one; the formatting and lexical controls occur in round two, testing
both contemporaneous co-clustering and later matching to a stored
representative. Validator-positive policy samples fill remaining positions and
subsequent batches.

Every round uses two independently permuted temperature-zero partitions with
seeds `470721` and `470722`. Only pairwise relations that agree in both passes
may create or reuse a key. Ambiguity, disagreement, invalid JSON, missing IDs,
merging of previously distinct representatives, transport errors, and parse
errors all fail closed. A structural call failure fails the calibration run.
The exact online `MathStrategyCanonicalizer` source used by E49 performs the
partitioning and checkpoints stable prompt-local representatives.

## Gate

E47B advances only if:

- exact-duplicate false-new is zero;
- overall injected false-new is at most 5%;
- lexical-paraphrase false-new is at most 10%;
- at least 80% of validator-positive policy samples receive a conservative
  consensus key;
- a blinded manual audit has same-pair false-new at most 5% and
  different-pair false-merge at most 20%; and
- every bounded problem completes without structural failure.

The audit packet contains up to 25 policy pairs the canonicalizer calls the
same and 25 it calls different. Labels are `same`, `different`, or
`uncertain`; uncertain pairs are reported and excluded from binary rates.

Passing establishes only a sampled online strategy key suitable for E46's
validator-bound bank. It does not make Qwen72 a correctness validator or prove
logical equivalence.
