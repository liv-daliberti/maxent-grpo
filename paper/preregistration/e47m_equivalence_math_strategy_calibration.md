# E47M-CAL: proof-gated MATH strategy-equivalence calibration

**Status: FROZEN BEFORE LAUNCH — 2026-07-24**

## Reason for the prospective successor

E47L cleanly rejected both known invalid modular derivations and completed
all 50 problems without structural failure. It admitted 185/275 (67.27%)
answer-matched policy responses after two proof-integrity screens and had
zero false-new events on 150 exact/format/lexical controls.

E47L nevertheless failed two gates. First, its inherited 80% coverage target
treated deliberate invalid-proof rejection as missing canonicalization.
Second, the blinded audit found one false-new boundary: Qwen72 split
“circumference times number of revolutions” from the algebraically identical
“linear speed times elapsed seconds.” The audited same-pair false-new rate was
1/9. Only one level-five problem exhibited two accepted policy keys, and that
boundary was the false split. E47L is preserved as a failure and cannot train.

## Prospective changes

E47M retains E47L's final-answer validator, two independent derivation
integrity calls, two independent partition calls, component-incidence rule,
model, seeds, temperature, bounded JSON, and fail-closed behavior.

The equivalence prompt adds one general clarification: algebraically
equivalent parameterizations of the same computation are one route. It names
per-event amount times event count versus per-time rate times elapsed time,
regrouping factors, and equivalent closed formulas as routine equivalences.
The existing distinction between a direct sign/bound/invariant proof and an
expanded case/root/discriminant proof remains explicit. The persistent schema
is `math_strategy_canonicalizer_two_stage_equivalence_v11`.

Before the 50-problem replay, a frozen live regression packet must pass:

1. one valid GCD anchor is admitted while the two E47K invalid modular
   witnesses are rejected;
2. revolution-count and speed-time versions of the same clock computation
   receive one key; and
3. a manually written, answer-validated absolute-value case proof receives a
   different key from the official direct-sign proof.

The constructed case proof and its different-route label are frozen in the
regression script before querying the judge.

## Coverage gate amendment

The old 80% denominator was appropriate when every final-answer-positive
response was presumed to be a valid solution. E47L demonstrated that this
premise is false: 102/475 streamed answer-matched rows failed the independent
proof screen. E47M therefore requires at least 50% of answer-matched policy
responses to receive a key, while retaining all precision gates. This threshold
is frozen before E47M is run. It is not evidence that rejected proofs are
valid; they continue to receive zero key and zero novelty.

## Frozen gate

E47M advances only if:

- all three live semantic regressions pass;
- exact-duplicate false-new is zero;
- overall injected false-new is at most 5%;
- lexical-paraphrase false-new is at most 10%;
- at least 50% of answer-matched policy responses receive a key;
- both known invalid policy witnesses receive no key;
- the blinded manual audit has same-pair false-new at most 5%;
- human-different false-merge is at most 20%, with the frozen valid
  different-proof regression supplying this check if no naturally occurring
  human-different policy pair survives the integrity screen; and
- every bounded problem completes without structural failure.

Passing licenses only a conservative sampled strategy key. A later hard-toy
stage must independently show support of at least two accepted strategies and
live Haarnoja behavior before full MATH training.
