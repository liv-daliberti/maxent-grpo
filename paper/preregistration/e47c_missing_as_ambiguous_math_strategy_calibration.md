# E47C-CAL: bounded MATH strategy calibration with omission rejection

**Status: FROZEN BEFORE LAUNCH — 2026-07-24**

E47B confirmed that even a bounded partition can occasionally omit one opaque
candidate ID. E47B's strict structural rule therefore fails. E47C is a
prospective safety refinement, not a relabeling of E47 or E47B.

E47C retains E47B's exact frozen data, bounded online schedule, Qwen72
checkpoint, prompt, two temperature-zero permutation seeds, stable
representative bank, and gates. The only change is:

- an expected ID omitted from otherwise parseable JSON is treated exactly as
  if the judge had placed it in `ambiguous_ids`.

Thus an omission receives no canonical key, no bank admission, and no entropy
or novelty reward. Unexpected IDs, duplicate assignments, malformed JSON,
invalid schema, transport errors, and parse errors still fail the call and the
calibration. If a stored representative is omitted, every candidate in that
round is rejected because existing strategy identity cannot be established.

The frozen gate remains:

- exact-duplicate false-new zero;
- overall injected false-new at most 5%;
- lexical false-new at most 10%;
- at least 80% canonicalization coverage among validator-positive policy
  samples;
- blinded same-pair false-new at most 5%;
- blinded different-pair false-merge at most 20%; and
- all 50 bounded problems complete without an unrecoverable structural error.

For the false-new endpoint, an omitted/ambiguous item is a false negative and
reduces reported canonicalization coverage; it is not counted as a new
strategy because it receives no key and no reward. A false-new event requires
two different admitted keys inside the known same-strategy injection family.
The family's comparison key is the anchor when admitted, otherwise the first
admitted control in the frozen order exact, formatting, lexical.

This rule is intentionally conservative: availability improves because one
bad row no longer crashes training, but uncertainty can only remove reward,
never create a new strategy.
