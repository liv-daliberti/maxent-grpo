# E47D-CAL: representative-consensus MATH strategy calibration

**Status: FROZEN BEFORE LAUNCH — 2026-07-24**

E47C safely maps omitted IDs to ambiguity, but its initial rule requires every
candidate's relation to every other candidate to be identical across judge
permutations—even when both passes independently match that candidate to the
same already accepted representative. This can discard a sound existing-key
decision because of an unrelated pair.

E47D prospectively narrows agreement to the relation needed for each decision:

- an existing key is reused only when both non-ambiguous passes match the
  candidate to exactly the same single stored representative;
- if either pass matches an existing representative but the two matches are
  not identical and unique, the candidate is rejected;
- a candidate is eligible to create a new key only when neither pass matches
  any existing representative, neither pass co-clusters it with a candidate
  already matched to an existing representative, and both passes agree on all
  pairwise relations among the remaining new candidates;
- representative omission/ambiguity/merging still rejects the whole round;
  candidate omission remains ambiguity and receives no key or reward.

Thus no existing-key or new-key admission depends on a single pass, while
irrelevant candidate-candidate instability cannot erase a unanimous
representative match.

All data, schedules, prompts, checkpoint identity, decoding, audit construction,
and gate thresholds are otherwise exactly E47C. E47D writes a new artifact and
does not overwrite E47, E47B, or E47C.
