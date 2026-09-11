# E50F4 — pair-local finite strategy-family calibration

**Status: PREREGISTERED 2026-07-26 after E50F3 failed and before E50C
teacher-route terminal output**

E50F3's finite ontology removed much of the open-ended relation ambiguity,
but batching 36 routes from unrelated problems caused 14 permutation-unstable
family assignments and one false-new error.  E50F3 remains failed.

E50F4 freezes the same finite ontology and boundary rules but presents only
the two routes for one problem per request.  This removes cross-problem label
interference; it does not change the family definitions, calibration cohort,
or thresholds.

Use the 18 E49R pairs with both routes manually sound: 12 distinct and six
equivalent.  Use Qwen2.5-72B at temperature zero with seeds 500791 and
500792; reverse route presentation on the second pass.  A pair is distinct
only when both passes assign stable, unequal, non-`other` canonical families.

The exact gate remains zero false-new, at most two false merges, all 36 route
assignments stable, and all responses terminal/well-formed.  As before,
family assignment occurs only after execution soundness and cannot itself
earn reward.
