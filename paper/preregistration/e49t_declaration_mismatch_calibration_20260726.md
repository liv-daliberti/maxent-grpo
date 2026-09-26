# E49T — declared-combo mismatch veto

**Status: FROZEN BEFORE 72B SCORING — 2026-07-26**

The main E49T route-confusion calibration checks whether natural mathematics
is assigned to the correct frozen menu route. This supplemental gate checks
the stronger execution contract requested for training: when a response
explicitly declares a strategy ID and action combo, its mathematics must
execute that exact combo.

For each of the 24 manually audited routes in the frozen E49T calibration,
one natural execution is copied twice. One copy receives the matching
strategy ID/combo declaration. The other receives the opposite route's
strategy ID/combo declaration while its mathematics is left byte-identical.
The 48 rows are deterministically shuffled within their twelve prompts and
hashed before scoring.

Two independently permuted, temperature-zero Qwen2.5-72B audits must agree.
Training advances only if no mismatched declaration is admitted, no admitted
matched declaration is assigned to the wrong route, no open-set key or schema
failure occurs, at least 75% of matched declarations are admitted, and both
correct routes are admitted for at least 10 of 12 prompts.
