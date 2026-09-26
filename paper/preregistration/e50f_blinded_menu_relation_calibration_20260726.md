# E50F — blinded finite-menu relation calibration

**Status: PREREGISTERED 2026-07-26, after E50A terminal failure and before
E49AB terminal output**

## Motivation

E50A exposed a false-merge failure in the *menu-level* relation audit that is
separate from the already-passed E47W/E49T response canonicalizer
calibrations.  For source 152, both independent audits found the
quadratic-formula route and the Vieta/root-identity route sound, complete,
answer-matched, non-leaking, concrete, and bound to observed exemplars.  Each
audit named their different decisive operations but nevertheless emitted
`relation=equivalent`.

No E50A route is accepted by this observation.  The relation prompt must first
pass a frozen blinded calibration.

## Frozen calibration cohort

Use all 26 opaque route pairs in
`e49r_combined_manual_audit_v1/manual_audit_packet.jsonl` and the labels frozen
before its private key was inspected in
`e49r_combined_manual_audit_v1/manual_audit_labels.json`.  There are exactly
16 manually labeled distinct pairs and 10 manually labeled equivalent pairs.
The classification target is the decisive mathematical strategy, independent
of the separate soundness decision.

Render only each problem, route plan, exact action combo, and declared action
operations.  Do not expose the manual labels, rationales, reference answer,
source identity, private key, or execution-audit verdicts to the model.

## Judge and decision

Use the frozen Qwen2.5-72B endpoint and two deterministic passes with seeds
500761 and 500762.  Randomize case order independently and swap route sides
deterministically.  The prompt uses the already-calibrated E47W boundary:

- same route means the same decisive identity, theorem, invariant,
  construction, counted object, substitution, or algorithm after routine
  algebra;
- different route means a genuinely different central mathematical
  operation or proof construction;
- a shared goal, answer, or final arithmetic step is no evidence of sameness;
- when the comparison itself names different central operations, it must
  return different unless it explains why they are only routine
  reparameterizations.

The calibration passes only with:

1. zero false-new decisions on all 10 equivalent pairs in both passes;
2. at most two false merges among the 16 distinct pairs in each pass;
3. unanimous relation across the two passes for every pair;
4. no missing, duplicated, malformed, or nonterminal judge response.

The full prompts, response IDs, content hashes, and assessments are retained.
Only a passing frozen E50F result may authorize a prospective re-audit.  It
does not retroactively change E50A, E49AB, E49AC, or E50C results.
