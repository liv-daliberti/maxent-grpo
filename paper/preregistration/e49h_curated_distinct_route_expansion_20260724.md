# E49H curated distinct-route expansion after blinded false-new failure

**Status: FROZEN BEFORE ANY E49H 72B REQUEST — 2026-07-24**

## Motivation and non-retroactivity

The identity-bound E49F audit completed all 100 toy certifications but found
22 false-new claims among 26 automatically retained pairs.  Only two train
and two evaluation rows retained genuinely distinct support, so no policy
training was authorized.  E49H does not relabel or restore any rejected
claim.

E49H tests a separately written, finite cohort of problem-specific route
pairs.  The cohort is intentionally larger than the support requirement so
that conservative rejection does not need to be weakened.

## Frozen cohort and route integrity

The exact cohort and action traces are
`ops/math_strategy_calibration/e49h_curated_distinct_routes_toy.json`.
Every strategy must:

1. be a closed sequence of declared actions;
2. be independently executed by both the literal executor and adversarial
   checker used by E49E;
3. derive the frozen reference answer under the audited answer normalizer;
4. use no undeclared decisive operation and no other strategy's result; and
5. pass the calibrated E49G four-way conservative pair veto.

A candidate pair is locally eligible only when all four soundness audits
(two per route) pass and all four order/role-swapped E49G assessments vote
distinct.  Malformed, incomplete, ambiguous, disagreeing, or unsound output
is rejection.

The E49G calibration report must itself pass before E49H may issue a request:
zero false-new on its blinded calibration cohort, all three hidden
equivalent controls rejected, and at least three of four manually distinct
pairs recovered.

## Manual gate and support

Passing the automated gate is necessary but insufficient.  A sealed,
order-randomized manual packet will be written before its private mapping or
labels are opened.  It will include hidden equivalent/paraphrase controls.
Training remains blocked until every retained pair is manually audited and:

- manual false-new is exactly zero;
- every hidden equivalent control is rejected;
- every retained route is sound and self-contained;
- at least 10 train and 10 evaluation rows have genuinely distinct support
  after combining E49H survivors with the four E49F survivors;
- every other row has at least one validated singleton route; and
- all 100 prompts remain within the frozen model context budget.

No threshold, label, route, or cohort may be changed after E49H requests
begin.  Any correction is a new named amendment and fresh evidence tree.

## Downstream experiment

Only a passing audited bank can launch the matched three-epoch toy pair:

- current E46 normalized canonical-bank Haarnoja control, with no alternative
  uncertainty controller; and
- ordinary matched Dr.GRPO.

Seed, model, data order, rollout count, optimizer, epoch count, and evaluation
schedule must match.  Scaling to the exact OAT MATH-500 split remains blocked
until this toy pair completes and its entropy/support/reward mechanism is
compared with Countdown and graph coloring.
