# E49G hardened 72B strategy-pair veto calibration

**Status: FROZEN BEFORE ANY E49G 72B REQUEST — 2026-07-24**

## Motivation

The identity-bound E49F/V5 blinded audit found 22 false-new claims among 26
retained pairs (84.6%). Only four pairs were both sound and genuinely
distinct. No policy training was launched.

E49G calibrates a high-precision veto; it does not restore, relabel, or train
on any rejected pair. The frozen inputs are the 29-pair blinded packet and
the manual labels written before its private key was inspected.

## Four-way conservative decision

Each pair receives four temperature-zero Qwen2.5-72B assessments:

1. equivalence reduction, displayed A then B, seed `493101`;
2. equivalence reduction, displayed B then A, seed `493102`;
3. theorem-signature falsification, displayed A then B, seed `493111`; and
4. theorem-signature falsification, displayed B then A, seed `493112`.

Every assessment must strip action IDs, declared kernel labels, step counts,
and prose before comparing the minimal sufficient mathematics. It must try
to construct a routine reduction between routes and identify one
route-exclusive necessary fact for each. A route with an invalid,
insufficient, or hidden step cannot support novelty.

The local decision is distinct only by four-way unanimity: both routes sound,
different decisive cores, no routine reduction, not merely different labels,
two different nonempty exclusive facts, and relation `distinct` in every
assessment. Any malformed, ambiguous, equivalent, or disagreeing result is
equivalent for bank admission.

## Calibration report

After all requests are durably complete, the frozen manual labels are joined
by pair ID. The report records false-new count and rate, true-distinct recall,
precision, confusion counts, the three hidden equivalent-control outcomes,
and every disagreement. Passing requires:

- zero false-new predictions across all manually non-distinct or unsound
  pairs;
- all three hidden equivalent controls rejected as new;
- at least three of the four manually distinct pairs recovered; and
- no incomplete request.

This is an in-domain calibration of the offline bank veto. A separately
curated bank remains subject to the same blinded manual audit before policy
training; E49G alone cannot authorize training.
