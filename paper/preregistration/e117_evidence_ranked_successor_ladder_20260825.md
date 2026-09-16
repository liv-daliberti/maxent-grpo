# E117 evidence-ranked successor ladder

Frozen: 2026-08-25T13:19:15-04:00 while all twelve E117-R1 jobs were
pending with zero realized optimizer updates and before any E117 endpoint or
mechanism outcome existed.

Status: design-investigation memo, not launch authorization and not an
amendment to the E117 v10 development or confirmation contract. It ranks the
smallest later intervention by the mechanism that fails. Exact jobs, seeds,
requests, thresholds, and analysis for any later arm require a separate
pre-outcome freeze. PointMaze remains excluded.

## What the completed experiments actually say

E102 combined new verified proposal admission, four initial replay-priority
visits at 4x raw weight, replay mass, and retention-safe whole-bank balance. Its
five-seed mean raw-distinct effect versus matched E78 replay was -0.025 in
Graph, +0.934 in Countdown, +1.503 in Python, +0.093 in MathIR, and +0.611 in
Pantry. The bundle was valuable in three contexts, weak in one, and negative in
the high-admission Graph boundary. It did not identify which component earned
the gains.

E103 changed only the proposal scheduler. It issued 7,977 extra fallback
groups and obtained 86 fallback admissions, about 1.08 admissions per 100 extra
groups. Its five-domain macro E103-minus-E102 effect was approximately +0.0003
pass@8 and -0.0305 raw distinct@8. Domain raw-distinct means were -0.013 Graph,
-0.049 Countdown, -0.044 Python, +0.020 MathIR, and -0.066 Pantry. Additional
search activated and discovered modes but did not improve the endpoints.

E108 then measured the admission-to-retention funnel. In the passive arm,
teacher-forced score retention was 100% for every followed Graph and Pantry
admission, while neutral-rollout conversion was 8/14 (57%) and 38/51 (75%).
The adaptive arm added 26 Graph and 194 Pantry priority visits; conversion was
8/10 (80%) and 34/50 (68%), respectively. This one-seed mechanism gate cannot
support an efficacy comparison, but it shows that score collapse was not the
observed bottleneck and that feedback-driven priority can create substantial,
mixed actuation.

E117-P is intentionally smaller than E102: verified likelihood per rollout,
uniform replay, zero initial priority visits, no retention-safe whole-bank
balance, and no adaptive priority. E117 therefore tests the minimum proposal /
replay component. A null P-C result would reject that minimum component in the
registered contexts; it would not falsify the bundled E102 mechanism.

## Smallest-next-intervention rule

1. If the E117 identity audit fails, repair execution only. Do not interpret an
   endpoint and do not change an algorithmic knob.
2. If proposal admission or F pressure is not exercised, repair or extend the
   mechanism preflight. Nonactivation is not an efficacy null.
3. If P-C advances and confirms, retain uniform proposal replay. Do not add a
   controller merely because one exists.
4. If P admits modes but neutral-rollout conversion is weak, the next candidate
   is exactly one fixed **bridge** increment: P plus four initial priority visits
   at 4x raw replay weight, renormalized to the same total replay mass. Compare
   `B-P`; keep eta zero, one proposal attempt, and adaptive refresh off. This
   tests the simplest useful part of E102 without its feedback controller or
   bank-balance term.
5. If admission and neutral conversion are already strong but P-C does not
   improve raw breadth, a bridge is unlikely to address the failure. The next
   candidate is instead one retention-safe whole-bank balance increment `S-P`,
   with no semantic PPO, no priority, and no extra proposal attempts. This
   tests distributional allocation rather than discovery volume.
6. Do not combine B and S until one has value on its own. Do not revive the
   E103 attempt sweep or E108 adaptive refresh unless a new mechanism result
   identifies the corresponding failure.
7. If F-P advances and confirms, add the already-required semantic-without-
   proposal arm before direct attribution. If F pressure activates but F-P
   does not advance, retire semantic PPO at eta 0.10; do not respond with an eta
   sweep or RMS controller.

## Why this is the simpler design

The branching variable is the broken link in one observable funnel:

`proposal -> admission -> fixed replay bridge -> neutral conversion -> breadth`.

Each later experiment changes one edge and has one matched denominator. Search
volume, adaptive feedback, safe balance, and semantic pressure are never
introduced together. A trustworthy null removes a component; it does not
trigger a larger bundle.

The E112 private prefix helped identify sentinel contexts but cannot supply a
gate for this ladder. E117 v10 development and sealed confirmation remain the
only efficacy-selection path currently frozen.
