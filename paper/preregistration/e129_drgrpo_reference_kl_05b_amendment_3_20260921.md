# E129 amendment 3 — one coefficient that straddles the predicted knee — September 21, 2026

Amends `e129_drgrpo_reference_kl_05b_20260918.md` and extends amendment 2.
Written before the arm it registers is submitted. Amendment 2's arms are
partly read: 68 of E129's 75 cells and 18 of E129-X's 50 are terminal.

## What amendment 2 predicted, and how it stands

Prediction (4), the correctness knee, holds and is sharper than registered.
It is not one knee but a different knee in each domain, and its position is
predicted by a quantity measured on the frozen model before training. With
`logit P* = logit mu(C) + c_G/beta`, the coefficient at which stationary
correctness falls to one half is

    beta* = c_G / (-logit mu(C)).

Measured `mu(C)` on the frozen policy, the predicted `beta*`, and the observed
terminal pass@8:

| domain | mu(C) | beta* | pass@8 at .04 / .1 / .2 |
| --- | --- | --- | --- |
| PantryPlan | .290 | 1.05 | .82 / .82 / --- |
| Graph | .153 | 0.55 | .62 / .59 / .57 |
| MathIR | .038 | 0.29 | .42 / .25 / .22 |
| Countdown | .014 | 0.22 | .64 / .37 / .13 |

The rank order is exact: the two domains whose `beta*` lies above the tested
range lose little or no correctness across it, and the two whose `beta*` lies
inside it collapse. Prediction (5), the breadth ceiling, is contradicted at the
margin: Graph reaches `.496` against a frozen `.454` and MathIR `.109` against
`.061`, both at the top of the coefficient range and both on one or two seeds.

## What this arm tests, registered before it runs

A single coefficient, `beta = .30`, at the same five domains and seeds 43--47,
25 cells, identical in every other respect. It is chosen because it falls
between the two pairs of `beta*` values rather than beyond all of them, so it
separates the prediction from a monotone trend.

**Prediction (6), the knee is where `beta*` says.** At `beta = .30`, terminal
correctness collapses in Countdown and MathIR, whose `beta*` is below `.30`,
and is substantially preserved in Graph and PantryPlan, whose `beta*` is above
it. A uniform collapse across all four, or none, disconfirms the predictor and
leaves the knee an unexplained property of each domain.

**Prediction (7), the ceiling at the widest window.** PantryPlan carries the
largest `beta*` and the highest frozen breadth, so it is where an anchor has
the most room to approach its bound. Terminal \pmd{} there does not materially
exceed the frozen `.913`. The two marginal excesses already observed sit within
the cross-pool measurement gap of `.023` estimated in amendment 1; an excess
at PantryPlan would not, and would contradict
Proposition "The retained conditional is the reference's" directly.

Disconfirmation of either is reported as a result about the flow's
applicability, not set aside.

## Status

Like amendment 2's arms, this one is chosen after reading results and is not
part of the frozen design; reports must keep the registered three coefficients
distinct from the four added since. What makes it admissible is that
predictions (6) and (7) are quantitative, derived from an identity written
before any cell of this cohort existed, and recorded here before submission.
