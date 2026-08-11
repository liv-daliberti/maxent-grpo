# E92: adaptive semantic MaxEnt on verified replay, Qwen2.5-3B

Registered 2026-08-11, before any E92 cell was submitted.

## Why

E89 tests whether an RMS-targeted coefficient is a better dosing rule than a
fixed one on Qwen2.5-0.5B. A dosing rule that only works on one model family is
not a dosing rule. E91 is the second-family replication and E92, here, the third.

## The arm

Identical to E89 in every registered parameter: controller, `eta <= .40` bound,
gain `.5`, EMA decay `.98`, warmup 64, minimum eligible fraction `.05`, refusal
rule, and target ratio `rho = .015`. Only the family and its cells change.

Reusing one `rho` across families is deliberate. A per-family target would make
"uniform semantic pressure" three tuned treatments instead of one, and the
cross-family claim would not survive it.

## Reachability, checked before submission

E88 failed by registering a target above what the safety bound can deliver. The
same check is therefore a precondition here, not a diagnostic.

The applied semantic advantage is exactly proportional to `eta`, so the ceiling
`eta = .40` reaches about four times the ratio realized at the fixed `eta = .10`.
Measured over E87, and over E85 for PantryPlan whose E87 cells were superseded:

| domain | realized at eta = .10 | reachable at the ceiling |
| --- | --- | --- |
| MathIR | .0121 | .0484 |
| Graph coloring | .0187 | .0747 |
| Countdown | .0274 | .1097 |
| PantryPlan | .0307 | .1227 |
| Python factors | .2658 | (excluded, see below) |

The binding domain is MathIR at `.0484`. `rho = .015` sits at 31% of it, so the
target is reachable in every domain with margin.

Python factors is excluded from the binding calculation. At 3B its
task-advantage RMS is near zero for most updates -- Dr.GRPO has almost nothing
to learn from there -- so its ratio measures a vanishing denominator, not
headroom. Its controller behaviour is reported under gate 3 rather than used to
set the target. The launcher refuses to submit
if this stops holding.

## Outcomes and estimands

Primary: paired seed difference `E92 - E87` at pass 8 for `distinct@8` and
`pass@8`, cell by cell, which isolates the coefficient control against the
fixed one on this family. Secondary: `E92 - E80-R1replay`.

This cohort is one seed (70), matching E87. Show all five domain differences; do not average them into one effect. Do not pool
domains and do not select a best checkpoint.

## Mechanism gate

The E89 gate, applied unchanged and read before any outcome:

1. Realized ratio within a factor of two of `rho` by pass 2 in at least four of
   five domains.
2. `eta` pinned at a bound on no more than 20% of applied updates, in every
   domain.
3. Per-domain frozen fraction reported; above 80% the domain is reported as
   *not adapted* and read as the fixed arm.
4. Mean response length within 25% of the fixed arm and no-EOS below `.05`.

Failing 1 or 2 on this family, when E89 passed on its own, means the rule is
family-specific. The registered response is to report that, **not** to derive a
per-family target.

## Registered interpretation

1. Gate met and E92 beats E87: equalized pressure is the better dosing rule and
   it transfers across families.
2. Gate met and E92 is weaker: the fixed coefficient is doing useful work, and
   E89's result on 0.5B does not generalize.
3. Gate met and the difference is within seed range: dosing does not matter over
   this range on this family.
4. Gate failed: report as family-specific, no re-derivation.
