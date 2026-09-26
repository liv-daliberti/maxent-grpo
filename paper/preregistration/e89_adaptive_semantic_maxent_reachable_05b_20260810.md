# E89: adaptive semantic MaxEnt at a globally reachable target, Qwen2.5-0.5B

**Frozen before submission on 2026-08-10.**

## Why E88 is superseded on this parameter, and only this one

E88 targeted a realized semantic/task advantage RMS ratio of `rho = .05`. Its
registered mechanism gate failed in two of five domains:

| domain | seed | final `eta` | realized ratio | updates pinned at a bound |
|---|---|---|---|---|
| PantryPlan | 43 / 44 | .216 / .186 | .047 / .050 | 3% / 5% |
| Countdown | 46 / 47 | .233 / .324 | .049 / .048 | 10% / 7% |
| Graph coloring | 43 | .305 | .049 | 14% |
| Python factors | 44 | **.400** | **.017** | **98%** |
| MathIR | 46 / 47 | **.400** / **.400** | **.036** / **.028** | **98%** / **97%** |

The failure is arithmetic, not instability. Those domains realize very little
semantic pressure per unit coefficient, so reaching `.05` would require `eta`
far above the ceiling.

**The ceiling is not the adjustable parameter.** `|A_sem| <= eta` against a unit
correct/incorrect reward gap means `eta = .40` already narrows a verified
response's advantage margin to `.60`. Raising it further starts eroding the
guarantee that semantic MaxEnt cannot invert the correctness ordering, which is
the property that makes the term safe to add at all. E88's protocol therefore
required re-deriving the target rather than widening the bound, and that is
what E89 does.

E88's cells that reached `rho` are not discarded and are not re-run. E89
supersedes E88 on the target ratio only.

## The re-derived target

Across E88's saturated cells the lowest sustained realized ratio at the ceiling
was `.0170` (Python factors seed 44); MathIR seeds 47 and 46 sustained `.0276`
and `.0361`. A globally reachable target must sit below the binding cell with
margin, so

    rho = .015     (88% of the binding constraint)

with every other controller setting inherited from E88 unchanged: `eta_0 = .10`,
`eta in [.02, .40]`, gain `.5`, EMA decay `.98`, per-update cap `1.1`, warmup 64,
and the same refusal rule that freezes the coefficient when the eligible
fraction is below `.05`, the task RMS below `1e-3`, or the semantic RMS below
`1e-6`.

`rho` is derived from mechanism telemetry only. No E88 evaluation outcome, and
no `distinct@8` or `pass@8` value, entered this derivation.

## What this costs, stated up front

Setting a single reachable target means the common dose is fixed by the weakest
domain. At `rho = .015`, Countdown is dosed *lower* than it was at the fixed
`eta = .10`, where it realized `3.7%` and produced the largest fixed-arm
`distinct@8` gain in E81 (`+.115`).

E89 therefore tests **uniformity, not strength**. The registered expectation is
that it may well be weaker than E81 in the domains where the fixed dose was
already generous, and that this is the correct price of a dose that means the
same thing everywhere. Reporting E89 as an improvement on the strength of a
Countdown result alone would be a misreading of its own design.

## Design

Model, domains, data, seeds, schedule, decoding, placement, replay dose,
eligibility gate, predictor, and comparators are inherited from E88 and E81
unchanged. 5 domains x 1 arm x 5 seeds = **25 runs**. Eight passes, 3,072
updates, evaluations and resumable checkpoints every 192 updates.

The launcher asserts that E89's objective differs from E88's in exactly one
environment variable, the target ratio, and fails closed otherwise.

## Outcomes and estimands

Primary: paired seed difference `E89 - E81` at pass 8 for `distinct@8` and
`pass@8`, cell by cell, which isolates the adapted dose against the fixed one.
Secondary: `E89 - E88` on the cells where E88 has a terminal endpoint, which
isolates the target ratio; and `E89 - replay` against E78.

Show all five paired seed differences and their mean and range.
Do not pool domains into one effect and
do not select a best checkpoint.

## Mechanism gate

The same gate E88 failed, applied again before any outcome is read:

1. Realized ratio within a factor of two of `rho` by pass 2 in at least four of
   five domains. E89's whole premise is that `rho` is now reachable, so a
   weaker pass than E88's threshold would not test it.
2. `eta` pinned at a bound on no more than 20% of applied updates, in every
   domain.
3. Per-domain frozen fraction reported; above 80% the domain is reported as
   *not adapted* and read as the fixed arm.
4. Mean response length within 25% of the fixed arm and no-EOS below `.05`.
   E88 tripped this on one cell (Python factors seed 44, length `+364%`,
   no-EOS `.054`) while pinned at the ceiling; at a reachable target no cell
   should sit at the ceiling at all.

Failing 1 or 2 again means the RMS-targeting design, not its parameter, is
wrong. The registered response is to stop and report that, **not** to derive a
third target.

## Registered interpretation

1. E89 reaches `rho` everywhere and beats E81: equalized pressure is the better
   dosing rule, and the fixed coefficient was an incidental choice.
2. E89 reaches `rho` everywhere and is weaker than E81: the fixed coefficient
   was accidentally well matched, and uniformity costs more than it buys. This
   is a real possibility given that `rho = .015` under-doses four of five
   domains relative to `eta = .10`, and it will be reported plainly.
3. E89 again fails to reach `rho`: RMS targeting is not viable under a bound
   that preserves the correctness ordering, and the adaptive program ends with
   that as its result.

## Integrity

All 25 cells are submitted held from one hash-bound snapshot, identical to
E88's patch set, and released only after their scheduler environments pass an
exact audit. E88's 9 cells with controller telemetry are retained as the
evidence for this re-derivation and are neither deleted nor re-run; its 16
unstarted cells were cancelled before this cohort was registered.
