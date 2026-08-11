# E90: bank-normalized verified replay, Qwen2.5-0.5B

Registered 2026-08-10, before any E90 cell was submitted.

## Why

Verified replay is uniform over the prompt's bank. The dose `alpha` is split
across the banked modes, so each banked mode receives `alpha / n`, where `n` is
the number of verified modes the bank currently holds. A prompt that has
discovered *more* modes therefore protects each one *less*.

That is the opposite of what the retention argument asks for. Appendix A's
result is a statement about each mode: any *positive recurrent* dose on a mode
creates a log barrier against its extinction, and the numerical size of that
dose controls finite-time strength. Under a fixed coefficient the per-mode dose
is not fixed --- it decays exactly as a prompt accumulates the modes the method
exists to retain.

## What the fixed dose actually realizes

Measured over the 25 fixed-dose replay cells of E78 (`alpha = .10`), reading
bank occupancy from `canonical_replay_actuator_modes`, which agrees with the
count recovered from the applied score-gradient L2 in 3069 of 3069 updates:

| domain | mean n | median | max | share of updates at n = 1 |
| --- | --- | --- | --- | --- |
| MathIR | 1.17 | 1 | 3 | 82.7% |
| Python factors | 1.19 | 1 | 3 | 81.2% |
| Countdown | 2.28 | 2 | 5 | 27.1% |
| Graph coloring | 3.98 | 4 | 14 | 9.9% |
| PantryPlan | 6.74 | 6 | 16 | 4.2% |
| pooled | 3.08 | 2 | 16 | 40.9% |

Per-mode pressure therefore spans `.10` (n = 1) to `.00625` (n = 16), a 16-fold
range at an identical nominal coefficient. This is the replay-side counterpart
of Appendix B.3's finding for the semantic coefficient.

A ratio-targeting controller of the kind E88/E89 use was considered and
rejected before implementation. The natural denominator, task-advantage RMS, is
exactly `sqrt(p(1-p))` for binary-reward group-centred Dr.GRPO (an identity that
held in 1356 of 1356 nonzero updates), and it is *zero* on 4.6% to 97.5% of
updates depending on the cell --- Python factors seeds 45/46/47 have mixed
groups in only 2.5%, 5.5% and 2.7% of updates. Holding replay at a fixed
fraction of task pressure would therefore cut replay hardest exactly where
Dr.GRPO supplies no gradient, which is where the retention argument says a
positive dose matters most. The design would be reachable and pointed the wrong
way. It is not registered here and will not be run.

## The arm

`alpha_t = c * n_t` with `c = .0325`, applied at the same site that reads the
fixed coefficient, so per-mode pressure is `c` by construction.

`c` is registered as `.10 / E[n]` with `E[n] = 3.08` the pooled mean occupancy
above. At the mean occupancy the two arms deliver the same dose, so this
redistributes a matched total replay mass across bank sizes rather than adding
dose. It is not a strength manipulation.

No new ceiling is introduced. `n` is already bounded by
`online_canonical_replay_capacity = 16`, so `alpha_t <= .52` follows from a
parameter that is already registered. The arm adds exactly one constant, `c`,
and no new bound.

Everything else --- data, seeds 43-47, five domains, schedule, decoding,
placement, bank capacity, replay objective, and code --- is inherited from the
E78 replay arm, which is the comparator. The two differ by one applied
derivative; a test pins the variant blocks to differ only in the dose rule.

## Outcomes and estimands

Primary: paired seed difference `E90 - E78replay` at pass 8 for `distinct@8`
and `pass@8`, cell by cell.

Show all five paired seed differences and their mean and range. Do not pool
domains into one effect and do not select a best checkpoint.

## Mechanism gate

Read before any outcome.

1. Realized per-mode pressure equal to `c` within 1% on at least 99% of applied
   updates, in every domain. This is a plumbing check, not a hypothesis: the
   rule makes it exact, so any violation means the dose is not being applied
   where it is being logged.
2. Pooled mean realized `alpha` within 25% of `.10`. This is the falsifiable
   one: `c` was calibrated on E78's occupancy, so this fails if bank growth
   under the new dose does not resemble bank growth under the fixed one.
3. `alpha` at the capacity-implied ceiling (`n = 16`) on no more than 20% of
   applied updates, in every domain.
4. Mean response length within 25% of the comparator and no-EOS below `.05`.

Failing 2 means the arm is not the matched redistribution it is registered as.
The registered response is to report it as a dose-shifted arm and read the
outcome against that, **not** to re-derive `c`.

## Registered interpretation

1. Gate met and E90 beats E78 replay: per-mode pressure is the better dosing
   rule, and the fixed coefficient's decay with bank growth was a real defect.
2. Gate met and E90 is weaker: the fixed coefficient's implicit annealing ---
   less pressure per mode as a prompt accumulates modes --- is doing useful
   work, and should be described as a feature rather than an accident.
3. Gate met and the difference is within seed range: the dose schedule does not
   matter over this range, which bounds how much of the method's effect is
   attributable to dosing at all.
4. Gate failed at 2: report as above, no re-derivation.

Under every branch the fixed `.10` dose stops being un-ablated, which is the
limitation the manuscript currently concedes.
