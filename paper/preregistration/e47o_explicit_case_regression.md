# E47O-REG: explicit valid-case elimination regression

**Status: FROZEN BEFORE LAUNCH — 2026-07-24**

E47N passed the valid-anchor, invalid-proof, and routine-equivalence
regressions. It rejected the valid different-route case proof because its
brief check said positivity of \((97\pm\sqrt{129})/32\) had not been
explicitly verified. E47N is preserved as a preflight failure.

E47O leaves the executable canonicalizer and judge prompts unchanged. The
manually authored different-route regression proof now spells out
\(0<\sqrt{129}<97\), hence both candidate roots are positive and neither
satisfies the assumed \(x<0\) case. This is routine expansion of a correct
step, not a new route or a changed expected label.

E47O advances only if all four frozen live semantic checks pass.
