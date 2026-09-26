# E49S strict-trace terminal result

Both matched seed-45 arms completed 150 updates (three prompt epochs), but
neither learned. Greedy accuracy, mean@8, pass@8, coverage@8, and distinct@8
were byte-for-byte constant at every evaluation:

| Metric | Step 0 | Step 150 |
|---|---:|---:|
| greedy | 0.38 | 0.38 |
| mean@8 | 0.14 | 0.14 |
| pass@8 | 0.50 | 0.50 |
| coverage@8 | 0.54 | 0.54 |
| distinct@8 | 0.54 | 0.54 |

The cause is identified, not inferred from a flat curve. At the three
post-epoch summaries, the ordinary answer validator found positive rows, but
the exact XML action-trace parser accepted zero in both arms. The shared gated
task reward was therefore 0 throughout. The E46 treatment accumulated zero
canonical support observations and took zero Haarnoja dual observations or
optimizer steps; alpha remained 0.10.

Thus E49S v1 is a format-contract failure and supplies no evidence for or
against the mathematical canonical-MaxEnt mechanism. E49T retains the same
answer/route execution gate but replaces XML as the sole admission path with
unanimous, finite-menu-bound 72B inference calibrated for false routes,
answer-only traps, and declared-combo mismatches.

Authoritative curve:
`var/artifacts/e49s_deterministic_mathir_toy_05b_3ep_v1_scaling_curve.json`.
