# E68 zero-step restoration to pvl-lowprio

Recorded 2026-07-27 after the first E68 zero-step placement amendment and
before any affected job materialized a run directory or optimizer metric.

The interactive `lowprio` probes predicted starts around 18:47--19:10, but
the actual released batch jobs received start estimates from July 28 03:40
through July 29 03:40. The previously recorded actual `pvl-lowprio` estimates
were earlier: July 28 01:00 through July 29 01:00.

This amendment restores only the nine still-pending jobs
`30130469--30130477` to their original partition and node pools. Graph returns
to A6000 `node103,node104`; Countdown/Python retain RTX 3090
`node020,node021,node022,node024,node026`. Account `mltheory` and every
scientific/runtime setting remain unchanged. MathIR `30130478--30130480` is
untouched.

Every affected job must first be held, pending, and unmaterialized. The
document and script are SHA-256-bound before release.
