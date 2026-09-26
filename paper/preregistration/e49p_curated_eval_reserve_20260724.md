# E49P curated evaluation reserve

**Status: FROZEN BEFORE ANY E49P 72B REQUEST — 2026-07-24**

Early E49N execution evidence showed that its first two pairs each failed at
least one of four audits, so E49N could contribute at most four pairs even
before MathIR and manual review. E49P is a new ten-row reserve; no earlier
cache or threshold is reused.

The exact contract is
`ops/math_strategy_calibration/e49p_curated_distinct_routes_eval_reserve.json`.
Its SHA-256 is
`bf2fd6dae7346330cd80fcc0b34d43f89e41aa6a18d6125e15b74e2e841b1869`.
Routes use a single declared action wherever the mathematical algorithm can
be complete in one action. This removes accidental action repetition or
omission without weakening soundness: each action must still expose and
execute the entire decisive calculation, produce the validated answer, and
use no undeclared operation.

Every route receives both frozen E49E execution audits. Every surviving pair
receives all four E49J restricted-MathIR parses and is retained only by
four-way unanimity. The required E49J calibration report SHA-256 is
`3400db77d6da7572395fe4d40b03ec59b792e2682c18c49ce861d32206bab3db`.

Survivors enter the same sealed randomized manual packet with hidden
equivalent controls. Manual false-new must be exactly zero, retained routes
must be sound and self-contained, combined support must reach at least ten
rows per split, no row may have zero support, and all prompts must fit the
context limit before training.
