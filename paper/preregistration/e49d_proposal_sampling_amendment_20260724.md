# E49D proposal-sampling amendment — 2026-07-24

**Status: FROZEN BEFORE ANY E49D TRAINING LAUNCH**

On a simple officer-counting problem, both answer-blind proposal calls entered
the same whitespace loop inside guided JSON and reached the token limit before
closing the object. The strict parser rejected both responses; no menu was
materialized.

Direct menu and route-ideation proposals now use temperature 0.2 with their
already frozen per-attempt seeds. Their response caps are 2048 and 1536 tokens
respectively. The two mathematical certification passes remain
temperature-zero, 4096-token, answer-bound, and otherwise unchanged.

This makes the direct and rescue searches reproducibly nonidentical and avoids
greedy whitespace degeneration. Proposals still have no authority: every
retained strategy and pair must pass the same double v4 certification.
Maximum-clique selection, singleton fallback, policy data, runtime checks,
reward, E46 controller, cohorts, schedule, and gates are unchanged. All
earlier v4 certifications remain valid.
