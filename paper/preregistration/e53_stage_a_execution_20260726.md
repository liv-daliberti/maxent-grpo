# E53 conditional Stage-A execution

**Status: FROZEN DURING SENTINEL PASS 0, BEFORE ANY TERMINAL OUTCOME — 2026-07-26**

This document operationalizes the conditional Stage A in
`e53_verified_exemplar_replay_05b.md`. It does not change the mechanism,
coefficients, capacity, information boundary, or decision thresholds.

Stage A may start only from a machine-readable E53 sentinel audit whose status
is `pass` and whose `authorizes_stage_a` field is true. The approval must bind
the exact sentinel identity, parent protocol, sentinel launcher, sentinel
auditor, source snapshot, and execution snapshot. The Stage-A launcher reuses
those immutable source and execution snapshots; it does not copy the mutable
repository source.

The cohort contains graph coloring, Countdown easy3, and executable Python
factors; matched Dr.GRPO, direct inverse conditional-content entropy, and
direct inverse entropy plus verified-exemplar replay; and fresh seeds 43, 44,
and 45. All 27 jobs begin from the same frozen Qwen2.5-0.5B-Instruct revision
and run exactly 50 prompt-pool passes with the E53 sentinel configuration.
No sentinel checkpoint, controller state, bank, exemplar, optimizer state, or
evaluation result is reused.

Every individual replay seed must pass E53's runtime, checkpoint, safety,
terminal-quality, final-eight distinctness, multiplicity, and 75%-self-
retention gates against its matched same-seed Dr.GRPO control. For each domain,
the three-seed mean trajectory must independently pass the same behavioral
gate. Any run, seed, domain, or seed-mean failure makes Stage A fail.

The launch is atomic: all jobs are submitted held, exact arm/seed membership
and immutable configuration are checked, then the cohort is released. Partial
submission is cancelled fail-closed. The sentinel approval cannot authorize
coefficient selection, a new support target, or any use of gold valid-answer
multiplicity.
