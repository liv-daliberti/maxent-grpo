# E52 sentinel runtime-validation repair

**Status: FROZEN BEFORE CORRECTED RELAUNCH — 2026-07-26**

The first E52 sentinel submission (`sentinel_v1`, Slurm jobs
`30124287--30124295`) exposed an execution-only validation conflict before any
optimizer update. An older exclusivity guard rejected every
`maxent_inverse_canonical` process because it prohibited all combinations of
direct MaxEnt and an online canonical bank.

All nine v1 jobs were cancelled. They are invalid engineering attempts, are
not resumed or pooled, and cannot authorize any later stage.

The corrected validation rule admits exactly the already frozen E52 hybrid:

- direct inverse adaptation is active;
- the direct objective is `conditional_token_mean`;
- canonical Haarnoja/dual adaptation is inactive; and
- canonical policy-entropy adaptation is inactive.

All other direct-MaxEnt plus canonical-bank combinations remain rejected.
The objective, controller equation, coefficients, seed, data, evaluation
schedule, 50-pass horizon, hardware placement, gates, and information
boundary remain unchanged from the parent E52 protocol. The corrected
sentinel uses fresh `sentinel_v2` run prefixes, source and execution hashes,
jobs, initialization, banks, controller states, optimizer states, and
checkpoints.
