# E49D proposal-schema amendment — 2026-07-24

**Status: FROZEN BEFORE ANY E49D TRAINING LAUNCH**

The first eight-worker preprocessing job produced a fail-closed record for the
consecutive-page-number problem: both the direct and rescue proposal violated
the deterministic action-combo grammar before either could be audited. No
menu for that row was materialized and no training had launched.

The proposal JSON schema now constrains action IDs to `A1..A8`, strategy IDs
to `S1..S3`, and action sequences to unique IDs. The existing deterministic
parser still requires consecutive action/strategy IDs, only references to
defined actions, distinct strategy combos, and all prior length/shape rules.

This amendment prevents a generator transport-format failure. It does not
change any mathematical proposal instruction, answer isolation, auditor,
seed, certification rule, maximum-clique selection, singleton behavior,
policy prompt, runtime execution validation, reward, controller, cohort,
training schedule, or outcome gate. Earlier v4 certifications remain valid
because the audit-only preflight recomputes them from their durable proposal
and auditor evidence rather than trusting the proposal schema.
