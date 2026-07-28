# E49D audit-transport bound amendment — 2026-07-24

**Status: FROZEN BEFORE ANY E49D TRAINING LAUNCH**

Two algebra rows produced truncated JSON inside otherwise successful HTTP
responses when the 72B auditor emitted unusually long scratch derivations.
The strict parser rejected every such response, so neither row was
materialized.

The v4 response schema now bounds the derived answer to 128 characters,
shared-core and decisive-operation phrases to 256 characters, and concise
derivation/pair checks to 512 characters. The prompt also requires compact
evidence and forbids extended scratch work. These limits keep the full
three-strategy audit comfortably inside the already frozen 4096-token
response budget.

No evidence field, answer-match boolean, soundness/failure status, pair
relation, role, seed, or response identity is removed. Mathematical
certification, maximal-clique selection, singleton behavior, runtime
validation, reward, controller, cohorts, schedule, and gates are unchanged.
Earlier v4 certifications already satisfy the same semantic contract and
remain recomputable.
