# E49D singleton support-recall amendment — 2026-07-24

**Status: FROZEN AFTER A PREPROCESSING-ONLY FAILURE AND BEFORE ANY E49D
TRAINING LAUNCH**

The first complete toy materialization produced 100/100 recomputable,
double-audited menus and passed prompt-length audit, but retained only 16
multi-route menus (8/50 train and 8/50 eval). The preregistered 20% gate
correctly blocked training; no matched toy job was submitted.

To distinguish truly single-route problems from proposal-recall failures,
every certified singleton now receives exactly one additional answer-blind
route-ideation rescue and no more. It uses rescue index 2 and deterministic
request seed 491803. The new proposal still cannot see the reference answer
or auditor derivations. Its routes must pass the unchanged answer-bound
`soundness_execution` and `equivalence_attack` auditors with seeds 491711 and
491712. If no double-certified pair is found, the original singleton remains.
Transport failures remain incomplete and are retried; they cannot count as a
completed recall attempt.

The same fixed direct-plus-two-rescue procedure applies to every row in the
conditional full 384-train/MATH-500 stage. This is not an until-passing loop:
if the toy or full cohort still has fewer than 20% honest multi-route menus,
its existing gate continues to block training. Prior records and the
16-multi preprocessing artifact remain preserved for audit.

Policy data, exact declarations, runtime execution validation, reward, E46
controller, matched arms, cohorts, three-epoch schedule, and learning gates
are unchanged.

## Immutable pre-recall archive

Before issuing any supplemental request, the complete 16-multi artifact,
append-only evidence bytes, generation summary, and terminal materialization
log were copied or moved to:

`var/artifacts/e49d_pre_recall_16multi_20260724`

Their frozen identities are:

```text
5c4885793f86b3e1af78287ae93fb366c89c42b7413d2d743cb0073c2c5ceffe  data tree
928d18a61b0c1146364a897cb3e5c14877e4207073656d1638e2bfdf7db1b023  menu_records.jsonl
40c684336c169c5fbce6217a33263c57c72e70142ec53e0a82e171d37505f5f6  generation_summary.json
7bde48b48cae0c8965c6c90456385c1b9c64ae2b3c7aabfb21b5e6a421437424  materialize-30073656.out
```
