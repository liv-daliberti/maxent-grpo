# PantryPlan prospective three-way split repair

**Status: FROZEN BEFORE V2 DATA GENERATION OR MODEL SAMPLING — 2026-07-29**

## Reason for the repair

PantryPlan v1 passed its deterministic source and support audit, but it
materialized only 96 train and 32 evaluation prompts. The already-frozen
base-model viability gate requires 64 development prompts and forbids use of
the evaluation split. V1 therefore cannot support a valid viability sample.

V1 remains an immutable admission artifact. This prospective v2 changes only
split construction and scale; it retains the reviewed USDA ingredient table,
prompt format, exact verifier, support canonicalizer, quantity grid, family
definitions, and support-count bounds.

## V2 construction

Use data seed `74002` and build balanced, disjoint splits:

- train: 96 prompts per family, 384 total;
- development: 16 prompts per family, 64 total; and
- evaluation: 32 prompts per family, 128 total.

Development uses seed `84002` and excludes every train fingerprint.
Evaluation uses seed `94002` and excludes every train and development
fingerprint. V1 rows are not copied, promoted, or used to select v2 rows.

For every row, exhaustively enumerate feasible allocations, require 8–64
distinct ingredient-support keys, and replay the certification hash. The full
source, manual review, split rows, fingerprints, and support counts must pass
the same audit logic as v1. Any cross-split fingerprint overlap rejects v2.

## Information boundary and next gate

Generation and deterministic audit use no language model. The v2 evaluation
split remains sealed during viability.

Only a passing v2 audit authorizes the previously frozen Qwen2.5-0.5B
development-only viability sample:

- 64 temperature-one completions per development prompt;
- the first 16 are the intended group-size prefix;
- at least 32 of 64 prompts have a verified formulation in the first 16; and
- at least 16 of 64 prompts expose two distinct verified ingredient-support
  keys in the full 64.

Passing viability authorizes only the matched Gate-A training smoke, not any
of the 80 confirmatory jobs.
