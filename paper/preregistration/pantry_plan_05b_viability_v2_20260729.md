# PantryPlan Qwen2.5-0.5B development-only viability gate

**Status: FROZEN BEFORE THE FIRST PANTRYPLAN V2 MODEL COMPLETION — 2026-07-29**

## Inputs

This gate uses only the 64-row `dev/multi_answer` split admitted by
`pantry-plan-modebench-admission-audit-v2`. The 384 training rows, 128
evaluation rows, exhaustive support catalogues, and certified allocations do
not enter prompts or sampling decisions.

The checkpoint is
`Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`.
For each development prompt, draw 64 completions at temperature 1.0, top-p
1.0, sampling seed `76002`, response budget 192, and context budget 2048.
The first 16 completions are the intended training-group prefix.

## Decision

The gate passes only if all 64 prompts complete without an infrastructure or
identity failure and:

1. at least 32 prompts have one exact-verifier-positive formulation in the
   first 16 samples; and
2. at least 16 prompts expose two distinct verified ingredient-support keys
   in the full 64 samples.

One prospective prompt-format repair is allowed only for parser/instruction
failures and must rerun the entire development sample. It cannot change
constraints, support identity, prompt membership, or these thresholds.

Passing authorizes only PantryPlan’s matched Gate-A training smoke. This
base-model sample is not part of the final 80-job estimate.
