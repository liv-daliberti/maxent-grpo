# PantryPlan Qwen2.5-0.5B viability v3: prompt-format repair

**Status: FROZEN BEFORE THE FIRST V3 MODEL COMPLETION — 2026-07-29**

## Reason for the repair

The frozen v2 gate (Slurm job `30184767`) completed with zero verified
completions across all 64 development prompts. Inspection was limited to
parser/instruction behavior: sampled responses did not preserve the required
`\boxed{...}` answer envelope and frequently ignored the already-present
`ingredient_id=grams` syntax. No evaluation prompt, exact support catalogue,
certified allocation, or training outcome was inspected.

V2 explicitly permits one prospective prompt-format repair. V3 changes only
the assistant response prefill: model context ends with the literal prefix
`\boxed{`, and verification reconstructs the full assistant response as that
fixed prefix plus the generated continuation. It does not change prompts,
generated tokens, verifier, support identity, model, or thresholds.

## Immutable sampling and decision contract

- exact 64-row `dev/multi_answer` split from
  `var/data/pantry_plan_modebench_v2`;
- pinned Qwen2.5-0.5B-Instruct snapshot
  `7ae557604adf67be50417f59c2c2f167def9a775`;
- temperature `1.0`, top-p `1.0`, fresh seed `76003`;
- 64 completions per prompt, first 16 as the group-size prefix;
- response budget 192 tokens and model context 2048 tokens;
- at least 32 prompts with a verifier-positive prefix completion; and
- at least 16 prompts with two distinct verified ingredient-support keys in
  the full sample.

Failure stops PantryPlan before training. Passing authorizes only a matched
Gate-A training smoke, not its five-seed confirmatory cells.
