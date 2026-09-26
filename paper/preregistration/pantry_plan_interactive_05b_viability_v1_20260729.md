# PantryPlan finite-action 0.5B viability v1

Frozen before model sampling on 2026-07-29.

## Purpose

Test whether Qwen2.5-0.5B-Instruct has nonzero PantryPlan endpoint capability
when formatting is removed by a public-information finite-action interface.
This is a prospective development gate, not a confirmatory MaxEnt comparison
and not a paper result.

The failed one-shot PantryPlan v2 and prompt-repaired v3 probes remain failed.
This protocol does not replace or reinterpret them.

## Frozen inputs

- Model: local immutable Qwen2.5-0.5B-Instruct snapshot
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Data: `var/data/pantry_plan_modebench_v2/dev`, split `multi_answer`.
- Prompt count: 64.
- Sampling: 64 rollouts per prompt, temperature 1.0, top-p 1.0.
- Prefix decision: first 16 rollouts per prompt.
- Seed: 76101.
- Evaluation prompts are not loaded.

## Interface

At each decision, the model emits exactly one single-token label from a masked
menu of at most eight options.

- Ingredient phase: unused non-forbidden ingredient IDs, plus `STOP` after the
  public minimum ingredient count is met.
- Quantity phase: the selected ingredient's public
  `min_if_used_g:step_g:available_g` quantities.
- Observation: original problem, partial allocation, and exact running totals
  computed from public attributes.
- Endpoint reward: existing `validate_pantry_plan` only at `STOP`.

The action mask contains no certified support, solver candidate, evaluation
answer, or endpoint-feasibility signal. All rollouts terminate after at most
`2 * max_ingredients + 1` choices.

## Frozen decision

Pass only if both hold:

- at least 32 of 64 prompts have one verified completion in the first 16
  rollouts; and
- at least 16 of 64 prompts have at least two distinct verified semantic modes
  across all 64 rollouts.

On pass, freeze the receipt and design a shared warm-start/matched-training
smoke. On fail, proceed prospectively to a train-only supervised warm start;
do not repair this probe, alter development examples, or relax its thresholds.
