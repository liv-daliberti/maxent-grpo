# E117 Stage-1-A3 paired evaluation-identity amendment

Frozen: 2026-08-25T11:19:15-04:00 while E117-R1 remains at 0/12
terminal, zero realized optimizer updates, and before any Stage-1 seed, split,
job, ledger, sampled response, or outcome exists. This amends only the
analysis-only Stage-1 development contract. It does not authorize a launch or
change E112-R1's already-frozen analysis.

## Trigger

The Stage-1 analyzer required the complete registered seed/arm/checkpoint/draw
grid and common draw indices, but a common integer draw label does not prove
that C, P, and F evaluated the same ordered prompts or used the same request
seeds. A silent prompt-order, option-order, split, or request-seed drift could
therefore masquerade as a paired component effect.

This defect was identified from the frozen table schema and synthetic fixtures.
No E117 or Stage-1 outcome was inspected.

## Response-free identity contract

Every future registered Stage-1 row must carry two lowercase SHA-256 digests.
The future table builder must compute them from the same canonical,
response-free projections frozen by E112-R1 final-analysis A3:

- `prompt_surface_sha256`: ordered projections of `answer_keys`,
  `answer_mode_count`, `option_ids`, `prompt`, `prompt_index`, and `reference`;
- `request_surface_sha256`: ordered projections of `option_ids`,
  `prompt_index`, and `request_seeds_by_option`.

Canonical JSON uses sorted object keys, compact separators, UTF-8, and original
prompt order. Metrics, responses, rewards, and all endpoint values are excluded
from both digests.

## Fail-closed invariants

Before computing an effect, the Stage-1 analyzer now requires:

1. a syntactically valid lowercase SHA-256 digest in both identity fields for
   every registered row;
2. one exact prompt-surface digest within a sentinel across all three training
   seeds, all C/P/F arms, every checkpoint, and every evaluation draw;
3. one exact request-surface digest within a sentinel and evaluation draw
   across all three training seeds, all C/P/F arms, and every checkpoint;
4. the previously frozen complete registered grid and exact step-zero endpoint
   equality within each paired C/P/F seed-draw block.

A missing field, malformed digest, prompt drift, request drift, duplicate row,
or incomplete grid aborts analysis. The executable result schema advances to
`e117_stage1_paired_vector_statistics_v4` and records the identity contract.

## Boundary

This closes paired evaluation identity only. It does not prove that the C/P/F
training intervention, resource envelope, source snapshot, or actual execution
order matched the registered design; the future Stage-1 launcher and audit must
still prove those properties. Stage 1 remains a three-seed development screen,
and confirmation still requires a separately frozen split with at least five
fresh paired training seeds. No endpoint, threshold, safety rule, sentinel
scope, or advancement decision changes here.
