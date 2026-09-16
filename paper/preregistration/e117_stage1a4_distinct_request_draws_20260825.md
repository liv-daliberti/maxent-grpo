# E117 Stage-1-A4 distinct request-draw amendment

Frozen: 2026-08-25T11:27:57-04:00 while E117-R1 remains at 0/12
terminal, zero realized optimizer updates, and before any Stage-1 seed, split,
job, ledger, sampled response, or outcome exists. This amends only the
analysis-only Stage-1 development contract. It does not authorize a launch or
change E112-R1's already-frozen analysis.

## Trigger

Stage-1-A3 proves that every C/P/F row bearing a given evaluation-draw label
uses the same response-free request surface. It did not prove that different
draw labels use different request surfaces. A broken future exporter could
therefore repeat one request-seed surface under 16 labels, creating
pseudoreplication and an invalidly small evaluation Monte Carlo standard error.

The defect was identified from the frozen identity algebra and a synthetic
fixture. No E117 or Stage-1 outcome was inspected.

## Corrected draw contract

For each sentinel, all registered `request_surface_sha256` values must satisfy
both conditions:

1. exact equality across training seeds, C/P/F arms, and checkpoints within an
   evaluation draw, as frozen by A3; and
2. exact uniqueness across all registered evaluation draws.

The request digest remains the response-free canonical projection of
`option_ids`, `prompt_index`, and `request_seeds_by_option`. Thus uniqueness is
checked on the realized complete request-seed surface, not merely on a draw
number or top-level seed label.

Any repeated request surface aborts analysis before an endpoint, effect, or
standard error is computed. The executable result schema advances to
`e117_stage1_paired_vector_statistics_v5` and emits
`request_surfaces_distinct_across_draws=true`.

## Boundary

This establishes distinct common-random-number draws; it does not claim the
draws are statistically independent beyond the registered sampler contract.
It changes no endpoint, effect threshold, pass-safety rule, sentinel scope, or
advancement decision. Stage 1 remains development-only, and confirmation still
requires a separately frozen split with at least five fresh paired training
seeds.
