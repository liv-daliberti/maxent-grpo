# E112-R1 final-analysis A4 distinct request-draw closure

Frozen: 2026-08-25T11:37:53-04:00 while E112-R1 is 49/75 terminal,
E109 is 13/15 terminal, E117-R1 is 0/12 terminal, and the complete official
E112 result does not exist. No response, reward, metric, endpoint, or private
interim value was inspected to make this amendment.

This adds one fail-closed validation to the A3 response-free evaluation
identity. It changes no treatment, comparator, checkpoint, draw, endpoint,
effect, interval, or decision rule.

## Trigger

A1 already requires four unique draw indices and exact top-level sampled seeds
following `seed_base_by_domain + draw_index`. A3 requires one exact realized
request digest within each draw across all checkpoints and between treatment
and comparator. Neither check explicitly rejects the same realized
`request_seeds_by_option` surface appearing under two different draw indices.
Such repetition would reduce the effective evaluation Monte Carlo diversity.

The gap was identified by static schema review and a synthetic fixture. The
existing distinct top-level seeds make this unlikely, but the final builder
should prove the realized request surfaces rather than infer them.

## Fail-closed invariant

For every treatment and comparator trajectory, after A1 and A3 validation:

1. retain the A3 SHA-256 digest of the ordered response-free request projection
   (`option_ids`, `prompt_index`, `request_seeds_by_option`) for each draw;
2. require each digest to remain exact across all 17 registered checkpoints;
3. require all four draw digests to be pairwise distinct; and
4. retain exact treatment/comparator equality of the complete identity object.

If two draw indices reuse one realized request-seed surface, final
materialization aborts before any official result is written. The result's
metric contract records
`require_distinct_request_surfaces_across_draws=true`.

## Boundary

This proves four distinct realized request surfaces, not more than four draws
and not independence beyond the registered sampler. Evaluation Monte Carlo is
still averaged within each training seed and must not be counted as additional
training seeds. E112 remains a bundled historical-comparator contrast and
non-confirmatory after private interim inspection.
