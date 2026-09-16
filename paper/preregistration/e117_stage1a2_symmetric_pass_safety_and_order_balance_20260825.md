# E117 Stage-1-A2 symmetric pass safety and order-balance amendment

Frozen: 2026-08-25T10:35:27-04:00 while E117-R1 remains at 0/12
terminal, zero realized optimizer updates, and before any Stage-1 seed, split,
job, ledger, or outcome exists. This amends only the analysis-only Stage-1
development contract. It does not authorize a launch or change E112-R1's
already-frozen analysis.

## Trigger

Stage-1-A1 correctly restored the primitive endpoint vector
`(pass@8, raw distinct correct modes@8)`, but retained two asymmetric pass
safeguards from the original screen:

1. `F-P` breadth was judged with pass safety only for `F-C`. A replay pass
   gain could therefore hide a material semantic pass loss relative to P and
   allow the semantic component to advance even though it moved one primitive
   coordinate in the wrong direction beyond the registered tolerance.
2. Breadth had to improve both terminally and over normalized AUC, while pass
   safety was checked only terminally. A transient pass collapse followed by
   terminal recovery could therefore pass a sample-efficiency screen.

Both defects were identified from the frozen algebra and synthetic regression
fixtures. No E117 or Stage-1 outcome was inspected.

## Corrected component gate

The breadth, uncertainty, seed-consistency, sentinel-scope, and confirmation
rules remain unchanged. For each component and sentinel:

- Apply the existing mean pass-effect floor of -0.03 and every-seed floor of
  -0.10 to both terminal and normalized-AUC pass@8.
- Apply those floors to the component's own contrast (`P-C` or `F-P`).
- Also apply them to the candidate numerator versus C (`P-C` or `F-C`).

For `P-C`, the two safety contrasts are algebraically identical and are kept
explicit in the result for schema uniformity. For `F-P`, both are required:
the semantic increment may not free-ride on a replay accuracy gain, and the
full F candidate must remain safe relative to the compute-only control.

The executable schema advances to
`e117_stage1_paired_vector_statistics_v3`. Every elementary safety check and
the terminal/AUC component-relative and C-relative pass summaries are emitted.

## Balanced execution order for a future launcher

The three fresh seed ranks within every sentinel must use the three cyclic arm
orders exactly once:

1. `C -> P -> F`
2. `P -> F -> C`
3. `F -> C -> P`

Actual start order—not merely submission order—must be enforced by dependency
chains or a block wrapper. Within a sentinel/seed block, C/P/F must use the
same physical node class, resource envelope, frozen source, data, and runtime
plumbing. The execution ledger must record intended and actual ordinal
positions. A launch whose realized order is not balanced fails its audit and
cannot enter the registered Stage-1 table.

This Latin-square nuisance control prevents arm identity from being perfectly
confounded with first/second/third execution position without adding another
statistical endpoint or tuning dimension.

## Status boundary

Stage 1 remains a three-seed development screen. A separate execution protocol
must still freeze the fresh seeds, split digest, at least 16 common evaluation
draws, full checkpoint grid, node blocks, source snapshot, complete exports,
and exact launch/audit machinery before submission. Confirmation still
requires a separately frozen fresh split and at least five fresh paired
training seeds with no tuning on Stage-1 outcomes.
