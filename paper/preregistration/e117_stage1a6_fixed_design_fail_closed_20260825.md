# E117 Stage-1-A6 fixed-design fail-closed amendment

Frozen: 2026-08-25T11:34:05-04:00 while E117-R1 remains at 0/12
terminal, zero realized optimizer updates, and before any Stage-1 seed, split,
job, ledger, sampled response, or outcome exists. This amends only the
analysis-only Stage-1 development contract. It does not authorize a launch or
change E112-R1's already-frozen analysis.

## Trigger

The prose contract freezes all four E117 sentinels and K=8, but the executable
analyzer accepted a caller-supplied sentinel subset and any integer K greater
than one. It also relied on Python key equality, under which Boolean row
identifiers can alias integer zero or one. A mistaken future builder could
therefore analyze a design different from the one described by the result.

These defects were identified by static schema review and synthetic fixtures.
No E117 or Stage-1 outcome was inspected.

## Corrected fixed design

Before reading any endpoint, the analyzer now requires:

1. the exact ordered four-sentinel registry:
   `qwen05b/countdown`, `qwen05b/graph_coloring`,
   `qwen05b/python_factors`, and `falcon1b/mathir`;
2. exactly K=8, with Boolean values rejected;
3. string sentinel and arm identifiers in every row; and
4. non-Boolean integer training-seed, checkpoint, and evaluation-draw
   identifiers in every row.

The prior exact three-seed, at-least-16-draw, checkpoint, C/P/F, completeness,
identity, endpoint-range, and step-zero checks remain. Any constant or type
drift aborts analysis. The executable schema advances to
`e117_stage1_paired_vector_statistics_v7`.

## Boundary

The actual fresh seed values, at-least-16 draw schedule, split digest, and
checkpoint grid do not exist yet and must be frozen by a separate future
execution protocol. This amendment locks only constants already fixed by the
analysis contract. It changes no endpoint, contrast, uncertainty calculation,
threshold, safety rule, or advancement scope.
