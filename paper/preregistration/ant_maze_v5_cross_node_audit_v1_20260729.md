# AntMaze v5 cross-node robustness audit v1

Frozen before execution on 2026-07-29.

## Purpose

Resolve a reproducibility discrepancy without changing maps or interpreting a
post-outcome rerun as a pass. Prospective admission job 30184769 remains
failed on `ant_admission_train_00 upper`. A later exact diagnostic using its
hash-bound source passed that fixture 5/5 times on the login host at final
distance 0.430, and passed the lower fixture at 0.449.

This audit asks whether controller v5 and the frozen route slate are robust on
the node class intended for subsequent GPU work. It is model-free and cannot
admit an LM experiment by itself.

## Frozen inputs

- The same 12 maps, in the same order, generated from nonempty subsets of
  `(1,5), (2,5), (4,5), (5,5)`.
- The same reset cells, goal cells, reset seeds 76300--76311, route gates,
  75-step action repeat, success threshold 0.5, and 32-action maximum.
- Upper program: `N` x5, `E` x6, `S` x5, `SE` x2, `NE`.
- Lower program: `S` x6, `E` x7, `N` x10, `NE`.
- The immutable v5 controller receipt
  `324d3301b8e21b4dbbfcc2e6b9a87aba479bd2da5aa3a040cff24adebaf8828e`.
- Three distinct MLTHEORY A5000 nodes, three deterministic repetitions of
  every map-route pair per node: 216 executions total.

No map substitution, route repair, seed repair, or success-threshold change is
permitted.

## Frozen decision

Each execution must:

1. pass the existing executable route validator;
2. produce the expected upper or lower gate identity; and
3. finish at goal distance at most 0.45, a 0.05 robustness margin inside the
   environment's 0.5 success threshold.

The aggregate passes only if all 216 executions pass and receipts name three
distinct nodes. Raw execution trajectories and first validation errors are
retained.

On pass, v5 may enter a separately preregistered closed-loop LM capability
design. On any failure, v5 is stopped and a feedback-based v6 controller must be
developed on training maps before a new prospective audit. Job 30184769 remains
failed in either case.
