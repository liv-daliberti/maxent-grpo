# PointMaze algorithm repair v3: split-balanced orientations

**Status:** FROZEN DURING THE V2 DEVELOPMENT PAIR, BEFORE ITS TERMINAL AUDIT,
V3 ROUTE EXECUTION, OR V3 MODEL SAMPLING  
**Date:** 2026-07-30  
**Role:** Disclosed secondary post-outcome engineering repair; excluded from
the original 80-cell estimator.

## Diagnosed failure mode

The v2 source assigned rotations 0 and 2 to all eight training rows, rotation
1 to all four development rows, and rotation 3 to all four evaluation rows.
During the immutable v2 pair, the training rows verified at roughly 85%, while
neutral evaluation on the held-out orientation was roughly 9%. Thus the
trainability ceiling and directional evaluation gap are a split/orientation
distribution mismatch. No MaxEnt coefficient is changed in v3.

## Frozen data repair

V3 uses the four medium 9x9 and four hard 11x11 source geometries already
defined before this repair.

- Train: eight rows, exactly two at each rotation 0/1/2/3.
- Development: four rows, exactly one at each rotation.
- Evaluation: four rows, exactly one at each rotation.
- Every training family appears once. Development and evaluation use disjoint
  four-family subsets, each already represented in training at a different
  rotation.
- No exact executable task fingerprint (maze, reset, goal, and route gates)
  may overlap between splits.
- New map IDs, seeds, fingerprints, and split hashes are fixed in source.

The row counts remain API-compatible with the 96-update training loop:
eight prompts times twelve passes. The four-row evaluation cardinality is
unchanged.

## Executable admission

Before any model sample, all 32 certified routes across 16 maps must replay
under the official networkless PointMaze checker, with two topology keys per
map. Admission also requires 100 perturbation replays per route, exact
runtime/source binding, failure-mutation rejection, no split overlap, and the
existing throughput floor.

Only a passing audit with decision
`admitted_to_point_maze_v3_balanced_viability_gate` permits a fresh
development-only 0.5B K=16 viability sample. The v2 pair remains immutable and
will be reported as the diagnostic antecedent regardless of its terminal
result.
