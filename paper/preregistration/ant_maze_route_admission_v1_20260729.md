# AntMaze v5 controller route admission

**Status: FROZEN BEFORE THE FIRST ADMISSION-MAP EXECUTION — 2026-07-29**

## Antecedent

The prospective Ant heading-controller v5 gate passed on 96 fresh open-plane
episodes before any maze route or language-model sample was observed. Its
immutable receipt SHA-256 is
`324d3301b8e21b4dbbfcc2e6b9a87aba479bd2da5aa3a040cff24adebaf8828e`.
This authorizes an AntMaze route gate only.

Development-only interface engineering used one unregistered map identity to
establish a fixed 75-simulator-step command window and two candidate program
shapes. Those exact map and reset identities are excluded below. No threshold,
admission map, perturbation, or model completion was inspected.

## Frozen admission slate

Materialize 12 fresh `AntMaze_UMaze-v5` specifications: four train, four
development, and four evaluation. Each is a distinct seven-by-seven map with
a central blocking cell and a preregistered nonempty subset of peripheral
right-column walls. Every specification fixes:

- reset cell `[3, 2]`, goal cell `[3, 4]`;
- one unique reset seed in `76300..76311`;
- the eight compass action tokens, with no `STOP`;
- 75 simulator steps per language token;
- 8–32 tokens per program;
- the official sparse goal threshold `0.5`; and
- two x=0 route gates whose nonoverlapping spans identify upper and lower
  obstacle-side crossings.

The same two source-frozen verifier fixtures are attempted once on every map:

- upper: `N` x5, `E` x6, `S` x5, `SE` x2, `NE`; and
- lower: `S` x6, `E` x7, `N` x10, `NE`.

A failed fixture makes that map ineligible. Do not replace a map, alter a
program, change its reset seed, or tune the command window after execution.
Fixtures and route catalogues never enter policy prompts or replay banks.

## Audit and decision

The gate passes only if:

1. all 24 real executions reach the goal and yield exactly two distinct
   topology keys per map;
2. 100 deterministic sub-cell trajectory perturbations per route preserve
   their key, for 2,400 perturbation replays with zero mismatch;
3. malformed programs, success near-misses, environment/controller identity
   mutations, and cross-route collisions are rejected;
4. train/development/evaluation specification fingerprints are disjoint;
5. the persistent external worker completes at least 0.15 real Ant
   executions per second over the full 24-execution slate; and
6. every runtime, environment, XML, controller, data, source, and audit hash
   is recorded.

Passing admits AntMaze only to its development-only Qwen2.5-0.5B viability
sample. It does not authorize a training arm or any of the 80 confirmatory
jobs.
