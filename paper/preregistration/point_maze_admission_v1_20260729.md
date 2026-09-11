# PointMaze language-action admission v1

Status: frozen after source feasibility and before admission audit or 0.5B sampling  
Frozen: 2026-07-29

## Already observed feasibility

The source-feasibility stage executed two hand-designed language action
programs for each of 16 specifications (8 train, 4 development, 4
evaluation). All 32 reached the MuJoCo goal and the route extractor separated
the two obstacle-side routes. These programs are verifier fixtures and may
not appear in a model prompt, replay bank, training batch, or viability
completion.

The 16 specifications cover four obstacle geometries. Exact map/start/goal
specifications do not overlap between splits: rotations 0 and 2 are train,
rotation 1 is development, and rotation 3 is evaluation.

## Frozen admission audit

Before any 0.5B sample:

- recompute the installed Python/package/module/XML runtime identity;
- require the worker to reject a mismatched environment hash;
- replay all 32 source-feasibility programs through one persistent external
  worker and reproduce their two distinct route keys per prompt;
- for every one of the 32 successful trajectories, replay 100 deterministic
  sub-cell/timing perturbations through the dependency-free canonicalizer;
- require all 3,200 perturbations to retain the original key;
- require no route-key collision within any prompt;
- require explicit unsuccessful, near-miss, and hash-mismatch mutations to
  fail closed;
- retain the parser, teleport, bounds, ambiguous-crossing, jitter,
  recrossing, and external reward/key regression tests; and
- measure at least 2.0 complete simulator executions per wall-clock second
  over the 32-program persistent-worker replay.

Sub-cell perturbations add at most `0.001` coordinate units to each interior
trajectory point and may replace one interior point with the midpoint of its
neighbors. They are identity-stability tests over already successful
simulator trajectories, not additional environment-success claims.

Any failure makes PointMaze ineligible until a prospectively separated
version is defined. Thresholds, maps, programs, gates, and split membership
may not change after this audit is inspected.

## Next gate

Passing this audit admits development-only Qwen2.5-0.5B viability sampling.
It does not admit the confirmatory 48-job cohort. The viability prompt set,
draw count, and criterion must be frozen before the first model completion.
