# PointMaze balanced algorithm repair v2

**Status: FROZEN AFTER V1 DEVELOPMENT-GATE FAILURE AND BEFORE V2 ROUTE EXECUTION OR MODEL SAMPLING — 2026-07-30**

The original PointMaze training rows were saturated: 15.98 of 16 rollouts
verified on average, so fewer than 3% of updates had any task advantage.
Repair v1 moved all four families to 11x11 geometry. It produced 19 verified
completions among 256 development draws but zero successes in the frozen
eight-draw prefixes, so it was not eligible for paired training.

V2 is a development-calibrated, balanced medium/hard slate. It selects the two
v1 geometry-shift families that previously showed both early success and
multiple eventual route modes (`cross9_shift` and `upper_offset9_shift`) and
the two 11x11 families with the strongest eventual verified support
(`block11_repair` and `bar11_repair`). They are rematerialized under new family
names, reset seeds, map IDs, fingerprints, and split hashes:

- `cross9_balanced`
- `upper_offset9_balanced`
- `block11_balanced`
- `bar11_balanced`

Admission is unchanged: 16 unique maps total (eight train, four development,
four evaluation), two real checker-accepted topology modes per map, 100
trajectory perturbations per route, exact runtime/source binding,
failure-mutation rejection, split disjointness, and the existing throughput
floor.

The frozen 0.5B warm start is sampled only on the four development rows, with
seed 76520, 64 draws per row, and an eight-draw prefix. Viability requires at
least one prefix-success prompt, at least one multimode prompt, nonzero
verified completion, and less than 90% verified completion.

If viability passes, the first paired calibration uses seed 76521, three
passes, 16 rollouts per prompt, learning rate 2e-7, binary official task
reward, within-prompt Dr.GRPO centering, and compute-matched verified MaxEnt.
Checkpoint evaluation uses common random numbers: arm seed, row, draw, sample,
and decision round determine request seeds, while checkpoint update does not.
Both arms must finish with verified rollout rate in [10%, 90%] and at least
24 of 96 updates with nonzero task advantage. A failed gate stops v2. The
original PointMaze campaign remains immutable and is reported separately.
