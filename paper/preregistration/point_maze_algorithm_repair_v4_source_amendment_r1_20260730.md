# PointMaze v4 source-feasibility amendment r1

**Status:** FROZEN AFTER THE FIRST V4 SOURCE GATE FAILED AND BEFORE V4 DATA
ADMISSION OR MODEL SAMPLING  
**Date:** 2026-07-30

The first deterministic source gate stopped on the first `large_block13_v4`
canonical route: proposed counts `(14, 40, 17)` did not pass the official
checker. No dataset, admission receipt, or language-model sample was produced.

R1 changes only the canonical known-route pulse counts. On the already frozen
four 13x13 geometries, a checker-only lexicographic search tested:

- lateral pulses 10 through 24;
- even forward pulses 32 through 56;
- return pulses equal to lateral plus three;
- rotations 0 and 2;
- both upper and lower certified programs.

The first common passing tuple for all four families and both rotations was
`(13, 32, 16)`. That tuple is now fixed. Geometry, split assignment, seeds,
thresholds, model, K=16 gate, optimizer, and algorithm are unchanged. This
source calibration used no model generations or training outcomes.
