# AntMaze route-gate runtime adapter repair

**Status: FROZEN BEFORE R1 SUBMISSION — 2026-07-29**

Job `30184765` stopped before the first admission-map execution because the
data materializer imported `gymnasium` through the paper Python environment.
The simulator packages intentionally exist only in the pinned maze runtime.
No data root or audit receipt was created, and no map, route, perturbation, or
language-model outcome was observed.

R1 changes one source-independent adapter call: obtain
`maze_runtime_identity()` by invoking
`var/maze_runtime/venv/bin/python`, exactly as the admitted PointMaze
materializer does. The 12 maps, split assignment, reset seeds, upper/lower
programs, 75-step command window, 2,400 perturbations, throughput floor, and
all decision criteria remain byte-for-byte unchanged.

R1 must use a fresh source/execution snapshot and a new job identity while
retaining job `30184765` as failed infrastructure provenance.
