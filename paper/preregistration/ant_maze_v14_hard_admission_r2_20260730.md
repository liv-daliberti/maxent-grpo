# AntMaze v14-hard admission r2: immutable snapshot repair

**Status: FROZEN AFTER R1 CONFIGURATION AND BEFORE R1 SUBMISSION — 2026-07-30**

R1 configuration passed, but its run phase stopped before `sbatch`: the source
snapshot created for failed job 30204495 contained newly generated
`__pycache__` files and no longer matched its content-addressed path. No r1
route job was submitted.

R2 uses a fresh source-snapshot namespace and exports
`PYTHONDONTWRITEBYTECODE=1` inside the admission job. This is a provenance-only
repair. It does not change maps, routes, seeds, controller, simulator,
horizons, split membership, or admission criteria from v14-hard r1.

