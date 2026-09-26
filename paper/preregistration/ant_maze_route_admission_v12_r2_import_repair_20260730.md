# AntMaze v12 route admission r2 generator-import repair

**Status: FROZEN AFTER TERMINAL R1 JOB 30200026 AND BEFORE R2 IMPLEMENTATION OR EXECUTION — 2026-07-30**

R1 correctly amended and released the sealed v12 job, but it failed in five
seconds before a worker process or simulator reset. The v12 generator imported
the v11 generator solely to inherit frozen constants. Importing that module
eagerly called the v11 controller identity validator, which rejected the v12
r1 identity schema. No route, map, seed, controller action, perturbation, data
row, or audit was executed.

R2 is an operations-only import repair. The v12 generator imports the common
base generator directly and restates the exact v11 constants: upper `N E E S`,
lower `S E E N`, eight peripheral cells, map size 11, wall/reset/goal geometry,
bounds, seeds `107300..107311`, action range 4--16, action repeat 400, and the
v12 anchored executor identity. The sealed v12 source snapshot is reused
byte-for-byte; only a fresh operations snapshot is allowed. R2 uses a fresh
identity and job, explicitly amends the held partition to `all`, and binds the
canceled v12 identity and terminal r1 import-failure log. No scientific,
controller, executor, map, route, seed, threshold, or audit change is allowed.
