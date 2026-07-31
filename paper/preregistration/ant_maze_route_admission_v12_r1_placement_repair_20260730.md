# AntMaze v12 route admission r1 held-partition repair

**Status: FROZEN AFTER CANCELED HELD JOB 30200003 AND BEFORE R1 SUBMISSION — 2026-07-30**

The v12 launcher submitted job `30200003` with literal
`sbatch --partition=all`, but this cluster recorded the still-held job in
partition `cs`. Its pre-release assertion expected `Partition=all`, so the
cleanup trap canceled it with `RunTime=00:00:00`, `StartTime` equal to
`EndTime`, and no stdout, stderr, data root, audit, simulator reset, or route
execution. The v12 source and execution snapshots and its identity are sealed
antecedents.

R1 makes one scheduler-only repair. It submits the exact sealed v12 source and
execution snapshots as a new held job, verifies zero runtime and the unchanged
CPU, memory, time, export, source, and execution contract, explicitly amends
the held job to `Partition=all`, verifies that amendment, sets `Requeue=0`,
writes a fresh identity bound to the canceled attempt, and only then releases
the job. The route geometry, maps, seeds, programs, controller, anchored
targeting, thresholds, perturbations, audit, and information boundary are
unchanged. No map or route has been executed between v12 and r1.
