# E117-R1-S4 / E113-R4-R2-S2 scheduler-continuity amendment

Frozen: 2026-08-26 EDT after the user explicitly requested that E117 begin and
that official-verl DAPO continue, before this scheduler transaction and
without inspecting any incomplete-cell endpoint. PointMaze remains excluded.

## Outcome-blind diagnosis

All twelve E117 jobs remain pending at zero runtime with intact frozen exports
and absent run directories. Their repaired nodes (node101, node103, node104,
and node203) are healthy and expose both `lowprio` and the non-preempting `all`
partition, but the eight-hour lowprio reservations currently project a serial
start schedule from August 27 through August 31. Each preflight cell contains
only 64 registered optimizer steps and checkpoints every 32 steps.

Official-verl DAPO has four terminal Qwen Graph cells and 46 released cells
pending at zero runtime. The four completed cells each reached all 24 accepted
updates in 2:09--2:14 with exit 0:0; their batch peak RSS was 49--57 GiB. The
remaining jobs retain seven-day limits and Slurm reports them blocked by a
maintenance reservation. DAPO checkpoints every five updates and uses
automatic resume.

## Authorized scheduler-only transaction

- For exact E117 jobs `30873695`--`30873706`, change partition `lowprio` to
  `all` and time limit `08:00:00` to `06:00:00`.
- For the 46 exact pending DAPO jobs in `30869115`--`30869160`, change only the
  time limit from `7-00:00:00` to `12:00:00`.
- Leave completed DAPO jobs `30869111`--`30869114` untouched.

Retain all job IDs, dependencies, accounts, QOS, nice values, exact node
constraints, accelerator types, CPU and memory requests, images, source
snapshots, models, seeds, domains, treatments, data, optimizer settings,
sampling distributions, checkpoints, evaluation surfaces, and stopping rules.
Do not hold the jobs, because a hold/release cycle would reset their accrued
age; validate every job before any in-place update and roll back scheduler
fields on failure.

This amendment changes placement opportunity and reservation fit only. E117
remains 0/12 and DAPO remains 4/50 until their existing scientific cells write
registered terminal receipts.

