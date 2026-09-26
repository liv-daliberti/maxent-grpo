# E95 Qwen backfill walltime amendment

Date: 2026-08-18.

## Trigger

The same-A5000 placement and priority amendment released all nine untouched
Qwen2.5-0.5B E95 cells, but their inherited 36-hour walltime prevents useful
backfill on the four eligible A5000 nodes. Slurm currently projects starts from
August 20 through August 25 even though capacity is intermittently available.

All sixteen completed sibling cells used the identical 384-update horizon and
finished in 3:40:07--4:57:12. The pending cells have never started.

## Frozen amendment

- Reduce only `TimeLimit` from `1-12:00:00` to `12:00:00` for the nine pending
  Qwen cells.
- Preserve their job IDs, A5000 resource class, `all`/`allcs` placement,
  eligible node pool, seeds, frozen source snapshot, data, objective,
  optimizer, evaluation cadence, checkpoint cadence, and 384-update horizon.
- Twelve hours is more than twice the maximum completed sibling runtime. The
  registered step-192 rolling checkpoint, automatic resume, and watchdog
  requeue remain enabled if an A5000 run is nevertheless slower.

This is a scheduler/backfill amendment, not a scientific change. Application
is fail-closed: hold all nine zero-step jobs, audit immutable exports and the
current placement, update the walltime, record the change atomically, and only
then release them.
