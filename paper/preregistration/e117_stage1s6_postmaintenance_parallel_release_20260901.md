# E117 Stage-1 S6 post-maintenance parallel release

Date: 2026-09-01

## Trigger

The cluster-wide maintenance reservation ending at 11:00 EDT left the intact
Falcon-1B MathIR seed-203 block transitively dependent on earlier seed blocks.
Three earlier block leaders had exhausted temporary five-hour backfill limits
and had already aged out of the live Slurm controller.  This is scheduler state,
not an outcome-triggered scientific change.

## Amendment

Remove only the cross-seed dependency from job `30980508`, the frozen first arm
of the MathIR seed-203 block.  Preserve its model, data, seed, F treatment,
source snapshot, optimizer, evaluation schedule, requested A5000, node105
placement, eight-hour limit, and the within-block dependencies of jobs
`30980509` and `30980510`.

The purpose is to allow the independent seed-203 block to use post-maintenance
`mltheory` capacity without waiting for checkpoint-resume replacements for the
timed-out seed-201 and seed-202 leaders.  No endpoint or telemetry was inspected
to select this amendment.
