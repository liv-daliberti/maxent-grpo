# E117-R2 prospective scheduler-placement amendment

Frozen: 2026-08-29, after release but before any E117-R2 job started or any
E117-R2 output directory existed.

The 12 registered jobs (30970803--30970814) were submitted with
`--partition=all --account=mltheory` and the preregistered physical-node pins
node202/node203. The site job-submission plugin rewrote those jobs to partition
`mltheory`. That partition contains only node105, node302, and node915--917, so
all 12 jobs remained pending with `Reason=BadConstraints`, elapsed time
`00:00:00`, no assigned node, and no output directory.

Slurm's non-submitting `sbatch --test-only` check established that
`--partition=all --account=allcs` maps to partition `cs`, which contains the
preregistered node202/node203. Therefore, before any execution, update only the
12 jobs' scheduler account from `mltheory` to `allcs` and effective partition
from `mltheory` to `cs`. Preserve every job ID, dependency, run stamp, exported
environment, source snapshot, physical-node pin, resource request, mechanism
setting, data request, seed, and audit rule.

This amendment changes placement eligibility only. It is not informed by any
endpoint or training telemetry, does not alter the common-source block, and
does not authorize Stage 1. The official frozen-snapshot audit remains
dependency-gated on all 12 original job IDs and must still pass fail closed.
