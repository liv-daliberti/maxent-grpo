# E117-R2 prospective scheduler backfill amendment

Frozen: 2026-08-29, after the scheduler-placement amendment and before any
E117-R2 job started or any E117-R2 output directory existed.

After the 12 original jobs became eligible on the preregistered A5000 nodes,
Slurm projected their eight-hour requests serially from 2026-08-31 through
2026-09-03. This is a scheduler fit issue, not a training observation.

The directly matched E117-R1 cells on the same physical nodes completed in
11:06--48:37. R2 changes only generic mechanism/audit correctness and adds the
same lightweight estimator traversal already present in the F arm. A two-hour
limit is therefore more than 2.4 times the slowest matched runtime while making
the jobs substantially easier to backfill.

Before any R2 execution, update only `TimeLimit` from 08:00:00 to 02:00:00 for
jobs 30970803--30970814. Preserve every job ID, account/partition placement,
dependency, run stamp, exported environment, source snapshot, physical-node
pin, CPU/GPU/memory request, mechanism setting, data request, seed, and audit
rule. Do not change the audit job.

This amendment uses scheduler metadata only. It is not informed by an E117-R2
endpoint or training metric, does not alter the common-source block, and does
not authorize Stage 1. A time-limit failure remains a fail-closed R2 failure;
it may not be waived or combined with an old cell.
