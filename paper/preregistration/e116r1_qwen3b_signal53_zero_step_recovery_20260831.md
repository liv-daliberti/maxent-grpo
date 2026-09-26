# E116-R1 Qwen-3B signal-53 zero-step recovery

Frozen: 2026-08-31 after scheduler accounting inspection and before any
replacement submission or E116 Qwen-3B run artifact existed.

## Trigger and boundary

E116 Qwen-3B smoke job 30790466 received an allocation on node302 but exited
immediately with Slurm exit code `0:53` and elapsed time `00:00:00`.  Its audit
job 30790467 and all 25 scientific jobs 30790468--30790492 were consequently
cancelled without an allocation or runtime.  None of the smoke or scientific
run directories exists.

Fourteen pool audits completed successfully. Python/s70, s71, and s73
collection jobs completed, but their audits found zero replay-eligible prompts.
Under E116's frozen hard gate these three scientific cells are infeasible and
must be reported as blocked; their pools may not be regenerated. Eight MathIR
and Pantry collection jobs were cancelled at zero runtime before producing an
artifact and are eligible for exact infrastructure retries.

This establishes a zero-step scheduler failure boundary: no model update,
sample, evaluation, checkpoint, or endpoint was produced or inspected.

## Authorized replacement

Replace the failed smoke, its audit, and the 22 feasible dependency-cancelled
scientific jobs.  Reconstruct each command from its held scheduler record in
`e116_sparse_rlep_qwen3b_jobs.json`.  Preserve the immutable source snapshot,
model and data identities, seeds, domains, sparse-RLEP treatment, optimizer,
sampling, evaluation and checkpoint schedules, run paths, resources, node302
placement, account, partition, time limits, and Nice values byte-for-byte.

Because Slurm no longer recognizes the old completed job IDs for dependencies,
fresh CPU audits recheck the 14 immutable valid pools. The eight zero-runtime
collection jobs and their audits are replaced. The only command changes are
fresh scheduler job IDs and dependencies:

1. the replacement smoke depends on the fresh Graph/s70 re-audit;
2. the replacement smoke audit depends after-ok on the replacement smoke; and
3. every feasible scientific job depends after-ok on its fresh pool audit and
   on the replacement smoke audit.

Submit all 54 jobs under user hold. Verify the zero-step boundary, pool
outcomes, absent output paths, frozen command contracts, and held scheduler
records before updating the ledger. Release the graph only after every job
passes.  On any pre-release failure, cancel only newly submitted jobs and leave
the original ledger unchanged.

This is an operational zero-step replacement within E116, not a new treatment,
seed, endpoint, or analysis opportunity.
