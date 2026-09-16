# E113-R1-M1-S2: Qwen smoke backfill amendment

**Frozen:** 2026-08-19 16:22 EDT, while M1 job 30790590 was pending at zero
updates with no run directory and no scheduler start estimate.

## Observed scheduling condition

The registered A6000 nodes were healthy but their currently free resources
were marked planned for higher-priority work. The M1 request retained an
eight-hour wall-time ceiling even though the 32-update smoke is expected to
finish well inside two hours. Slurm therefore reported
`ReqNodeNotAvail, May be reserved for other job` and no backfill start.

## Single admitted change

Reduce only job 30790590's pending scheduler wall-time ceiling from eight hours
to two hours. Keep its job ID, fresh run path, A6000 node set, `lowprio`
partition, `mltheory` account, 64 GB host-memory request, one-GPU topology,
complete 16-response optimizer batch, fused Adam, disabled offloads, model,
domain, seed, prompts, verifier, DAPO objective, 32-update target, ten-batch
generation cap, and 5,120-query ceiling unchanged.

The amendment is valid only while the job is pending and no output directory
exists. Record the complete scheduler state before and after the update in an
atomic amendment ledger. This is a scheduling repair, not a new scientific or
operational trajectory.
