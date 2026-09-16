# E105 post-gate automatic release transaction

Frozen on 2026-08-18 while the outcome-blind effective E104/E106/E110 gate was
14/15 terminal and E110 job 30647379 had not started. This is scheduler
automation only. It changes no model, data, seed, optimizer, evaluator,
treatment, comparator, placement rule, stopping rule, or analysis.

One CPU-only Slurm job depends on `afterany:30647379`. After that dependency
ends, the transaction refreshes the combined outcome-blind mechanism audit and
fails closed unless it exits successfully with all 15 effective cells complete
and passed. Only then does it, in order:

1. render the complete mechanism diagnostic;
2. apply the already registered paired Qwen-3B A6000 placement amendment;
3. submit and release the 15 repaired-parser E109 Python ReplayDr.GRPO controls;
4. submit and release the 75 E105 repaired-v6 treatment cells; and
5. print the unified campaign status.

The dependency is `afterany`, not `afterok`, because Slurm can retain a stale
aggregate `PREEMPTED` state even when a receipt-backed run completed. The fresh
scientific audit, not the scheduler label, decides release. Any failed command
stops the transaction. PointMaze and post-update efficacy outcomes are excluded.

