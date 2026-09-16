# E118 capacity priority — 2026-09-05

The user requested prioritizing additional E118 starts. This is an operational
placement amendment for the existing 150-cell factorial. No scientific cell,
data, model, seed, objective, optimizer setting, evaluation cadence, eight-pass
horizon, run directory, or frozen runtime snapshot changes.

At the audit, 108 E118 cells were terminal and all 42 unfinished cells were
Qwen2.5-3B jobs waiting in Slurm. Three Pantry jobs had BadConstraints because
12-hour allcs submissions requesting `all` were rewritten to `cs`, where
node208 is unavailable. The other 39 jobs had lost the previously admitted
node202 route during replacement. Local `/etc/slurm/job_submit.lua` requires
new submissions to change account or partition; explicit `lowprio` and the
`mltheory` account are supported routes.

Placement changes:

- Queue eight durable-checkpoint continuations on node302 through mltheory:
  Python ReplayMaxRL seeds 72–74; Graph MaxRL seed 73; both arms of MathIR seed
  74 and Graph seed 72. The first four finish pairs whose other arm is already
  terminal. Selection uses completion and checkpoint progress only.
- Queue both Pantry arms for seeds 72–74 on node208 through allcs/lowprio,
  repairing the three invalid placements and assigning each pair to the same
  hardware route. These jobs retain automatic resume, requeue, and the existing
  192-step durable checkpoint cadence. Low-priority allocations may be preempted;
  an interruption resumes the latest durable state under the existing protocol.
- Restore node202 to the remaining pending allcs/cs E118 jobs, using the valid
  safe list node202,node205,node206,node207. The scheduler determines availability;
  a drained node remains unavailable until cluster administrators restore it.

All jobs retain one GPU, 16 CPUs, 128 GiB requested host memory, 12-hour walltime,
and the existing explicit PVL exclusion:
`node[004-008,020-026,101,103-104,403,805-808,901-902,906-909,911-914]`.
No running or completed allocation is stopped. E119 and E120 are unchanged.

Replacement submissions are held and audited before replacing old pending job
identities in the authoritative Qwen-3B ledger and rebuilding the aggregate.
Old pending jobs are held to prevent simultaneous writers, then cancelled only
after their audited replacement and ledger entry exist. Replacement exports
preserve the original scientific environment and explicitly bind ROOT_DIR,
OAT_ZERO_REPO_ROOT, MAXENT_GRPO_ROOT and MAXENT_GRPO_VAR_ROOT to the repository.
The previously amended E118 Ninja, node-local extension builds, and checkpoint
resume helper remain in use.

Execution evidence is recorded in
`var/artifacts/e118_capacity_priority_20260905.json`. Startup verification uses
each job's actual Slurm stdout, current-attempt metrics and checkpoint restoration;
prior historical maxima are not treated as proof of resumed training.

## Startup amendment: withdraw unproven A5000 placement

Graph MaxRL seed 74 (31048115) allocated on node202 but its actor failed with
`RuntimeError: vllm cannot load the model` before any new optimizer metrics.
The prior node202 evidence established allocation, not successful Qwen-3B
training. The exact underlying actor exception was not available in the main
stdout; no algorithm or GPU-memory setting was changed to force a start.

All 28 cs jobs therefore return to the valid A6000 pool node205,node206,node207.
The failed allocation is requeued under the same job ID after preserving its
startup log, then released on that pool. The eight node302 continuations and
six valid node208 low-priority placements remain as submitted. Node208 is still
blocked by scheduler priority despite being physically idle and is not counted
as running capacity. Evidence: `var/artifacts/e118_capacity_a5000_withdrawal_20260905.json`.

The three Python ReplayMaxRL starters on node302 (31073899–31073901) restored
step 1920 and produced new current-attempt optimizer metrics. Startup evidence
is saved under `var/artifacts/e118_capacity_startup_20260905/`.
