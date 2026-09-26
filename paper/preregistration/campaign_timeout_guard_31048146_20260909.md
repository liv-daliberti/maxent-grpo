# Additional bounded Pantry timeout guard — September 9, 2026

The user requested recovering broken E118/E119/E120 jobs and accelerating
completion. This operational extension covers only existing E118 Qwen2.5-3B
Pantry ReplayMaxRL seed 71, Slurm job 31048146. At inspection it was RUNNING
at step 431/3072 with fewer than five hours left in its original 12-hour
allocation. Current training continues uninterrupted. No outcome is inspected.

The September 8 guard and its four targets remain unchanged. This separate
CPU-only guard inherits that reviewed controller design with a distinct
singleton lock, audit directory, and one-job target set. Its frozen plan is
`var/artifacts/campaign_timeout_guard_31048146_20260909/plan.json` and its
controller is `ops/exp_scaling/guard_pantry_timeout_31048146_20260909.py`.

The guard polls every 30 seconds for at most 48 hours, retaining its deadline
across restarts. It preserves the exact source, exported training settings,
registered 3072-step horizon, checkpoint cadence, memory, GPU, CPU, node pool,
partition, account, priority and original 12-hour allocation limit. It does
not modify an experiment ledger or submit replacement science jobs.

Only an inactive, accounted TIMEOUT under the existing Slurm ID can be
requeued. Require a controller record, no terminal receipt, a valid saved
model/optimizer checkpoint strictly between steps 0 and 3072, and no other
active or pending writer. Freeze and recheck identity and resource fields.
Never alter RUNNING, CONFIGURING, COMPLETING, SUSPENDED or PENDING jobs.

Allow at most three guard-initiated retries and require a strictly newer
checkpoint after every prior guard retry. Record durable intent before the
same-ID requeuehold call; release only the verified owned hold with exactly
one incremented restart count and unchanged recipe/resources. Do not repeat
an uncertain scheduler operation. Changed identity/resources, missing or
nonadvancing checkpoint, another writer, any terminal state other than
TIMEOUT, or an exhausted retry cap stops this target for manual inspection.
A terminal receipt completes the guard. These checks are the same as in
`paper/preregistration/campaign_timeout_guard_20260908.md`.
