# AntMaze Stage-B audit runtime repair r2

Frozen on 2026-07-30 after all ten final AntMaze training jobs completed 0:0
and audit job 30203042 failed before replay.  The audit subprocess inherited
the immutable source snapshot but not `OAT_ZERO_REPO_ROOT`; its worker
therefore searched for the frozen v11 controller receipt beneath the source
snapshot and exited with `FileNotFoundError`.  No audit receipt was written.

Training jobs, terminal receipts, metrics, state replay, source and execution
snapshots, controller, model, data, arms, seeds, and auditor code are
unchanged.  The replacement submits the same immutable audit batch with
`OAT_ZERO_REPO_ROOT` bound to the repository root.  It is held while identity
hashes are updated, uses an explicit conjunction of all ten already-terminal
training jobs, and then runs the full independent replay audit from scratch.
