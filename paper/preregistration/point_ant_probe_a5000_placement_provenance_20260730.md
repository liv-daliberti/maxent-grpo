# PointMaze/AntMaze probe A5000 placement provenance

**Placement mutation executed before allocation or outcome — 2026-07-30**

The following already-frozen probe jobs were pending at runtime `00:00:00`
under the congested MLTheory placement when the scheduler-only mutation was
made:

- PointMaze paired smoke control: `30201821`;
- PointMaze paired smoke verified MaxEnt: `30201822`;
- AntMaze v13 warm-start viability: `30202183`.

The mutation retained each job ID and changed only the effective scheduler
account and partition from `mltheory/mltheory` to `allcs/all`.  The requested
accelerator remained one A5000, with 8 CPUs and 64 GB of memory.  PointMaze's
two arms retained identical resource requests and were allowed to co-schedule
without a fixed node.  No job was requeued or restarted.

The scientific identities were already frozen and remain unchanged.  In
particular, this mutation did not change source or execution snapshots,
commands, model, data, arm, seed, prompt/task rows, rollouts, optimizer
updates, sampling, evaluation, admission thresholds, dependency chains, or
audit logic.  The relevant frozen snapshot hashes are:

- PointMaze source `8c37e7de5db2d4e2bb582443cce40f6c6ee111f0c95c53994ca6874d5e6825c8`;
- PointMaze execution `5e6028a9742d2dd853a342ff1f4301c3961adc495b4f7424d84e3c2e8c3046c0`;
- AntMaze source `e27bacef2758c8aeedb809f54e431ea7b64a03ab170b1d713733f06da9da2bcd`;
- AntMaze execution `3c4cf122137e9ce3c31fffad7d9cf6efd2930087c8acf1a3c7244514ec437d67`.

The scheduler subsequently allocated all three jobs to `node202`: PointMaze
control at `2026-07-30T05:15:08`, and PointMaze treatment plus AntMaze at
`2026-07-30T05:15:09`.  This record transcribes that prospective,
outcome-neutral placement action; it does not authorize a rerun, threshold
change, seed replacement, or post-outcome environment substitution.
