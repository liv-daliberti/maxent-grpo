# E50 Python-factor non-mltheory placement amendment

**Status: FROZEN BEFORE RETRY — 2026-07-24**

The first Python-factor submission attempt created six held jobs under prefix
`pye50_uncapped_normalized_canonical_haarnoja_05b_50ep_v1`. Its pre-release
audit found that `OAT_ZERO_COMPARATIVE_TASK=python_factor` was inherited by
Slurm but was not repeated explicitly in the submit record. The launcher
cancelled all six jobs while they were still held. No job ran, no model update
occurred, and the v1 manifest and identity are retained as an aborted
pre-release attempt.

At the user's direction, the clean retry moves away from the `mltheory`
account and its constrained nodes. The scientific contract in
`e50_python_factor_extension_20260724.md` is otherwise unchanged.

The first non-MLTheory retry (`v2_allcs`) was also canceled while fully held.
The site job-submit policy rewrote a direct `allcs/all` submission to
partition `cs`, which made its requested RTX 3090 nodes ineligible. No job ran.
The clean `v3_allcs` retry mirrors the already-running Countdown placement
repair: after held submission, it explicitly updates the scheduler fields to
`allcs/all`, QOS `none`, and the eligible RTX 3090 node set, audits the
effective fields, and only then releases the cohort.

## Retry placement

- Fresh prefix:
  `pye50_uncapped_normalized_canonical_haarnoja_05b_50ep_v3_allcs`.
- Account: `allcs`.
- Partition: `all`.
- Eligible nodes: `node020,node024,node025,node026`.
- GPU constraint: one `rtx_3090` per job.
- CPUs and memory: 8 CPUs and 64 GiB per job.
- Time limit: seven days.
- All six jobs are submitted held, moved to the effective amended placement,
  audited, and released together.

These nodes expose the same RTX 3090 accelerator class and had schedulable
capacity when this amendment was frozen. Placement is shared by both arms and
all seeds, so hardware class is controlled within the Python comparison.

The submitter now repeats `OAT_ZERO_COMPARATIVE_TASK=python_factor` explicitly
inside `--export`. The held-job audit requires that exact value before release.
It also verifies the source snapshot, execution snapshot, data root, protocol
identity, arms, seeds, 50-pass budget, evaluation cadence, controller
parameters, account, partition, node eligibility, GPU class, CPU count,
memory, and time limit.

The live monitor follows only the fresh v3 retry prefix. The aborted v1 and v2
pre-release attempts are not treated as experimental runs.
