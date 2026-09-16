# E113-R4 / E116 / E112-R1 scheduler-only acceleration package

Frozen on 2026-08-21 EDT before application and explicitly author-approved in
the active campaign session. This amendment uses scheduler state, frozen launch
ledgers, node inventory, and file existence only. It does not inspect a target
outcome, change a scientific environment, create a cell, or alter an estimand.
PointMaze is excluded.

## Trigger

The approved targets are operationally unable to make useful progress under
their current placement:

- official DAPO R4 smoke jobs `30800804` and `30800805` are released but remain
  `PENDING (Priority)` at zero runtime in partition `cs`; all 50 R4 science
  jobs remain dependency-pending on both smokes;
- E116 Qwen-0.5B pool jobs `30790388`, `30790390`, `30790399`, and `30790401`
  are `PENDING (BadConstraints)` at zero runtime because submit-side routing
  changed their effective partition/account to `cs/allcs` while retaining the
  frozen `node105` A5000 requirement; and
- E112-R1 Falcon jobs `30791519`, `30791520`, `30791522`, `30791523`, and
  `30791529`--`30791533` are `PENDING (JobHeldUser)` at zero runtime under the
  already diagnosed single-node A6000 auto-hold pattern.

The live inventory exposes A6000 GPUs in partition `all` on
`node[103-104,205-208]`; account `allcs` is allowed there and `PreemptMode=OFF`.
Partition `mltheory` exposes node105's A5000 GPUs and requires account
`mltheory`. Node805 is deliberately excluded because it is down.

## Authorized changes

For DAPO smoke jobs `30800804` and `30800805`:

- change partition `cs` to `all`; and
- set the node requirement to `node[103-104,205-208]`.

They are already released, so no release action is needed. Retain account
`allcs`, one A6000, 16 CPUs, 128 GiB, the R4-P2 one-day smoke limit,
`Nice=0`, both one-step smoke recipes, their immutable official-verl runtime
snapshot, and all environment variables. The 50 science jobs and their
two-smoke `afterok` gate are not modified.

For E116 pool jobs `30790388`, `30790390`, `30790399`, and `30790401`:

- change partition `cs` to `mltheory`; and
- change account `allcs` to `mltheory`.

Retain node105, one A5000, 8 CPUs, 64 GiB, the 36-hour limit, `Nice=100`, each
frozen control checkpoint, collection seed, prompt set, sampling recipe,
output directory, and full environment. Their audit and science dependencies
are not modified.

For E112-R1 jobs `30791519`, `30791520`, `30791522`, `30791523`, and
`30791529`--`30791533`:

- change partition `cs` to `all`;
- widen the node requirement from its frozen single A6000 node to
  `node[103-104,205-208]`; and
- release the job.

Retain account `allcs`, one A6000, 8 CPUs, 64 GiB, the three-day limit,
`Nice=100`, model, domain, seed, optimizer, verified-support estimator,
proposal/replay settings, stopping rule, checkpoint policy, output directory,
source snapshots, and every environment variable. Existing run directories,
if any, are neither edited nor deleted; the frozen auto-resume policy remains
unchanged.

## Application and evidence boundary

`ops/exp_scaling/apply_scheduler_acceleration_package_20260821.py` must, before
mutation, verify every live `--export` block against its frozen ledger record,
the exact zero-runtime pending state, GPU/CPU/memory/time requirements, and
partition inventory. It records full before/after scheduler records and source
hashes in
`var/artifacts/scheduler_acceleration_package_20260821.json`.

The amendment changes placement eligibility only. DAPO remains non-efficacy
evidence until both smokes pass and scientific cells reach their registered
terminal endpoints. E116 pool collections must still pass their independent
audits before dependent sparse-RLEP science can run. E112-R1 remains excluded
from paper efficacy under its existing private-interim boundary.
