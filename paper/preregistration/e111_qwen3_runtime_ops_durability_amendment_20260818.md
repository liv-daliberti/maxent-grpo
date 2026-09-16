# E111 Qwen-3B runtime-ops durability amendment

Date frozen: 2026-08-18, before changing the E111 runtime-ops snapshot and
without inspecting any E111 endpoint evaluation result.

## Correction to the first durability amendment

The earlier file
`e111_qwen3_restart_durability_amendment_20260818.md` placed an exact-job block
in the checkout Slurm wrapper. `scontrol write batch_script` subsequently
verified that Slurm requeues execute the batch script copied at submission;
they do not reread the checkout wrapper. That first block is therefore inert
for the five already-submitted E111 jobs and cannot solve the observed restart
loop.

The stored batch script does, on every allocation, copy and execute
`train.sh` from the submitted `OAT_ZERO_OPS_SNAPSHOT_ROOT`. This amendment
places the same storage-only override in that actual restart-time input:

`var/artifacts/source_snapshots/e76_tuned_scale_2febfdc12e36650d/ops/train.sh`

## Exact scope and change

This applies only to E111 Qwen-3B jobs `30674758`, `30674759`, `30674760`,
`30674761`, and `30674762`. The runtime block must additionally match the
exact E111 source snapshot, ops snapshot, treatment variant, and submitted
32-step storage values. On a mismatch it exits before constructing the
training command.

For the next and later allocation of an exact matching job, only these values
change:

- `OAT_ZERO_SAVE_STEPS`: 32 -> 8
- `OAT_ZERO_SAVE_FROM`: 32 -> 8
- `OAT_ZERO_RESUME_STEPS`: 32 -> 8

The block is installed prospectively. Attempts already running continue with
their copied pre-amendment script. No running job is killed, requeued, reset,
or duplicated.

## Invariants

- Python source is unchanged.
- Model, seed, data, prompt order, optimizer, learning-rate schedule, MaxEnt
  estimator/coefficient, ReplayDr objective/weight, proposal policy, verifier,
  evaluation settings, and target step count are unchanged.
- Proposal rows remain excluded from PPO and on-policy counts.
- Atomic checkpoint serialization, exact optimizer/RNG restoration, and the
  two-checkpoint retention limit are unchanged.
- The five original job IDs and run directories remain authoritative.
- PointMaze remains excluded.
- The change is motivated only by scheduler restart counts and step/checkpoint
  telemetry, not reward, accuracy, coverage, or another endpoint.

## Audit rule

The amendment record must preserve the E111 ledger and protocol digests, the
runtime `train.sh` before/after digests, the exact inserted block digest,
pre/post scheduler records and restart counts, and `scontrol write
batch_script` evidence that the stored wrapper rereads the frozen runtime-ops
path. The final E111 auditor must validate this record and exact live block.
