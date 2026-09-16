# E95: Cross-scale plain-GRPO controls

**Frozen:** 2026-08-12, before release of the repaired scientific jobs.

## Question

Every reward-only control in the main scaling campaign uses Dr.GRPO. E95 asks
whether correct-mode collapse also occurs under ordinary GRPO, rather than
being an artifact of Dr.GRPO's removal of difficulty and length bias.

## Frozen cells

One training seed means one cell in each of the five registered ModeBench
domains: GraphColoring, Countdown, PythonFactors, MathIR, and PantryPlan.

- Qwen2.5-3B: seed 70, 5 cells.
- Falcon3-1B: seeds 55--59, 25 cells.
- Qwen2.5-0.5B: seeds 43--47, 25 cells.
- Total: 55 cells, one plain-GRPO arm.

Each cell inherits its model, data, prompt template, optimizer, learning rate,
placement, resource request, evaluation cadence, checkpoint cadence, and
3,072-update/eight-pass horizon from the matching reward-only control in E80-R1,
E79, or E78. The only objective change is `critic_type: drgrpo -> grpo`.
There is no replay, entropy bonus, semantic objective, or critic.

## Runtime and release guard

The exact immutable runtime used by each parent family is copied and overlaid
with only `src/oat_drgrpo/args.py`, `ops/run_experiment.sh`, and `ops/train.sh`.
The derived snapshot must admit `critic_type=grpo`, select
`grpo_plain_control`, and pass a no-training runtime preflight. All scientific
jobs are submitted held, audited against their frozen seed, variant, and
snapshot exports, written to family ledgers, and only then released.

## Failed pre-registration attempt

Job 30497762 (Qwen2.5-3B, GraphColoring, seed 70) inherited the unmodified E80-R1
runtime and failed before training with `Unknown OAT_ZERO_VARIANT=grpo_plain_control`.
It completed zero optimizer updates and contributes no result. The repaired
suite supersedes that failed ledger entry; its log remains as an audit record.

## Analysis and stopping

Report all submitted cells, including numerical or training instability.
Within each domain and seed, compare the E95 endpoint and trajectory directly
with the inherited reward-only Dr.GRPO control. Do not tune, replace, or drop a
cell based on observed reward, diversity, instability, or direction of effect.
Infrastructure failures may be retried with the same frozen cell and runtime.
