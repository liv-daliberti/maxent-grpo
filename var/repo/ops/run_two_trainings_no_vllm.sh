#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SLURM="$SCRIPT_DIR/slurm/train_no_vllm.slurm"

if [[ ! -f "$TRAIN_SLURM" ]]; then
  echo "Missing train.slurm at $TRAIN_SLURM" >&2
  exit 1
fi

MODEL="${MODEL:-Qwen2.5-0.5B-Instruct}"
CONFIG_SUFFIX="${CONFIG_SUFFIX:-math_no_vllm}"
ACCELERATOR="${ACCELERATOR:-zero3}"
VLLM_PORT_BASE="${VLLM_PORT_BASE:-29525}"
VLLM_GROUP_PORT_BASE="${VLLM_GROUP_PORT_BASE:-29535}"
JOB_PREFIX="${JOB_PREFIX:-open_r1}"

MAXENT_ARGS="${MAXENT_ARGS:-}"
GRPO_ARGS="${GRPO_ARGS:-}"

SBATCH_ARGS=()
if [[ -n "${SBATCH_PARTITION:-}" ]]; then SBATCH_ARGS+=(--partition "$SBATCH_PARTITION"); fi
if [[ -n "${SBATCH_ACCOUNT:-}" ]]; then SBATCH_ARGS+=(--account "$SBATCH_ACCOUNT"); fi
if [[ -n "${SBATCH_TIME:-}" ]]; then SBATCH_ARGS+=(--time "$SBATCH_TIME"); fi
if [[ -n "${SBATCH_NODES:-}" ]]; then SBATCH_ARGS+=(--nodes "$SBATCH_NODES"); fi
if [[ -n "${SBATCH_GRES:-}" ]]; then SBATCH_ARGS+=(--gres "$SBATCH_GRES"); fi
if [[ -n "${SBATCH_CPUS_PER_TASK:-}" ]]; then SBATCH_ARGS+=(--cpus-per-task "$SBATCH_CPUS_PER_TASK"); fi

PORT1="$VLLM_PORT_BASE"
PORT2="$((VLLM_PORT_BASE + 1))"
GROUP_PORT1="$VLLM_GROUP_PORT_BASE"
GROUP_PORT2="$((VLLM_GROUP_PORT_BASE + 1))"

MAXENT_JOB_NAME="${JOB_PREFIX}_maxent_0p5b"
GRPO_JOB_NAME="${JOB_PREFIX}_grpo_0p5b"

set -x
MAXENT_JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" --job-name "$MAXENT_JOB_NAME" \
  "$TRAIN_SLURM" \
  --model "$MODEL" \
  --task maxent-grpo \
  --config "$CONFIG_SUFFIX" \
  --accelerator "$ACCELERATOR" \
  --vllm-port "$PORT1" \
  --vllm-group-port "$GROUP_PORT1" \
  --args "$MAXENT_ARGS" | awk '{print $4}')

GRPO_JOB_ID=$(sbatch "${SBATCH_ARGS[@]}" --job-name "$GRPO_JOB_NAME" \
  --dependency "afterany:$MAXENT_JOB_ID" \
  "$TRAIN_SLURM" \
  --model "$MODEL" \
  --task grpo \
  --config "$CONFIG_SUFFIX" \
  --accelerator "$ACCELERATOR" \
  --vllm-port "$PORT2" \
  --vllm-group-port "$GROUP_PORT2" \
  --args "$GRPO_ARGS" | awk '{print $4}')

set +x
echo "Submitted: $MAXENT_JOB_NAME (job $MAXENT_JOB_ID)"
echo "Submitted: $GRPO_JOB_NAME (job $GRPO_JOB_ID)"
