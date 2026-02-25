#!/usr/bin/env bash
# Submit one 8-GPU node that runs both GRPO and MaxEnt concurrently:
# - GRPO:  3 train + 1 vLLM (GPUs 0-3)
# - MaxEnt:3 train + 1 vLLM (GPUs 4-7)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SLURM_SCRIPT:-$SCRIPT_DIR/slurm/train_dual_4plus4.slurm}"
if [[ ! -f "$SLURM_SCRIPT" ]]; then
  echo "Missing Slurm script: $SLURM_SCRIPT" >&2
  exit 1
fi

MODEL="${MODEL:-Qwen2.5-0.5B-Instruct}"
CONFIG_SUFFIX="${CONFIG_SUFFIX:-math}"
ACCELERATOR="${ACCELERATOR:-zero3}"
JOB_NAME="${JOB_NAME:-open_r1_dual_4plus4}"
RUN_ONLY="$(echo "${RUN_ONLY:-both}" | tr '[:upper:]' '[:lower:]')"

SBATCH_PARTITION="${SBATCH_PARTITION:-mltheory}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-mltheory}"
SBATCH_TIME="${SBATCH_TIME:-48:00:00}"
SBATCH_CONSTRAINT="${SBATCH_CONSTRAINT:-}"
SBATCH_GRES="${SBATCH_GRES:-gpu:a100:8}"
SBATCH_CPUS_PER_TASK="${SBATCH_CPUS_PER_TASK:-64}"
SBATCH_MEM="${SBATCH_MEM:-256G}"

GRPO_ARGS="${GRPO_ARGS:-}"
MAXENT_ARGS="${MAXENT_ARGS:-}"
DRY_RUN="${DRY_RUN:-0}"

cmd=(
  sbatch
  --job-name "$JOB_NAME"
  --nodes 1
  --gres "$SBATCH_GRES"
  --cpus-per-task "$SBATCH_CPUS_PER_TASK"
  --mem "$SBATCH_MEM"
)
if [[ -n "$SBATCH_PARTITION" ]]; then cmd+=(--partition "$SBATCH_PARTITION"); fi
if [[ -n "$SBATCH_ACCOUNT" ]]; then cmd+=(--account "$SBATCH_ACCOUNT"); fi
if [[ -n "$SBATCH_TIME" ]]; then cmd+=(--time "$SBATCH_TIME"); fi
if [[ -n "$SBATCH_CONSTRAINT" ]]; then cmd+=(--constraint "$SBATCH_CONSTRAINT"); fi

cmd+=(
  "$SLURM_SCRIPT"
  --model "$MODEL"
  --config "$CONFIG_SUFFIX"
  --accelerator "$ACCELERATOR"
  --run-only "$RUN_ONLY"
)
if [[ -n "$GRPO_ARGS" ]]; then cmd+=(--grpo-args "$GRPO_ARGS"); fi
if [[ -n "$MAXENT_ARGS" ]]; then cmd+=(--maxent-args "$MAXENT_ARGS"); fi

echo "[submit] ${cmd[*]}"
if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi
"${cmd[@]}"
