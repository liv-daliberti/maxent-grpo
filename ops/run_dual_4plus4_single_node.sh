#!/usr/bin/env bash
# Submit one node for the experiment launcher:
# - default: GRPO + entropy-MaxEnt concurrently on 8 GPUs
# - single-stack modes: grpo | maxent | listwise | seed on 4 GPUs via the same Slurm script

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SLURM_SCRIPT:-$SCRIPT_DIR/slurm/train_dual_4plus4.slurm}"
if [[ ! -f "$SLURM_SCRIPT" ]]; then
  echo "Missing Slurm script: $SLURM_SCRIPT" >&2
  exit 1
fi

CONFIG_SUFFIX="${CONFIG_SUFFIX:-math}"
MODEL="${MODEL:-}"
RECIPE_PROFILE="${RECIPE_PROFILE:-experiment}"
ACCELERATOR="${ACCELERATOR:-zero3}"
JOB_NAME="${JOB_NAME:-open_r1_dual_4plus4}"
RUN_ONLY="$(echo "${RUN_ONLY:-both}" | tr '[:upper:]' '[:lower:]')"

RESOURCE_PROFILE="${RESOURCE_PROFILE:-}"
if [[ -z "$RESOURCE_PROFILE" ]]; then
  case "$JOB_NAME" in
    cs_interim_*) RESOURCE_PROFILE="interim_a6000" ;;
    open_r1_quartet_*) RESOURCE_PROFILE="quartet_a100" ;;
    *) RESOURCE_PROFILE="default_a100" ;;
  esac
fi

case "$RESOURCE_PROFILE" in
  interim_a6000)
    SBATCH_PARTITION_DEFAULT="lowprio"
    SBATCH_ACCOUNT_DEFAULT="allcs"
    STEP0_PAPER_EVAL_ENFORCE_EXPECTED_DEFAULT="0"
    if [[ "$RUN_ONLY" == "both" ]]; then
      SBATCH_GRES_DEFAULT="gpu:a6000:8"
      SBATCH_CPUS_DEFAULT="64"
      SBATCH_MEM_DEFAULT="256G"
    else
      SBATCH_GRES_DEFAULT="gpu:a6000:4"
      SBATCH_CPUS_DEFAULT="32"
      SBATCH_MEM_DEFAULT="128G"
    fi
    ;;
  quartet_a100|default_a100)
    SBATCH_PARTITION_DEFAULT="mltheory"
    SBATCH_ACCOUNT_DEFAULT="mltheory"
    STEP0_PAPER_EVAL_ENFORCE_EXPECTED_DEFAULT="1"
    if [[ "$RUN_ONLY" == "both" ]]; then
      SBATCH_GRES_DEFAULT="gpu:a100:8"
      SBATCH_CPUS_DEFAULT="64"
      SBATCH_MEM_DEFAULT="256G"
    else
      SBATCH_GRES_DEFAULT="gpu:a100:4"
      # Allow two 4-GPU single-stack jobs to share one 8xA100 node.
      SBATCH_CPUS_DEFAULT="48"
      SBATCH_MEM_DEFAULT="240G"
    fi
    ;;
  *)
    echo "Unknown RESOURCE_PROFILE: $RESOURCE_PROFILE" >&2
    exit 1
    ;;
esac

SBATCH_PARTITION="${SBATCH_PARTITION:-$SBATCH_PARTITION_DEFAULT}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-$SBATCH_ACCOUNT_DEFAULT}"
SBATCH_TIME="${SBATCH_TIME:-48:00:00}"
SBATCH_CONSTRAINT="${SBATCH_CONSTRAINT:-}"
SBATCH_GRES="${SBATCH_GRES:-$SBATCH_GRES_DEFAULT}"
SBATCH_CPUS_PER_TASK="${SBATCH_CPUS_PER_TASK:-$SBATCH_CPUS_DEFAULT}"
SBATCH_MEM="${SBATCH_MEM:-$SBATCH_MEM_DEFAULT}"
export MAXENT_STEP0_PAPER_EVAL_ENFORCE_EXPECTED="${MAXENT_STEP0_PAPER_EVAL_ENFORCE_EXPECTED:-$STEP0_PAPER_EVAL_ENFORCE_EXPECTED_DEFAULT}"

GRPO_ARGS="${GRPO_ARGS:-}"
MAXENT_ARGS="${MAXENT_ARGS:-}"
DRY_RUN="${DRY_RUN:-0}"
SBATCH_DEPENDENCY="${SBATCH_DEPENDENCY:-}"

if [[ -z "$MODEL" ]]; then
  case "$CONFIG_SUFFIX" in
    math|math_fair|math_stable) MODEL="Qwen2.5-1.5B-Instruct" ;;
    code_mbpp) MODEL="Qwen2.5-0.5B-Instruct" ;;
    *) MODEL="Qwen2.5-1.5B-Instruct" ;;
  esac
fi

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
if [[ -n "$SBATCH_DEPENDENCY" ]]; then cmd+=(--dependency "$SBATCH_DEPENDENCY"); fi

cmd+=(
  "$SLURM_SCRIPT"
  --model "$MODEL"
  --config "$CONFIG_SUFFIX"
  --recipe-profile "$RECIPE_PROFILE"
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
