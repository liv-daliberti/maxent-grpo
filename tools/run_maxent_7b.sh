#!/usr/bin/env bash
# Submit paired 7B MaxEnt‑GRPO math runs (GRPO-only vs MaxEnt weighting) for 2000 steps.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SLURM_SCRIPT="ops/slurm/maxent-grpo.slurm"
MODEL="Qwen2.5-7B"
CONFIG_SUFFIX="math"
CONFIG_SUFFIX_GRPO="math_maxent_hparams"
ACCELERATOR="${ACCELERATOR:-zero3}"
MAX_STEPS="${MAX_STEPS:-2000}"
DRY_RUN="${DRY_RUN:-0}"
GPU_CONSTRAINT="${GPU_CONSTRAINT:-}"   # e.g., "a100|h100"
GPU_GRES="${GPU_GRES:-}"               # e.g., "gpu:a100:8"
# Optional: RUN_ONLY can be "grpo", "maxent", or "both" (default: both)
RUN_ONLY="${RUN_ONLY:-both}"

CONFIG_FILE_MAXENT="configs/recipes/${MODEL}/maxent-grpo/config_${CONFIG_SUFFIX}.yaml"
CONFIG_FILE_GRPO="configs/recipes/${MODEL}/grpo/config_${CONFIG_SUFFIX_GRPO}.yaml"
if [[ ! -f "$CONFIG_FILE_MAXENT" ]]; then
  echo "[error] Missing config: $CONFIG_FILE_MAXENT" >&2
  exit 1
fi
if [[ ! -f "$CONFIG_FILE_GRPO" ]]; then
  echo "[error] Missing config: $CONFIG_FILE_GRPO" >&2
  exit 1
fi

declare -a RUN_NAMES=(
  "qwen25-7b-grpo-only-math-2k"
  "qwen25-7b-maxent-math-2k"
)
declare -a OUTPUT_DIRS=(
  "var/data/Qwen2.5-7B-Open-R1-GRPO-math-2k"
  "var/data/Qwen2.5-7B-Open-R1-MaxEnt-GRPO-math-2k"
)
declare -a HUB_MODEL_IDS=(
  "od2961/Qwen2.5-7B-Open-R1-GRPO-math-2k"
  "od2961/Qwen2.5-7B-Open-R1-MaxEnt-GRPO-math-2k"
)
declare -a TRAIN_GRPO_FLAGS=(
  "true"   # GRPO objective only
  "false"  # MaxEnt weighting enabled
)
declare -a TASK_SUBDIRS=(
  "grpo"
  "maxent-grpo"
)
declare -a CONFIG_SUFFIXES=(
  "${CONFIG_SUFFIX_GRPO}"
  "${CONFIG_SUFFIX}"
)

for idx in "${!RUN_NAMES[@]}"; do
  variant="${TASK_SUBDIRS[$idx]}"
  if [[ "$RUN_ONLY" == "grpo" && "$variant" != "grpo" ]]; then
    continue
  fi
  if [[ "$RUN_ONLY" == "maxent" && "$variant" != "maxent-grpo" ]]; then
    continue
  fi
  run_name="${RUN_NAMES[$idx]}"
  output_dir="${OUTPUT_DIRS[$idx]}"
  hub_model_id="${HUB_MODEL_IDS[$idx]}"
  train_grpo="${TRAIN_GRPO_FLAGS[$idx]}"
  task_subdir="${TASK_SUBDIRS[$idx]}"
  config_suffix="${CONFIG_SUFFIXES[$idx]}"

  optional_args=(
    "--output_dir ${output_dir}"
    "--hub_model_id ${hub_model_id}"
    "--max_steps ${MAX_STEPS}"
    "--train_grpo_objective ${train_grpo}"
  )
  args_str="${optional_args[*]}"

  sbatch_opts=(--job-name "$run_name")
  if [[ -n "$GPU_CONSTRAINT" ]]; then
    sbatch_opts+=(--constraint "$GPU_CONSTRAINT")
  fi
  if [[ -n "$GPU_GRES" ]]; then
    sbatch_opts+=(--gres "$GPU_GRES")
  fi

  cmd=(sbatch "${sbatch_opts[@]}" "$SLURM_SCRIPT" --model "$MODEL" --config "$config_suffix" --task-subdir "$task_subdir" --accelerator "$ACCELERATOR" --args "$args_str")
  echo "[submit] ${cmd[*]}"
  if [[ "$DRY_RUN" != "1" ]]; then
    "${cmd[@]}"
  fi
done
