#!/usr/bin/env bash
# Submit a MaxEnt alpha sweep with bounded concurrency.
# Defaults:
# - alphas: 0,0.01,0.02,0.04,0.06,0.08,0.10,0.12,0.15
# - seeds: 0,1,2,3,4
# - max_steps: 1000
# - concurrency: 2 jobs at once (via dependency lanes)
#
# Uses the maintained dual launcher in run-only=maxent mode.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

SLURM_SCRIPT="${SLURM_SCRIPT:-var/repo/ops/slurm/train_dual_4plus4.slurm}"
MODEL="${MODEL:-Qwen2.5-0.5B-Instruct}"
CONFIG_SUFFIX="${CONFIG_SUFFIX:-math}"
ACCELERATOR="${ACCELERATOR:-zero3}"
MAX_STEPS="${MAX_STEPS:-1000}"
CONCURRENCY="${CONCURRENCY:-2}"
ALPHAS_CSV="${ALPHAS_CSV:-0,0.01,0.02,0.04,0.06,0.08,0.10,0.12,0.15}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2,3,4}"
DRY_RUN="${DRY_RUN:-0}"
JOB_PREFIX="${JOB_PREFIX:-maxent-alpha-sweep}"
MAXENT_VLLM_GPU_LOCAL="${MAXENT_VLLM_GPU_LOCAL:-0}"
MAXENT_TRAIN_GPUS_LOCAL="${MAXENT_TRAIN_GPUS_LOCAL:-1,2,3}"
PORT_BASE_START="${PORT_BASE_START:-18000}"
PORT_STRIDE="${PORT_STRIDE:-10}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-open-r1-maxent-alpha-sweep}"
WANDB_ENTITY_NAME="${WANDB_ENTITY_NAME:-}"

# Resource defaults tuned for run-only=maxent on 4 GPUs.
SBATCH_GRES_DEFAULT="${SBATCH_GRES:-gpu:a100:4}"
SBATCH_CPUS_DEFAULT="${SBATCH_CPUS_PER_TASK:-32}"
SBATCH_MEM_DEFAULT="${SBATCH_MEM:-128G}"
# Use same-user exclusivity so two 4-GPU jobs can share a node.
SBATCH_EXCLUSIVE_MODE="${SBATCH_EXCLUSIVE_MODE:-user}"

if [[ ! -f "$SLURM_SCRIPT" ]]; then
  echo "[error] Missing Slurm script: $SLURM_SCRIPT" >&2
  exit 1
fi
if ! [[ "$CONCURRENCY" =~ ^[0-9]+$ ]] || (( CONCURRENCY < 1 )); then
  echo "[error] CONCURRENCY must be a positive integer (got: $CONCURRENCY)" >&2
  exit 1
fi
if ! [[ "$MAX_STEPS" =~ ^[0-9]+$ ]] || (( MAX_STEPS < 1 )); then
  echo "[error] MAX_STEPS must be a positive integer (got: $MAX_STEPS)" >&2
  exit 1
fi
if ! [[ "$PORT_BASE_START" =~ ^[0-9]+$ ]] || (( PORT_BASE_START < 1024 || PORT_BASE_START > 64000 )); then
  echo "[error] PORT_BASE_START must be an integer in [1024, 64000] (got: $PORT_BASE_START)" >&2
  exit 1
fi
if ! [[ "$PORT_STRIDE" =~ ^[0-9]+$ ]] || (( PORT_STRIDE < 3 )); then
  echo "[error] PORT_STRIDE must be an integer >= 3 (got: $PORT_STRIDE)" >&2
  exit 1
fi

MODEL_TAG="${MODEL//\//-}"
MODEL_TAG="${MODEL_TAG// /-}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_GROUP="${WANDB_RUN_GROUP:-${JOB_PREFIX}_${MODEL_TAG}_${CONFIG_SUFFIX}_${STAMP}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-var/data/${MODEL_TAG}-maxent-alpha-sweep-steps${MAX_STEPS}-${STAMP}}"

SBATCH_ARGS=()
if [[ -n "${SBATCH_PARTITION:-}" ]]; then SBATCH_ARGS+=(--partition "$SBATCH_PARTITION"); fi
if [[ -n "${SBATCH_ACCOUNT:-}" ]]; then SBATCH_ARGS+=(--account "$SBATCH_ACCOUNT"); fi
if [[ -n "${SBATCH_TIME:-}" ]]; then SBATCH_ARGS+=(--time "$SBATCH_TIME"); fi
if [[ -n "${SBATCH_CONSTRAINT:-}" ]]; then SBATCH_ARGS+=(--constraint "$SBATCH_CONSTRAINT"); fi
if [[ -n "$SBATCH_GRES_DEFAULT" ]]; then SBATCH_ARGS+=(--gres "$SBATCH_GRES_DEFAULT"); fi
if [[ -n "$SBATCH_CPUS_DEFAULT" ]]; then SBATCH_ARGS+=(--cpus-per-task "$SBATCH_CPUS_DEFAULT"); fi
if [[ -n "$SBATCH_MEM_DEFAULT" ]]; then SBATCH_ARGS+=(--mem "$SBATCH_MEM_DEFAULT"); fi
if [[ "${SBATCH_EXCLUSIVE_MODE:-}" != "none" && -n "${SBATCH_EXCLUSIVE_MODE:-}" ]]; then
  SBATCH_ARGS+=(--exclusive="$SBATCH_EXCLUSIVE_MODE")
fi

IFS=',' read -r -a RAW_ALPHAS <<< "$ALPHAS_CSV"
ALPHAS=()
for raw in "${RAW_ALPHAS[@]}"; do
  alpha="$(echo "$raw" | xargs)"
  [[ -z "$alpha" ]] && continue
  if ! [[ "$alpha" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "[error] Invalid alpha value: '$alpha'" >&2
    exit 1
  fi
  ALPHAS+=("$alpha")
done
if (( ${#ALPHAS[@]} == 0 )); then
  echo "[error] No alpha values parsed from ALPHAS_CSV='$ALPHAS_CSV'" >&2
  exit 1
fi

IFS=',' read -r -a RAW_SEEDS <<< "$SEEDS_CSV"
SEEDS=()
for raw in "${RAW_SEEDS[@]}"; do
  seed="$(echo "$raw" | xargs)"
  [[ -z "$seed" ]] && continue
  if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
    echo "[error] Invalid seed value: '$seed'" >&2
    exit 1
  fi
  SEEDS+=("$seed")
done
if (( ${#SEEDS[@]} == 0 )); then
  echo "[error] No seed values parsed from SEEDS_CSV='$SEEDS_CSV'" >&2
  exit 1
fi

declare -a LANE_LAST_JOB
for ((lane=0; lane<CONCURRENCY; lane++)); do
  LANE_LAST_JOB[$lane]=""
done

declare -a SUBMITTED
TOTAL_JOBS=$(( ${#ALPHAS[@]} * ${#SEEDS[@]} ))

echo "[info] run_group=$RUN_GROUP"
echo "[info] output_root=$OUTPUT_ROOT"
echo "[info] alphas=${ALPHAS[*]}"
echo "[info] seeds=${SEEDS[*]}"
echo "[info] concurrency=$CONCURRENCY max_steps=$MAX_STEPS"
echo "[info] total_jobs=$TOTAL_JOBS"
echo "[info] resources: gres=$SBATCH_GRES_DEFAULT cpus=$SBATCH_CPUS_DEFAULT mem=$SBATCH_MEM_DEFAULT exclusive=${SBATCH_EXCLUSIVE_MODE:-none}"
echo "[info] local maxent layout: vllm_gpu=$MAXENT_VLLM_GPU_LOCAL train_gpus=$MAXENT_TRAIN_GPUS_LOCAL"
echo "[info] wandb_project=$WANDB_PROJECT_NAME"

combo_idx=0
for alpha in "${ALPHAS[@]}"; do
  alpha_tag="${alpha//./p}"
  for seed in "${SEEDS[@]}"; do
    lane=$((combo_idx % CONCURRENCY))
    port_base=$((PORT_BASE_START + combo_idx * PORT_STRIDE))
    if (( port_base + 2 > 65535 )); then
      echo "[error] Computed port range exceeds 65535 (base=${port_base}); increase PORT_BASE_START/PORT_STRIDE settings." >&2
      exit 1
    fi
    maxent_vllm_port="$port_base"
    maxent_group_port=$((port_base + 1))
    maxent_master_port=$((port_base + 2))
    run_name="${RUN_GROUP}-a${alpha_tag}-s${seed}"
    output_dir="${OUTPUT_ROOT}/alpha_${alpha}/seed_${seed}"
    job_name="${JOB_PREFIX}-a${alpha_tag}-s${seed}"

    maxent_args=(
      "--maxent_alpha" "$alpha"
      "--seed" "$seed"
      "--max_steps" "$MAX_STEPS"
      "--output_dir" "$output_dir"
      "--run_name" "$run_name"
      "--wandb_project" "$WANDB_PROJECT_NAME"
      "--wandb_run_group" "$RUN_GROUP"
    )
    maxent_args_str="${maxent_args[*]}"

    job_env=(
      "WANDB_PROJECT=${WANDB_PROJECT_NAME}"
      "MAXENT_VLLM_GPU=${MAXENT_VLLM_GPU_LOCAL}"
      "MAXENT_TRAIN_GPUS=${MAXENT_TRAIN_GPUS_LOCAL}"
      "MAXENT_VLLM_PORT=${maxent_vllm_port}"
      "MAXENT_GROUP_PORT=${maxent_group_port}"
      "MAXENT_MASTER_PORT=${maxent_master_port}"
    )
    if [[ -n "$WANDB_ENTITY_NAME" ]]; then
      job_env+=("WANDB_ENTITY=${WANDB_ENTITY_NAME}")
    fi

    cmd=(env "${job_env[@]}" sbatch "${SBATCH_ARGS[@]}" --job-name "$job_name")
    if [[ -n "${LANE_LAST_JOB[$lane]}" ]]; then
      cmd+=(--dependency "afterany:${LANE_LAST_JOB[$lane]}")
    fi
    cmd+=(
      "$SLURM_SCRIPT"
      --model "$MODEL"
      --config "$CONFIG_SUFFIX"
      --accelerator "$ACCELERATOR"
      --run-only maxent
      --maxent-args "$maxent_args_str"
    )

    echo "[submit lane=$lane] ${cmd[*]}"
    if [[ "$DRY_RUN" == "1" ]]; then
      combo_idx=$((combo_idx + 1))
      continue
    fi

    out="$("${cmd[@]}")"
    job_id="$(awk '{print $4}' <<< "$out")"
    if ! [[ "$job_id" =~ ^[0-9]+$ ]]; then
      echo "[error] Failed to parse sbatch job id from: $out" >&2
      exit 1
    fi
    LANE_LAST_JOB[$lane]="$job_id"
    SUBMITTED+=("$job_id")
    combo_idx=$((combo_idx + 1))
  done
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[done] dry-run complete; no jobs submitted."
  exit 0
fi

echo "[done] submitted ${#SUBMITTED[@]} jobs."
echo "[done] job ids: ${SUBMITTED[*]}"
