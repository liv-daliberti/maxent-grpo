#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SLURM="$SCRIPT_DIR/slurm/train.slurm"

if [[ ! -f "$TRAIN_SLURM" ]]; then
  echo "Missing train.slurm at $TRAIN_SLURM" >&2
  exit 1
fi

MODEL="${MODEL:-Qwen2.5-0.5B-Instruct}"
CONFIG_SUFFIX="${CONFIG_SUFFIX:-math}"
ACCELERATOR="${ACCELERATOR:-zero3}"
VLLM_PORT_BASE="${VLLM_PORT_BASE:-8000}"
VLLM_GROUP_PORT_BASE="${VLLM_GROUP_PORT_BASE:-29535}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-6000}"
JOB_PREFIX="${JOB_PREFIX:-open_r1}"
SUBMIT_MODE="$(echo "${SUBMIT_MODE:-sequential}" | tr '[:upper:]' '[:lower:]')"
STAGGER_MINUTES="${STAGGER_MINUTES:-60}"
STAGGER_SAME_NODE="${STAGGER_SAME_NODE:-1}"
STAGGER_WAIT_TIMEOUT_SECS="${STAGGER_WAIT_TIMEOUT_SECS:-1800}"
# Set SBATCH_SHARE_NODE=1 to pass --exclusive=user so both jobs can share a node
# with your own jobs while remaining exclusive against other users.
if [[ -z "${SBATCH_SHARE_NODE+x}" ]]; then
  if [[ "$SUBMIT_MODE" == "parallel" || "$SUBMIT_MODE" == "staggered" ]]; then
    SBATCH_SHARE_NODE=1
  else
    SBATCH_SHARE_NODE=0
  fi
fi

MAXENT_ARGS="${MAXENT_ARGS:-}"
GRPO_ARGS="${GRPO_ARGS:-}"
# Keep paired runs aligned with recipe defaults. Override to colocate explicitly
# via `PAIR_VLLM_MODE=colocate` when needed.
PAIR_VLLM_MODE="${PAIR_VLLM_MODE:-server}"

MODEL_TAG="${MODEL//\//-}"
MODEL_TAG="${MODEL_TAG// /-}"
RUN_GROUP="${WANDB_RUN_GROUP:-${JOB_PREFIX}_${MODEL_TAG}_${CONFIG_SUFFIX}_$(date +%Y%m%d_%H%M%S)}"
export WANDB_RUN_GROUP="$RUN_GROUP"
export MAXENT_WANDB_METRICS_MODE="${MAXENT_WANDB_METRICS_MODE:-slim}"
export MAXENT_EVAL_LOGPROBS="${MAXENT_EVAL_LOGPROBS:-1}"
# Skip per-job preflight pip installs (preinstall once in the env).
export MAXENT_SKIP_PREFLIGHT_PIP="${MAXENT_SKIP_PREFLIGHT_PIP:-1}"
# Turn on full debug logging by default for this script (override by exporting
# MAXENT_MAX_LOGS=0 before running).
export MAXENT_MAX_LOGS="${MAXENT_MAX_LOGS:-1}"
export MAXENT_PROGRESS_LOG="${MAXENT_PROGRESS_LOG:-1}"

MAXENT_RUN_NAME="${WANDB_MAXENT_RUN_NAME:-${RUN_GROUP}-maxent}"
GRPO_RUN_NAME="${WANDB_GRPO_RUN_NAME:-${RUN_GROUP}-grpo}"

if [[ "$MAXENT_ARGS" != *"--run_name"* ]]; then
  MAXENT_ARGS="${MAXENT_ARGS:+$MAXENT_ARGS }--run_name ${MAXENT_RUN_NAME}"
fi
if [[ "$GRPO_ARGS" != *"--run_name"* ]]; then
  GRPO_ARGS="${GRPO_ARGS:+$GRPO_ARGS }--run_name ${GRPO_RUN_NAME}"
fi
if [[ "$MAXENT_ARGS" != *"--wandb_run_group"* ]]; then
  MAXENT_ARGS="${MAXENT_ARGS:+$MAXENT_ARGS }--wandb_run_group ${RUN_GROUP}"
fi
if [[ "$GRPO_ARGS" != *"--wandb_run_group"* ]]; then
  GRPO_ARGS="${GRPO_ARGS:+$GRPO_ARGS }--wandb_run_group ${RUN_GROUP}"
fi
if [[ "$MAXENT_ARGS" != *"--vllm_mode"* ]]; then
  MAXENT_ARGS="${MAXENT_ARGS:+$MAXENT_ARGS }--vllm_mode ${PAIR_VLLM_MODE}"
fi
if [[ "$GRPO_ARGS" != *"--vllm_mode"* ]]; then
  GRPO_ARGS="${GRPO_ARGS:+$GRPO_ARGS }--vllm_mode ${PAIR_VLLM_MODE}"
fi

case "$SUBMIT_MODE" in
  sequential|parallel|staggered) ;;
  *)
    echo "Unsupported SUBMIT_MODE='$SUBMIT_MODE' (use: sequential|parallel|staggered)" >&2
    exit 1
    ;;
esac

SBATCH_COMMON_ARGS=()
if [[ -z "${SBATCH_NODES:-}" ]]; then
  SBATCH_NODES=1
fi
if [[ -n "${SBATCH_PARTITION:-}" ]]; then SBATCH_COMMON_ARGS+=(--partition "$SBATCH_PARTITION"); fi
if [[ -n "${SBATCH_ACCOUNT:-}" ]]; then SBATCH_COMMON_ARGS+=(--account "$SBATCH_ACCOUNT"); fi
if [[ -n "${SBATCH_TIME:-}" ]]; then SBATCH_COMMON_ARGS+=(--time "$SBATCH_TIME"); fi
if [[ -n "${SBATCH_NODES:-}" ]]; then SBATCH_COMMON_ARGS+=(--nodes "$SBATCH_NODES"); fi
if [[ -n "${SBATCH_NODELIST:-}" ]]; then SBATCH_COMMON_ARGS+=(--nodelist "$SBATCH_NODELIST"); fi
if [[ -n "${SBATCH_CPUS_PER_TASK:-}" ]]; then SBATCH_COMMON_ARGS+=(--cpus-per-task "$SBATCH_CPUS_PER_TASK"); fi
if [[ "$SBATCH_SHARE_NODE" == "1" ]]; then SBATCH_COMMON_ARGS+=(--exclusive=user); fi

MAXENT_GRES="${MAXENT_GRES:-${SBATCH_GRES:-}}"
GRPO_GRES="${GRPO_GRES:-${SBATCH_GRES:-}}"
if [[ "$SUBMIT_MODE" != "sequential" && -z "$MAXENT_GRES" && -z "$GRPO_GRES" ]]; then
  MAXENT_GRES="gpu:a100:4"
  GRPO_GRES="gpu:a100:4"
fi

MAXENT_SBATCH_ARGS=("${SBATCH_COMMON_ARGS[@]}")
GRPO_SBATCH_ARGS=("${SBATCH_COMMON_ARGS[@]}")
if [[ -n "$MAXENT_GRES" ]]; then MAXENT_SBATCH_ARGS+=(--gres "$MAXENT_GRES"); fi
if [[ -n "$GRPO_GRES" ]]; then GRPO_SBATCH_ARGS+=(--gres "$GRPO_GRES"); fi

if [[ "$SUBMIT_MODE" == "parallel" && "$SBATCH_SHARE_NODE" != "1" ]]; then
  echo "[warn] SUBMIT_MODE=parallel without SBATCH_SHARE_NODE=1 may still serialize because train.slurm uses '#SBATCH --exclusive'."
fi
if [[ "$SUBMIT_MODE" == "staggered" && "$SBATCH_SHARE_NODE" != "1" ]]; then
  echo "[warn] SUBMIT_MODE=staggered without SBATCH_SHARE_NODE=1 may serialize unexpectedly."
fi

PORT1="$VLLM_PORT_BASE"
PORT2="$((VLLM_PORT_BASE + 1))"
GROUP_PORT1="$VLLM_GROUP_PORT_BASE"
GROUP_PORT2="$((VLLM_GROUP_PORT_BASE + 1))"
MASTER1="$MASTER_PORT_BASE"
MASTER2="$((MASTER_PORT_BASE + 1))"
MAXENT_EXPORT="${MAXENT_EXPORT:-ALL,MASTER_PORT=${MASTER1}}"
GRPO_EXPORT="${GRPO_EXPORT:-ALL,MASTER_PORT=${MASTER2}}"
MAXENT_PORT="$PORT1"
GRPO_PORT="$PORT2"
MAXENT_GROUP_PORT="$GROUP_PORT1"
GRPO_GROUP_PORT="$GROUP_PORT2"
if [[ "$SUBMIT_MODE" == "staggered" ]]; then
  # In staggered mode we launch GRPO first, so keep it on the base ports.
  GRPO_PORT="$PORT1"
  MAXENT_PORT="$PORT2"
  GRPO_GROUP_PORT="$GROUP_PORT1"
  MAXENT_GROUP_PORT="$GROUP_PORT2"
fi

MAXENT_JOB_NAME="${JOB_PREFIX}_maxent_0p5b"
GRPO_JOB_NAME="${JOB_PREFIX}_grpo_0p5b"
GRPO_DEP_ARGS=()
MAXENT_DEP_ARGS=()
GRPO_NODE=""
GRPO_START=""
MAXENT_BEGIN=""

set -x
if [[ "$SUBMIT_MODE" == "staggered" ]]; then
  GRPO_JOB_ID=$(sbatch "${GRPO_SBATCH_ARGS[@]}" --job-name "$GRPO_JOB_NAME" \
    --export "$GRPO_EXPORT" \
    "$TRAIN_SLURM" \
    --model "$MODEL" \
    --task grpo \
    --config "$CONFIG_SUFFIX" \
    --accelerator "$ACCELERATOR" \
    --vllm-port "$GRPO_PORT" \
    --vllm-group-port "$GRPO_GROUP_PORT" \
    --args "$GRPO_ARGS" | awk '{print $4}')

  echo "Submitted: $GRPO_JOB_NAME (job $GRPO_JOB_ID)"
  echo "Waiting for GRPO job to start and get a node..."
  poll_s=5
  waited_s=0
  while (( waited_s < STAGGER_WAIT_TIMEOUT_SECS )); do
    GRPO_STATE="$(squeue -h -j "$GRPO_JOB_ID" -o "%T" 2>/dev/null | head -n1 || true)"
    GRPO_NODE="$(squeue -h -j "$GRPO_JOB_ID" -o "%N" 2>/dev/null | head -n1 || true)"
    GRPO_START="$(squeue -h -j "$GRPO_JOB_ID" -o "%S" 2>/dev/null | head -n1 || true)"
    if [[ "$GRPO_STATE" == "RUNNING" && -n "$GRPO_NODE" && "$GRPO_NODE" != "(null)" && -n "$GRPO_START" && "$GRPO_START" != "N/A" ]]; then
      break
    fi
    sleep "$poll_s"
    waited_s=$((waited_s + poll_s))
  done
  if [[ -z "$GRPO_NODE" || "$GRPO_NODE" == "(null)" || -z "$GRPO_START" || "$GRPO_START" == "N/A" ]]; then
    echo "Failed to resolve GRPO start/node within ${STAGGER_WAIT_TIMEOUT_SECS}s." >&2
    exit 1
  fi

  # squeue returns start times like 2026-02-23T18:44:20; normalize for `date -d`.
  GRPO_START_FOR_DATE="${GRPO_START/T/ }"
  MAXENT_BEGIN="$(date -d "${GRPO_START_FOR_DATE} +${STAGGER_MINUTES} minutes" '+%Y-%m-%dT%H:%M:%S' 2>/dev/null || true)"
  if [[ -z "${MAXENT_BEGIN}" ]]; then
    echo "Failed to compute MaxEnt begin time from GRPO_START='${GRPO_START}'." >&2
    exit 1
  fi
  MAXENT_DEP_ARGS=(--begin "$MAXENT_BEGIN")
  if [[ "$STAGGER_SAME_NODE" == "1" ]]; then
    MAXENT_DEP_ARGS+=(--nodelist "$GRPO_NODE")
  fi
  echo "Scheduling MaxEnt for ${MAXENT_BEGIN} on node ${GRPO_NODE} (same-node=${STAGGER_SAME_NODE})."

  MAXENT_JOB_ID=$(sbatch "${MAXENT_SBATCH_ARGS[@]}" --job-name "$MAXENT_JOB_NAME" \
    "${MAXENT_DEP_ARGS[@]}" \
    --export "$MAXENT_EXPORT" \
    "$TRAIN_SLURM" \
    --model "$MODEL" \
    --task maxent-grpo \
    --config "$CONFIG_SUFFIX" \
    --accelerator "$ACCELERATOR" \
    --vllm-port "$MAXENT_PORT" \
    --vllm-group-port "$MAXENT_GROUP_PORT" \
    --args "$MAXENT_ARGS" | awk '{print $4}')
else
  MAXENT_JOB_ID=$(sbatch "${MAXENT_SBATCH_ARGS[@]}" --job-name "$MAXENT_JOB_NAME" \
    --export "$MAXENT_EXPORT" \
    "$TRAIN_SLURM" \
    --model "$MODEL" \
    --task maxent-grpo \
    --config "$CONFIG_SUFFIX" \
    --accelerator "$ACCELERATOR" \
    --vllm-port "$MAXENT_PORT" \
    --vllm-group-port "$MAXENT_GROUP_PORT" \
    --args "$MAXENT_ARGS" | awk '{print $4}')

  if [[ "$SUBMIT_MODE" == "sequential" ]]; then
    GRPO_DEP_ARGS=(--dependency "afterany:$MAXENT_JOB_ID")
  fi

  GRPO_JOB_ID=$(sbatch "${GRPO_SBATCH_ARGS[@]}" --job-name "$GRPO_JOB_NAME" \
    "${GRPO_DEP_ARGS[@]}" \
    --export "$GRPO_EXPORT" \
    "$TRAIN_SLURM" \
    --model "$MODEL" \
    --task grpo \
    --config "$CONFIG_SUFFIX" \
    --accelerator "$ACCELERATOR" \
    --vllm-port "$GRPO_PORT" \
    --vllm-group-port "$GRPO_GROUP_PORT" \
    --args "$GRPO_ARGS" | awk '{print $4}')
fi

set +x
echo "Submitted: $MAXENT_JOB_NAME (job $MAXENT_JOB_ID)"
echo "Submitted: $GRPO_JOB_NAME (job $GRPO_JOB_ID)"
if [[ "$SUBMIT_MODE" == "staggered" ]]; then
  echo "Stagger config: GRPO first, MaxEnt starts ${STAGGER_MINUTES} minutes later${GRPO_NODE:+ on ${GRPO_NODE}}."
fi
