#!/usr/bin/env bash
# Submit GRPO + MaxEnt jobs in parallel with server-mode vLLM.
#
# Default per job:
# - 4 nodes total
# - 3 training nodes + 1 vLLM node (handled by slurm/train.slurm)
# - 8x A100 GPUs per node

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SLURM="${TRAIN_SLURM:-$SCRIPT_DIR/slurm/train.slurm}"
if [[ ! -f "$TRAIN_SLURM" ]]; then
  echo "Missing train launcher: $TRAIN_SLURM" >&2
  exit 1
fi

MODEL="${MODEL:-Qwen2.5-0.5B-Instruct}"
CONFIG_SUFFIX="${CONFIG_SUFFIX:-math}"
ACCELERATOR="${ACCELERATOR:-zero3}"
JOB_PREFIX="${JOB_PREFIX:-open_r1}"

NODES_PER_JOB="${NODES_PER_JOB:-4}"
GRES="${GRES:-gpu:a100:8}"
SKIP_RESOURCE_PREFLIGHT="${SKIP_RESOURCE_PREFLIGHT:-0}"

SBATCH_PARTITION="${SBATCH_PARTITION:-}"
SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-}"
SBATCH_TIME="${SBATCH_TIME:-}"
SBATCH_CONSTRAINT="${SBATCH_CONSTRAINT:-}"

VLLM_PORT_BASE="${VLLM_PORT_BASE:-8000}"
VLLM_GROUP_PORT_BASE="${VLLM_GROUP_PORT_BASE:-29535}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-6000}"

RUN_ONLY="$(echo "${RUN_ONLY:-both}" | tr '[:upper:]' '[:lower:]')"
DRY_RUN="${DRY_RUN:-0}"

MAXENT_ARGS="${MAXENT_ARGS:-}"
GRPO_ARGS="${GRPO_ARGS:-}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_GROUP="${WANDB_RUN_GROUP:-${JOB_PREFIX}_${MODEL//\//-}_${CONFIG_SUFFIX}_${STAMP}}"

append_flag_value() {
  local args="$1"
  local flag="$2"
  local value="$3"
  if [[ " $args " == *" ${flag} "* || "$args" == *"${flag}="* ]]; then
    printf '%s\n' "$args"
  else
    if [[ -n "$args" ]]; then
      printf '%s %s %s\n' "$args" "$flag" "$value"
    else
      printf '%s %s\n' "$flag" "$value"
    fi
  fi
}

preflight_partition_resources() {
  if [[ "$SKIP_RESOURCE_PREFLIGHT" == "1" ]]; then
    return 0
  fi
  if [[ -z "$SBATCH_PARTITION" ]]; then
    # No explicit partition requested; skip strict preflight.
    return 0
  fi
  if [[ "$GRES" != gpu:* ]]; then
    return 0
  fi

  local req_model=""
  local req_count=""
  local req_token="$GRES"
  if [[ "$req_token" =~ ^gpu:([^:]+):([0-9]+)$ ]]; then
    req_model="${BASH_REMATCH[1]}"
    req_count="${BASH_REMATCH[2]}"
  elif [[ "$req_token" =~ ^gpu:([0-9]+)$ ]]; then
    req_count="${BASH_REMATCH[1]}"
  else
    # Unknown GRES format; don't block submit.
    return 0
  fi

  local sinfo_rows=""
  sinfo_rows="$(timeout 20s sinfo -p "$SBATCH_PARTITION" -N -h -o '%N %G' 2>/dev/null || true)"
  if [[ -z "$sinfo_rows" ]]; then
    echo "[warn] Preflight skipped: unable to read nodes for partition '$SBATCH_PARTITION' via sinfo."
    return 0
  fi

  local eligible=0
  local details=()
  while read -r node gres_field; do
    [[ -z "${node:-}" || -z "${gres_field:-}" ]] && continue
    local node_match=0
    IFS=',' read -r -a gres_tokens <<< "$gres_field"
    for tok in "${gres_tokens[@]}"; do
      tok="${tok%%(*}" # strip Slurm suffixes like "(S:0-1)"
      if [[ -n "$req_model" ]]; then
        if [[ "$tok" =~ ^gpu:${req_model}:([0-9]+)$ ]]; then
          if (( ${BASH_REMATCH[1]} >= req_count )); then
            node_match=1
            break
          fi
        fi
      else
        if [[ "$tok" =~ ^gpu:[^:]+:([0-9]+)$ ]]; then
          if (( ${BASH_REMATCH[1]} >= req_count )); then
            node_match=1
            break
          fi
        fi
      fi
    done
    if (( node_match == 1 )); then
      eligible=$((eligible + 1))
      details+=("${node} ${gres_field}")
    fi
  done <<< "$sinfo_rows"

  local jobs_to_submit=0
  case "$RUN_ONLY" in
    both) jobs_to_submit=2 ;;
    grpo|maxent) jobs_to_submit=1 ;;
    *) jobs_to_submit=1 ;;
  esac
  local need_for_one="$NODES_PER_JOB"
  local need_for_parallel=$((NODES_PER_JOB * jobs_to_submit))

  if (( eligible < need_for_one )); then
    echo "[error] Requested shape cannot be satisfied in partition '$SBATCH_PARTITION':" >&2
    echo "        requested per job: nodes=${NODES_PER_JOB}, gres=${GRES}" >&2
    echo "        eligible nodes found: ${eligible}" >&2
    if (( ${#details[@]} > 0 )); then
      echo "        eligible node types:" >&2
      printf '          %s\n' "${details[@]}" >&2
    else
      echo "        no nodes in '$SBATCH_PARTITION' match gres=${GRES}" >&2
    fi
    echo "        tip: lower NODES_PER_JOB, change GRES, or use a partition with more matching nodes." >&2
    return 1
  fi

  if (( jobs_to_submit > 1 && eligible < need_for_parallel )); then
    echo "[warn] Parallel paired run requested (${jobs_to_submit} jobs), but only ${eligible} matching nodes are visible."
    echo "[warn] One or both jobs may remain queued if resources are busy."
  fi
}

build_args() {
  local base_args="$1"
  local run_name="$2"
  local out="$base_args"
  out="$(append_flag_value "$out" "--run_name" "$run_name")"
  out="$(append_flag_value "$out" "--wandb_run_group" "$RUN_GROUP")"
  out="$(append_flag_value "$out" "--vllm_mode" "server")"
  out="$(append_flag_value "$out" "--use_vllm" "true")"
  printf '%s\n' "$out"
}

SBATCH_ARGS=(--nodes "$NODES_PER_JOB" --gres "$GRES")
if [[ -n "$SBATCH_PARTITION" ]]; then SBATCH_ARGS+=(--partition "$SBATCH_PARTITION"); fi
if [[ -n "$SBATCH_ACCOUNT" ]]; then SBATCH_ARGS+=(--account "$SBATCH_ACCOUNT"); fi
if [[ -n "$SBATCH_TIME" ]]; then SBATCH_ARGS+=(--time "$SBATCH_TIME"); fi
if [[ -n "$SBATCH_CONSTRAINT" ]]; then SBATCH_ARGS+=(--constraint "$SBATCH_CONSTRAINT"); fi

submit_job() {
  local job_name="$1"
  local task="$2"
  local master_port="$3"
  local vllm_port="$4"
  local vllm_group_port="$5"
  local args_str="$6"

  local cmd=(
    sbatch
    "${SBATCH_ARGS[@]}"
    --job-name "$job_name"
    --export "ALL,MASTER_PORT=${master_port}"
    "$TRAIN_SLURM"
    --model "$MODEL"
    --task "$task"
    --config "$CONFIG_SUFFIX"
    --accelerator "$ACCELERATOR"
    --vllm-port "$vllm_port"
    --vllm-group-port "$vllm_group_port"
    --args "$args_str"
  )

  echo "[submit] ${cmd[*]}"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "${cmd[@]}"
}

if [[ "$RUN_ONLY" != "both" && "$RUN_ONLY" != "grpo" && "$RUN_ONLY" != "maxent" ]]; then
  echo "RUN_ONLY must be one of: both | grpo | maxent" >&2
  exit 1
fi

preflight_partition_resources

MAXENT_RUN_NAME="${WANDB_MAXENT_RUN_NAME:-${RUN_GROUP}-maxent}"
GRPO_RUN_NAME="${WANDB_GRPO_RUN_NAME:-${RUN_GROUP}-grpo}"
MAXENT_ARGS="$(build_args "$MAXENT_ARGS" "$MAXENT_RUN_NAME")"
GRPO_ARGS="$(build_args "$GRPO_ARGS" "$GRPO_RUN_NAME")"

MAXENT_JOB_NAME="${JOB_PREFIX}_maxent_0p5b"
GRPO_JOB_NAME="${JOB_PREFIX}_grpo_0p5b"

PORT_MAXENT="$VLLM_PORT_BASE"
PORT_GRPO="$((VLLM_PORT_BASE + 1))"
GROUP_PORT_MAXENT="$VLLM_GROUP_PORT_BASE"
GROUP_PORT_GRPO="$((VLLM_GROUP_PORT_BASE + 1))"
MASTER_MAXENT="$MASTER_PORT_BASE"
MASTER_GRPO="$((MASTER_PORT_BASE + 1))"

if [[ "$RUN_ONLY" == "both" || "$RUN_ONLY" == "maxent" ]]; then
  submit_job "$MAXENT_JOB_NAME" "maxent-grpo" "$MASTER_MAXENT" "$PORT_MAXENT" "$GROUP_PORT_MAXENT" "$MAXENT_ARGS"
fi
if [[ "$RUN_ONLY" == "both" || "$RUN_ONLY" == "grpo" ]]; then
  submit_job "$GRPO_JOB_NAME" "grpo" "$MASTER_GRPO" "$PORT_GRPO" "$GROUP_PORT_GRPO" "$GRPO_ARGS"
fi

echo "Run group: $RUN_GROUP"
