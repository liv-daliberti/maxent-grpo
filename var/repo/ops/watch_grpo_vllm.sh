#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/n/fs/similarity/maxent-grpo}"
TRAIN_SLURM="${TRAIN_SLURM:-$ROOT_DIR/var/repo/ops/slurm/train.slurm}"
TASK="${TASK:-grpo}"
MODEL="${MODEL:-Qwen2.5-0.5B-Instruct}"
CONFIG_SUFFIX="${CONFIG_SUFFIX:-math}"
ACCELERATOR="${ACCELERATOR:-zero3}"

ACCOUNT="${ACCOUNT:-mltheory}"
GRES="${GRES:-gpu:a100:4}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"

MAX_RESTARTS="${MAX_RESTARTS:-6}"
CHECK_INTERVAL_SECS="${CHECK_INTERVAL_SECS:-20}"
RUNNING_STARTUP_TIMEOUT_SECS="${RUNNING_STARTUP_TIMEOUT_SECS:-1800}"
STABLE_HEALTH_SECS="${STABLE_HEALTH_SECS:-600}"

BASE_VLLM_PORT="${BASE_VLLM_PORT:-8012}"
BASE_VLLM_GROUP_PORT="${BASE_VLLM_GROUP_PORT:-29547}"

JOB_ID="${1:-}"
if [[ -z "${JOB_ID}" ]]; then
  echo "Usage: $0 <initial_job_id>" >&2
  exit 1
fi

RESTART_COUNT=0
RUNNING_SINCE=0
HEALTHY_SINCE=0

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

current_state() {
  local state
  state="$(squeue -h -j "$JOB_ID" -o "%T" 2>/dev/null || true)"
  if [[ -n "$state" ]]; then
    echo "$state"
    return 0
  fi
  local final
  final="$(sacct -n -X -j "$JOB_ID" --format=State 2>/dev/null | head -n1 | awk '{print $1}' || true)"
  if [[ -n "$final" ]]; then
    echo "$final"
  else
    echo "UNKNOWN"
  fi
}

has_failure_signature() {
  local open_log="$ROOT_DIR/var/artifacts/logs/open_r1_4gpu_mlt-${JOB_ID}.out"
  local vllm_log="$ROOT_DIR/var/artifacts/logs/vllm-${JOB_ID}.out"
  rg -n \
    "NCCL error: invalid usage|init_communicator failed|POST /init_communicator/ HTTP/1\\.1\" 504|Timed out waiting for 1 +worker|CANCELLED .* SIGNAL Terminated" \
    "$open_log" "$vllm_log" >/dev/null 2>&1
}

has_healthy_signature() {
  local vllm_log="$ROOT_DIR/var/artifacts/logs/vllm-${JOB_ID}.out"
  if [[ ! -f "$vllm_log" ]]; then
    return 1
  fi
  rg -n "Application startup complete\\.|GET /health/ HTTP/1\\.1\" 200 OK|GET /health/ HTTP/1.1\" 200 OK" "$vllm_log" >/dev/null 2>&1
}

mark_restart() {
  local reason="$1"
  if (( RESTART_COUNT >= MAX_RESTARTS )); then
    log "Max restarts reached (${MAX_RESTARTS}). Last reason: ${reason}. Exiting."
    exit 1
  fi
  log "Restarting job ${JOB_ID}. Reason: ${reason}"
  scancel "$JOB_ID" >/dev/null 2>&1 || true
  sleep 5
  submit_job
  RESTART_COUNT=$((RESTART_COUNT + 1))
  RUNNING_SINCE=0
  HEALTHY_SINCE=0
}

submit_job() {
  local stamp run_name offset vllm_port group_port submit_out
  stamp="$(date +%Y%m%d_%H%M%S)"
  run_name="mltheory_4gpu_1plus3_${stamp}"
  offset=$((RANDOM % 500))
  vllm_port=$((BASE_VLLM_PORT + offset))
  group_port=$((BASE_VLLM_GROUP_PORT + offset))

  submit_out="$(sbatch \
    --job-name=open_r1_4gpu_mlt \
    --account="$ACCOUNT" \
    --gres="$GRES" \
    --time="$TIME_LIMIT" \
    --export=ALL,MAXENT_SKIP_PREFLIGHT_PIP=1,MAXENT_SKIP_EDITABLE_INSTALL=1,OPENR1_PARITY_MODE=1,VLLM_SERVER_ENFORCE_EAGER=false \
    "$TRAIN_SLURM" \
    --model "$MODEL" \
    --task "$TASK" \
    --config "$CONFIG_SUFFIX" \
    --accelerator "$ACCELERATOR" \
    --vllm-port "$vllm_port" \
    --vllm-group-port "$group_port" \
    --args "--vllm_mode server --run_name ${run_name}")"
  JOB_ID="$(awk '{print $4}' <<<"$submit_out")"
  log "Submitted replacement job: ${JOB_ID} (vllm_port=${vllm_port}, group_port=${group_port})"
}

log "Watchdog started for job ${JOB_ID} (max_restarts=${MAX_RESTARTS}, check_interval=${CHECK_INTERVAL_SECS}s)"
while true; do
  state="$(current_state)"
  reason="$(squeue -h -j "$JOB_ID" -o "%R" 2>/dev/null || true)"
  log "job=${JOB_ID} state=${state}${reason:+ reason=${reason}}"

  case "$state" in
    PENDING|CONFIGURING)
      RUNNING_SINCE=0
      HEALTHY_SINCE=0
      ;;
    RUNNING)
      now="$(date +%s)"
      if (( RUNNING_SINCE == 0 )); then
        RUNNING_SINCE="$now"
      fi
      if has_failure_signature; then
        mark_restart "startup logs contain known failure signature"
      elif has_healthy_signature; then
        if (( HEALTHY_SINCE == 0 )); then
          HEALTHY_SINCE="$now"
          log "Healthy vLLM signature detected for job ${JOB_ID}; beginning stable window."
        fi
        if (( now - HEALTHY_SINCE >= STABLE_HEALTH_SECS )); then
          log "Job ${JOB_ID} healthy for ${STABLE_HEALTH_SECS}s. Watchdog exiting successfully."
          exit 0
        fi
      elif (( now - RUNNING_SINCE > RUNNING_STARTUP_TIMEOUT_SECS )); then
        mark_restart "startup timeout (${RUNNING_STARTUP_TIMEOUT_SECS}s) without healthy vLLM signal"
      fi
      ;;
    COMPLETING)
      # Wait for terminal accounting state (COMPLETED/FAILED/...) before deciding.
      ;;
    COMPLETED)
      log "Job ${JOB_ID} completed successfully. Watchdog exiting."
      exit 0
      ;;
    FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL|PREEMPTED)
      mark_restart "state transitioned to ${state}"
      ;;
    *)
      # Keep monitoring unknown transient states unless they persist into failure.
      ;;
  esac

  sleep "$CHECK_INTERVAL_SECS"
done
