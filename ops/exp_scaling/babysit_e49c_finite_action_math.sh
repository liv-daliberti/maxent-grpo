#!/usr/bin/env bash
# Durable E49C orchestration: judge -> menus -> matched toy -> gate -> full.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
SERVER_LAUNCHER="$ROOT_DIR/ops/math_strategy_calibration/launch_e49c_qwen72_node105.sh"
ENDPOINT="$ROOT_DIR/var/artifacts/e49c_math_strategy_qwen72_v1/qwen72_endpoint.json"
SERVER_RECORD="$ROOT_DIR/var/artifacts/e49c_math_strategy_qwen72_v1/server_job.json"
MATERIALIZER="$ROOT_DIR/ops/math_strategy_calibration/materialize_e49c_strategy_menu_data.py"
MENU_JOB_LAUNCHER="$ROOT_DIR/ops/math_strategy_calibration/launch_e49c_menu_job.sh"
LAUNCHER="$ROOT_DIR/ops/exp_scaling/launch_e49c_finite_action_math_05b.sh"
ANALYZER="$ROOT_DIR/ops/exp_scaling/analyze_e49c_finite_action_math.py"
AUDIT_JOB_LAUNCHER="$ROOT_DIR/ops/exp_scaling/launch_e49c_eval_audit_job.sh"
TOY_SOURCE="$ROOT_DIR/var/data/e49b_math_strategy_toy"
TOY_DATA="$ROOT_DIR/var/data/e49c_math_strategy_menu_toy"
TOY_EVIDENCE="$ROOT_DIR/var/artifacts/e49c_math_strategy_menu_toy_v1"
FULL_SOURCE="$ROOT_DIR/var/data/math12k_384_math500"
FULL_DATA="$ROOT_DIR/var/data/e49c_math_strategy_menu_full"
FULL_EVIDENCE="$ROOT_DIR/var/artifacts/e49c_math_strategy_menu_full_v1"
TOY_MANIFEST="$ROOT_DIR/var/artifacts/e49c_finite_action_math_toy_05b_v1_comparative_jobs.tsv"
FULL_MANIFEST="$ROOT_DIR/var/artifacts/e49c_finite_action_math_full_05b_v1_comparative_jobs.tsv"
TOY_DECISION="$ROOT_DIR/var/artifacts/e49c_finite_action_math_toy_advancement.json"
FULL_REPORT="$ROOT_DIR/var/artifacts/e49c_finite_action_math_full_report.json"
STATUS="$ROOT_DIR/var/artifacts/e49c_babysitter_status.json"
TOY_MONITOR_PID="$ROOT_DIR/var/artifacts/e49c_toy_monitor.pid"
FULL_MONITOR_PID="$ROOT_DIR/var/artifacts/e49c_full_monitor.pid"
TOY_MENU_JOB="$TOY_EVIDENCE/materialization_job.json"
FULL_MENU_JOB="$FULL_EVIDENCE/materialization_job.json"
TOY_AUDIT="$ROOT_DIR/var/artifacts/e49c_toy_terminal_execution_audit.json"
FULL_AUDIT="$ROOT_DIR/var/artifacts/e49c_full_terminal_execution_audit.json"
TOY_AUDIT_JOB="$ROOT_DIR/var/artifacts/e49c_toy_terminal_execution_audit_job.json"
FULL_AUDIT_JOB="$ROOT_DIR/var/artifacts/e49c_full_terminal_execution_audit_job.json"

cd "$ROOT_DIR"
source "$ROOT_DIR/ops/repo_env.sh"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ROOT_DIR/var/seed_paper_eval/paper310/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# The site controller hostname is intermittently absent from compute DNS.
# Prefer the already validated direct-controller config when the orchestrator
# has one, so a completed job can always be archived and the next stage can
# start without manual intervention.
if [[ -f /tmp/e49c_slurm.conf ]]; then
  export SLURM_CONF=/tmp/e49c_slurm.conf
fi

status() {
  local phase="$1"
  local detail="${2:-}"
  "$PYTHON_BIN" - "$STATUS" "$phase" "$detail" <<'PY'
import datetime
import json
import os
import pathlib
import sys
import tempfile
path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "e49c_babysitter_status_v1",
    "phase": sys.argv[2],
    "detail": sys.argv[3],
    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY
}

start_monitor() {
  local stage="$1"
  local pid_file="$2"
  if [[ -f "$pid_file" ]] \
    && kill -0 "$(<"$pid_file")" 2>/dev/null; then
    return
  fi
  bash "$ROOT_DIR/ops/exp_scaling/monitor_e49c_finite_action_math.sh" \
    "$stage" >/dev/null 2>&1 &
  local monitor_pid=$!
  printf '%s\n' "$monitor_pid" >"$pid_file"
}

stop_monitor() {
  local pid_file="$1"
  if [[ -f "$pid_file" ]] \
    && kill -0 "$(<"$pid_file")" 2>/dev/null; then
    kill "$(<"$pid_file")" 2>/dev/null || true
  fi
}

endpoint_healthy() {
  # The server writes this only after a localhost health check and removes it
  # on exit. spin cannot route to the compute-node private fabric.
  [[ -f "$ENDPOINT" ]]
}

archive_finished_record() {
  local record="$1"
  [[ -f "$record" ]] || return 0
  local job_id
  job_id="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["job_id"])' "$record")"
  local state
  if ! state="$(
    timeout 20 squeue -h -j "$job_id" -o '%T' 2>/dev/null
  )"; then
    return 1
  fi
  if [[ -n "$state" ]]; then
    return 1
  fi
  mv "$record" "${record%.json}.finished-${job_id}.json"
  return 0
}

while ! endpoint_healthy; do
  status waiting_for_qwen72 "Slurm/server unavailable; retrying safely"
  if [[ ! -f "$SERVER_RECORD" ]]; then
    "$SERVER_LAUNCHER" || true
  else
    archive_finished_record "$SERVER_RECORD" || true
  fi
  sleep 60
done

status generating_toy_menus "100 rows; two independent audits each"
while [[ ! -f "$TOY_DATA/MATERIALIZATION_MANIFEST.json" ]]; do
  if ! endpoint_healthy; then
    if [[ ! -f "$SERVER_RECORD" ]]; then
      "$SERVER_LAUNCHER" || true
    else
      archive_finished_record "$SERVER_RECORD" || true
    fi
  elif [[ ! -f "$TOY_MENU_JOB" ]]; then
    "$MENU_JOB_LAUNCHER" toy || true
  else
    archive_finished_record "$TOY_MENU_JOB" || true
  fi
  status generating_toy_menus "node302 CPU job; durable records preserved"
  sleep 60
done

status auditing_toy_prompts "verifying every rendered prompt fits 2048 tokens"
"$PYTHON_BIN" "$ROOT_DIR/ops/exp_scaling/audit_e49c_prompt_lengths.py" \
  --data "$TOY_DATA" \
  --output "$ROOT_DIR/var/artifacts/e49c_toy_prompt_length_audit.json" \
  --max-prompt-tokens 2048 >/dev/null

status launching_toy "matched gated Dr.GRPO and E46 Haarnoja"
if [[ ! -f "$TOY_MANIFEST" ]]; then
  "$LAUNCHER" config
  "$LAUNCHER" toy
fi
start_monitor toy "$TOY_MONITOR_PID"

status running_toy "waiting for both 3-epoch jobs"
while true; do
  if "$PYTHON_BIN" "$ANALYZER" --stage toy --output "$TOY_DECISION"; then
    break
  fi
  if [[ -f "$TOY_DECISION" ]] \
    && [[ "$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("terminal_failure",False))' "$TOY_DECISION")" == True ]]; then
    status toy_failed "full stage blocked; see advancement report"
    exit 2
  fi
  if ! endpoint_healthy; then
    archive_finished_record "$SERVER_RECORD" || true
    if [[ ! -f "$SERVER_RECORD" ]]; then
      "$SERVER_LAUNCHER" || true
    fi
  fi
  toy_status="$(
    "$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "$TOY_DECISION"
  )"
  if [[ "$toy_status" == waiting_for_terminal_execution_audit ]] \
    && [[ ! -f "$TOY_AUDIT" ]]; then
    if [[ ! -f "$TOY_AUDIT_JOB" ]]; then
      "$AUDIT_JOB_LAUNCHER" toy || true
    else
      archive_finished_record "$TOY_AUDIT_JOB" || true
    fi
  fi
  sleep 60
done

advance="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("advance_to_full",False))' "$TOY_DECISION")"
if [[ "$advance" != True ]]; then
  status toy_failed "full stage blocked; see advancement report"
  exit 2
fi

status generating_full_menus "884 exact OAT train/eval rows"
while [[ ! -f "$FULL_DATA/MATERIALIZATION_MANIFEST.json" ]]; do
  if ! endpoint_healthy; then
    if [[ ! -f "$SERVER_RECORD" ]]; then
      "$SERVER_LAUNCHER" || true
    else
      archive_finished_record "$SERVER_RECORD" || true
    fi
  elif [[ ! -f "$FULL_MENU_JOB" ]]; then
    "$MENU_JOB_LAUNCHER" full || true
  else
    archive_finished_record "$FULL_MENU_JOB" || true
  fi
  status generating_full_menus "node302 CPU job; durable records preserved"
  sleep 60
done

status auditing_full_prompts "verifying every rendered prompt fits 2048 tokens"
"$PYTHON_BIN" "$ROOT_DIR/ops/exp_scaling/audit_e49c_prompt_lengths.py" \
  --data "$FULL_DATA" \
  --output "$ROOT_DIR/var/artifacts/e49c_full_prompt_length_audit.json" \
  --max-prompt-tokens 2048 >/dev/null

status launching_full "384-train/500-eval, three epochs, matched arms"
if [[ ! -f "$FULL_MANIFEST" ]]; then
  "$LAUNCHER" full
fi
stop_monitor "$TOY_MONITOR_PID"
start_monitor full "$FULL_MONITOR_PID"

status running_full "waiting for final matched report"
while ! "$PYTHON_BIN" "$ANALYZER" --stage full --output "$FULL_REPORT"; do
  if ! endpoint_healthy; then
    archive_finished_record "$SERVER_RECORD" || true
    if [[ ! -f "$SERVER_RECORD" ]]; then
      "$SERVER_LAUNCHER" || true
    fi
  fi
  full_status="$(
    "$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "$FULL_REPORT"
  )"
  if [[ "$full_status" == waiting_for_terminal_execution_audit ]] \
    && [[ ! -f "$FULL_AUDIT" ]]; then
    if [[ ! -f "$FULL_AUDIT_JOB" ]]; then
      "$AUDIT_JOB_LAUNCHER" full || true
    else
      archive_finished_record "$FULL_AUDIT_JOB" || true
    fi
  fi
  sleep 60
done
status complete "E49C full-stage report passed"
