#!/usr/bin/env bash
# Keep the current analytical campaign moving during an unattended window.
set -uo pipefail

ROOT_DIR="${OAT_ZERO_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$ROOT_DIR" || ! -f "$ROOT_DIR/ops/exp_scaling/monitor_campaign.py" ]]; then
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
LOG_DIR="$ROOT_DIR/var/artifacts/logs"
EVENT_LOG="$LOG_DIR/overnight_campaign_guard_events.log"
SNAPSHOT_LOG="$LOG_DIR/overnight_campaign_guard_snapshots.log"
FIGURE_LOG="$LOG_DIR/overnight_campaign_guard_figures.log"
INTERVAL_SECONDS="${OAT_ZERO_GUARD_INTERVAL_SECONDS:-300}"
STALE_SECONDS="${OAT_ZERO_GUARD_STALE_SECONDS:-5400}"
MAX_RECOVERIES="${OAT_ZERO_GUARD_MAX_RECOVERIES:-4}"

mkdir -p "$LOG_DIR"

# Current analytical IDs only. Historical failed/held namespaces are excluded.
JOB_IDS=(
  # E21 MATH-500 free-form cohort.
  30020395 30020396 30020397 30020398 30020399 30020400
  30020401 30020402 30020403 30020404 30020405 30020406
  # E22-v2 free-form graph-coloring and Countdown cohorts.
  30031160 30031161 30031162 30031163 30031164 30031165
  30031166 30031167 30031168 30031169 30031170 30031171
  # Remaining E17/E18 3B Countdown cells.
  30020611 30024458 30031645
  # E23 Countdown 7B and E24 graph-coloring 7B cohorts.
  30031175 30031176 30031177 30031178 30031179 30031180
  30031181 30031182 30031183
  30031621 30031622 30031623 30031624 30031625 30031626
  30031627 30031628 30031629
)

declare -A recovery_count=()

log_event() {
  printf '%s %s\n' "$(date -Iseconds)" "$*" >> "$EVENT_LOG"
}

recover_job() {
  local job_id="$1" cause="$2"
  local count="${recovery_count[$job_id]:-0}"
  if (( count >= MAX_RECOVERIES )); then
    log_event "job=$job_id action=blocked cause=$cause recoveries=$count"
    return
  fi
  if scontrol requeue "$job_id" >> "$EVENT_LOG" 2>&1; then
    recovery_count[$job_id]=$((count + 1))
    log_event "job=$job_id action=requeue cause=$cause recoveries=$((count + 1))"
  else
    log_event "job=$job_id action=requeue_failed cause=$cause recoveries=$count"
  fi
}

log_event "guard=start interval_seconds=$INTERVAL_SECONDS stale_seconds=$STALE_SECONDS"

while true; do
  now_epoch="$(date +%s)"
  active_count=0

  python3 "$ROOT_DIR/ops/exp_scaling/monitor_campaign.py" --once --no-clear \
    >> "$SNAPSHOT_LOG" 2>&1 || log_event "action=monitor_failed"
  python3 "$ROOT_DIR/ops/exp_scaling/refresh_campaign_figures.py" \
    >> "$FIGURE_LOG" 2>&1 || log_event "action=figure_refresh_failed"

  for job_id in "${JOB_IDS[@]}"; do
    job_record="$(scontrol show job -o "$job_id" 2>/dev/null || true)"
    [[ -n "$job_record" ]] || continue
    state="$(sed -n 's/.* JobState=\([^ ]*\).*/\1/p' <<< "$job_record")"

    case "$state" in
      PENDING|RUNNING|CONFIGURING|COMPLETING|SUSPENDED|REQUEUED|RESIZING)
        active_count=$((active_count + 1))
        ;;
    esac

    case "$state" in
      FAILED|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL|TIMEOUT)
        recover_job "$job_id" "terminal_$state"
        continue
        ;;
      RUNNING)
        stdout_path="$(sed -n 's/.* StdOut=\([^ ]*\).*/\1/p' <<< "$job_record")"
        if [[ -n "$stdout_path" && -f "$stdout_path" ]]; then
          modified_epoch="$(stat -c %Y "$stdout_path" 2>/dev/null || printf '0')"
          age_seconds=$((now_epoch - modified_epoch))
          if (( age_seconds > STALE_SECONDS )); then
            recover_job "$job_id" "stale_log_${age_seconds}s"
          fi
        fi
        ;;
    esac
  done

  if (( active_count == 0 )); then
    log_event "guard=complete active_jobs=0"
    python3 "$ROOT_DIR/ops/exp_scaling/refresh_campaign_figures.py" \
      >> "$FIGURE_LOG" 2>&1 || true
    exit 0
  fi

  log_event "guard=heartbeat active_jobs=$active_count"
  sleep "$INTERVAL_SECONDS"
done
