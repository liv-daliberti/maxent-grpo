#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/n/fs/similarity/maxent-grpo
cd "$ROOT_DIR"

e50g="$ROOT_DIR/var/artifacts/e50g_safe_signature_teacher_route_calibration_v1/result.json"

passes() {
  local path="$1"
  local schema="$2"
  [[ -f "$path" ]] && jq -e \
    --arg schema "$schema" \
    '.schema == $schema
     and .pass == true
     and (.selected_source_indices | length) == 10
     and .bidirectionally_executable_count >= 10' \
    "$path" >/dev/null
}

if passes "$e50g" e50g_safe_signature_teacher_route_calibration_v1; then
  variant=e50d
  bash "$ROOT_DIR/ops/math_strategy_calibration/materialize_e50d_teacher_route_math_toy.sh"
else
  echo "E50G did not pass; no matched toy is eligible" >&2
  exit 1
fi

bash "$ROOT_DIR/ops/math_strategy_calibration/launch_e50_route_probe.sh" \
  "$variant" baseline
printf 'prepared %s and submitted its baseline route probe\n' "$variant"
