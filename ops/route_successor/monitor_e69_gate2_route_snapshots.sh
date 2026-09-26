#!/usr/bin/env bash
# Capture every E69 Gate 2 route checkpoint before two-checkpoint pruning.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

PYTHON_BIN="${OAT_ZERO_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
if [[ -f "$ROOT_DIR/var/artifacts/e69_gate2_r2_route_endpoint_bookkeeping_repair_identity.json" ]]; then
  ARTIFACT="$ROOT_DIR/var/artifacts/e69_gate2_r2_route_temporal_snapshots.json"
elif [[ -f "$ROOT_DIR/var/artifacts/e69_gate2_r1_execution_repair_identity.json" ]]; then
  ARTIFACT="$ROOT_DIR/var/artifacts/e69_gate2_r1_route_temporal_snapshots.json"
else
  ARTIFACT="$ROOT_DIR/var/artifacts/e69_gate2_route_temporal_snapshots.json"
fi

cd "$ROOT_DIR"
while true; do
  PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR" \
    "$PYTHON_BIN" ops/route_successor/snapshot_e69_gate2_route_replay.py
  complete="$(
    "$PYTHON_BIN" - "$ARTIFACT" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(int(payload.get("summary", {}).get("observed_snapshots", 0) == 24))
PY
  )"
  if [[ "$complete" == "1" ]]; then
    echo "[e69-route-monitor] all 24 route checkpoints captured"
    exit 0
  fi
  sleep 180
done
