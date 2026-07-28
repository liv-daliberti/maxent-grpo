#!/usr/bin/env bash
# Move zero-optimizer-step E67 jobs to immediately available advertised partitions.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AMENDMENT="$ROOT_DIR/paper/preregistration/e67_zero_step_partition_amendment_20260727.md"
OUT="$ROOT_DIR/var/artifacts/e67_zero_step_partition_amendment.json"
GRAPH_IDS=(30128500 30128501 30128502)
RTX_IDS=(30128503 30128504 30128505 30128506 30128507 30128508)
MATH_IDS=(30128509 30128510 30128511)
ALL_IDS=("${GRAPH_IDS[@]}" "${RTX_IDS[@]}" "${MATH_IDS[@]}")

[[ -f "$AMENDMENT" ]] || exit 1
[[ ! -e "$OUT" ]] || {
  echo "Fresh E67 partition amendment artifact required: $OUT" >&2
  exit 1
}

released=0
rollback() {
  local status="$?"
  trap - EXIT
  if [[ "$released" != 1 ]]; then
    for job_id in "${GRAPH_IDS[@]}"; do
      scontrol update JobId="$job_id" Partition=lowprio \
        NodeList=node104,node205,node206,node207,node805 2>/dev/null || true
    done
    for job_id in "${RTX_IDS[@]}"; do
      scontrol update JobId="$job_id" Partition=lowprio \
        NodeList=node020,node021,node022,node024,node026 2>/dev/null || true
    done
    for job_id in "${MATH_IDS[@]}"; do
      scontrol update JobId="$job_id" Partition=lowprio \
        NodeList=node302 2>/dev/null || true
    done
    for job_id in "${ALL_IDS[@]}"; do
      scontrol release "$job_id" 2>/dev/null || true
    done
  fi
  exit "$status"
}
trap rollback EXIT

for job_id in "${ALL_IDS[@]}"; do
  scontrol hold "$job_id"
done
for job_id in "${ALL_IDS[@]}"; do
  record="$(scontrol show job "$job_id" -o)"
  [[ "$record" == *'JobState=PENDING'* ]] || exit 1
  [[ "$record" == *'Reason=JobHeldUser'* ]] || exit 1
  [[ "$record" == *'OAT_ZERO_EXPECT_ONLINE_CANONICAL_NOVELTY_BETA=0.50'* ]] \
    || exit 1
  if find "$ROOT_DIR/var/data" -path "*/debug_job${job_id}/train_metrics.jsonl" \
    -print -quit | grep -q .; then
    echo "E67 job $job_id has optimizer metrics; refusing partition change" >&2
    exit 1
  fi
done

for job_id in "${GRAPH_IDS[@]}"; do
  scontrol update JobId="$job_id" Partition=pvl-lowprio \
    NodeList=node103,node104
done
for job_id in "${RTX_IDS[@]}"; do
  scontrol update JobId="$job_id" Partition=pvl-lowprio \
    NodeList=node020,node021,node022,node024,node026
done
for job_id in "${MATH_IDS[@]}"; do
  scontrol update JobId="$job_id" Partition=mltheory NodeList=node302
done

for job_id in "${GRAPH_IDS[@]}" "${RTX_IDS[@]}"; do
  record="$(scontrol show job "$job_id" -o)"
  [[ "$record" == *'Partition=pvl-lowprio'* ]] || exit 1
done
for job_id in "${MATH_IDS[@]}"; do
  record="$(scontrol show job "$job_id" -o)"
  [[ "$record" == *'Partition=mltheory'* ]] || exit 1
done

python - "$OUT" "$AMENDMENT" "$0" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

out = Path(sys.argv[1])
payload = {
    "schema": "e67_zero_step_partition_amendment_v1",
    "amendment_sha256": digest(sys.argv[2]),
    "script_sha256": digest(sys.argv[3]),
    "precondition": "all jobs held, pending, and without optimizer metrics",
    "jobs": {
        "graph_coloring": [30128500, 30128501, 30128502],
        "countdown_python": [
            30128503, 30128504, 30128505,
            30128506, 30128507, 30128508,
        ],
        "mathir": [30128509, 30128510, 30128511],
    },
    "partitions": {
        "graph_coloring": "pvl-lowprio",
        "countdown_python": "pvl-lowprio",
        "mathir": "mltheory",
    },
    "scientific_settings_changed": False,
}
out.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=f".{out.name}.", dir=out.parent)
with os.fdopen(fd, "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, out)
PY

for job_id in "${ALL_IDS[@]}"; do
  scontrol release "$job_id"
done
released=1
trap - EXIT
echo "[e67-partition] released amended jobs: ${ALL_IDS[*]}"
