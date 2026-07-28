#!/usr/bin/env bash
# Atomically move only zero-step pending E67 jobs to compatible GPU capacity.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AMENDMENT="$ROOT_DIR/paper/preregistration/e67_zero_step_placement_amendment_20260727.md"
IDENTITY="$ROOT_DIR/var/artifacts/e67_corrected_same_objective_actuator_ablation_identity.json"
OUT="$ROOT_DIR/var/artifacts/e67_zero_step_placement_amendment.json"
GRAPH_IDS=(30128500 30128501 30128502)
RTX_IDS=(30128503 30128504 30128505 30128506 30128507 30128508)
ALL_IDS=("${GRAPH_IDS[@]}" "${RTX_IDS[@]}")
GRAPH_NODES=node104,node205,node206,node207,node805
RTX_NODES=node020,node021,node022,node024,node026

for required in "$AMENDMENT" "$IDENTITY"; do
  [[ -f "$required" ]] || {
    echo "Missing E67 amendment prerequisite: $required" >&2
    exit 1
  }
done
[[ ! -e "$OUT" ]] || {
  echo "Fresh E67 placement amendment artifact required: $OUT" >&2
  exit 1
}

released=0
rollback() {
  local status="$?"
  trap - EXIT
  if [[ "$released" != 1 ]]; then
    for job_id in "${GRAPH_IDS[@]}"; do
      scontrol update \
        JobId="$job_id" Account=allcs NodeList=node103 2>/dev/null || true
    done
    for job_id in "${RTX_IDS[@]}"; do
      scontrol update \
        JobId="$job_id" Account=allcs NodeList=node022,node026 \
        2>/dev/null || true
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
  for required in \
    'JobState=PENDING' 'Reason=JobHeldUser' \
    'OAT_ZERO_VARIANT=verified_entropy_gated_singleton_escape_canonical' \
    'OAT_ZERO_EXPECT_ONLINE_CANONICAL_NOVELTY_BETA=0.50'; do
    [[ "$record" == *"$required"* ]] || {
      echo "E67 zero-step hold audit failed for $job_id: $required" >&2
      exit 1
    }
  done
  if find "$ROOT_DIR/var/data" -path "*/debug_job${job_id}" -print -quit \
    | grep -q .; then
    echo "E67 job $job_id already materialized; refusing placement change" >&2
    exit 1
  fi
done

for job_id in "${GRAPH_IDS[@]}"; do
  scontrol update \
    JobId="$job_id" Account=mltheory NodeList="$GRAPH_NODES"
done
for job_id in "${RTX_IDS[@]}"; do
  scontrol update \
    JobId="$job_id" Account=mltheory NodeList="$RTX_NODES"
done

for job_id in "${GRAPH_IDS[@]}"; do
  record="$(scontrol show job "$job_id" -o)"
  [[ "$record" == *'Account=mltheory'* ]] || exit 1
  [[ "$record" == *'TresPerNode=gres/gpu:a6000:1'* ]] || exit 1
done
for job_id in "${RTX_IDS[@]}"; do
  record="$(scontrol show job "$job_id" -o)"
  [[ "$record" == *'Account=mltheory'* ]] || exit 1
  [[ "$record" == *'TresPerNode=gres/gpu:rtx_3090:1'* ]] || exit 1
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
    "schema": "e67_zero_step_placement_amendment_v1",
    "amendment_sha256": digest(sys.argv[2]),
    "script_sha256": digest(sys.argv[3]),
    "affected_jobs": {
        "graph_coloring": [30128500, 30128501, 30128502],
        "countdown": [30128503, 30128504, 30128505],
        "python_factor": [30128506, 30128507, 30128508],
    },
    "unaffected_running_jobs": [30128509, 30128510, 30128511],
    "precondition": "all affected jobs held, pending, and unmaterialized",
    "mutation": {
        "graph_coloring": {
            "account": "mltheory",
            "nodes": ["node104", "node205", "node206", "node207", "node805"],
            "gres": "gpu:a6000:1",
        },
        "countdown_python": {
            "account": "mltheory",
            "nodes": ["node020", "node021", "node022", "node024", "node026"],
            "gres": "gpu:rtx_3090:1",
        },
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
echo "[e67-placement] released amended zero-step jobs: ${ALL_IDS[*]}"
