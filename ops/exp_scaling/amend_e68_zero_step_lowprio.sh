#!/usr/bin/env bash
# Move only pending, unmaterialized E68 jobs to earlier same-family capacity.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AMENDMENT="$ROOT_DIR/paper/preregistration/e68_zero_step_lowprio_amendment_20260727.md"
IDENTITY="$ROOT_DIR/var/artifacts/e68_separated_support_actuator_ablation_identity.json"
OUT="$ROOT_DIR/var/artifacts/e68_zero_step_lowprio_amendment.json"
GRAPH_IDS=(30130469 30130470 30130471)
RTX_IDS=(30130472 30130473 30130474 30130475 30130476 30130477)
ALL_IDS=("${GRAPH_IDS[@]}" "${RTX_IDS[@]}")
GRAPH_ORIGINAL=node103,node104
GRAPH_NODES=node103,node104,node205,node206,node207,node805
RTX_NODES=node020,node021,node022,node024,node026

for required in "$AMENDMENT" "$IDENTITY"; do
  [[ -f "$required" ]] || {
    echo "Missing E68 amendment prerequisite: $required" >&2
    exit 1
  }
done
[[ ! -e "$OUT" ]] || {
  echo "Fresh E68 lowprio amendment artifact required: $OUT" >&2
  exit 1
}

released=0
rollback() {
  local status="$?"
  trap - EXIT
  if [[ "$released" != 1 ]]; then
    for job_id in "${GRAPH_IDS[@]}"; do
      scontrol update JobId="$job_id" Partition=pvl-lowprio \
        NodeList="$GRAPH_ORIGINAL" 2>/dev/null || true
    done
    for job_id in "${RTX_IDS[@]}"; do
      scontrol update JobId="$job_id" Partition=pvl-lowprio \
        NodeList="$RTX_NODES" 2>/dev/null || true
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
    'JobState=PENDING' 'Reason=JobHeldUser' 'Account=mltheory' \
    'OAT_ZERO_VARIANT=verified_entropy_gated_singleton_escape_canonical' \
    'OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SEPARATE_OBJECTIVE_SUPPORT=1' \
    'OAT_ZERO_EXPECT_ONLINE_CANONICAL_NOVELTY_BETA=0.50'; do
    [[ "$record" == *"$required"* ]] || {
      echo "E68 zero-step hold audit failed for $job_id: $required" >&2
      exit 1
    }
  done
  if find "$ROOT_DIR/var/data" -path "*/debug_job${job_id}" -print -quit \
    | grep -q .; then
    echo "E68 job $job_id already materialized; refusing placement change" >&2
    exit 1
  fi
done

for job_id in "${GRAPH_IDS[@]}"; do
  scontrol update JobId="$job_id" Partition=lowprio \
    NodeList="$GRAPH_NODES"
done
for job_id in "${RTX_IDS[@]}"; do
  scontrol update JobId="$job_id" Partition=lowprio NodeList="$RTX_NODES"
done

for job_id in "${GRAPH_IDS[@]}"; do
  record="$(scontrol show job "$job_id" -o)"
  [[ "$record" == *'Partition=lowprio'* ]] || exit 1
  [[ "$record" == *'Account=mltheory'* ]] || exit 1
  [[ "$record" == *'TresPerNode=gres/gpu:a6000:1'* ]] || exit 1
done
for job_id in "${RTX_IDS[@]}"; do
  record="$(scontrol show job "$job_id" -o)"
  [[ "$record" == *'Partition=lowprio'* ]] || exit 1
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
    "schema": "e68_zero_step_lowprio_amendment_v1",
    "amendment_sha256": digest(sys.argv[2]),
    "script_sha256": digest(sys.argv[3]),
    "affected_jobs": {
        "graph_coloring": [30130469, 30130470, 30130471],
        "countdown": [30130472, 30130473, 30130474],
        "python_factor": [30130475, 30130476, 30130477],
    },
    "unaffected_running_jobs": [30130478, 30130479, 30130480],
    "precondition": "all affected jobs held, pending, and unmaterialized",
    "mutation": {
        "graph_coloring": {
            "account": "mltheory",
            "partition": "lowprio",
            "nodes": [
                "node103", "node104", "node205",
                "node206", "node207", "node805",
            ],
            "gres": "gpu:a6000:1",
        },
        "countdown_python": {
            "account": "mltheory",
            "partition": "lowprio",
            "nodes": [
                "node020", "node021", "node022", "node024", "node026",
            ],
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
echo "[e68-placement] released amended zero-step jobs: ${ALL_IDS[*]}"
