#!/usr/bin/env bash
# Move only pending E66 non-Math controls to broader same-family pvl pools.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AMENDMENT="$ROOT_DIR/paper/preregistration/e66_same_family_resume_placement_amendment_20260727.md"
IDENTITY="$ROOT_DIR/var/artifacts/e66_same_plumbing_actuator_ablation_identity.json"
AUDIT="$ROOT_DIR/var/artifacts/e66_same_plumbing_actuator_ablation_audit_latest.json"
OUT="$ROOT_DIR/var/artifacts/e66_same_family_resume_placement_amendment.json"
GRAPH_IDS=(30128394 30128395 30128396)
RTX_IDS=(30128397 30128398 30128399 30128400 30128401 30128402)
ALL_IDS=("${GRAPH_IDS[@]}" "${RTX_IDS[@]}")
GRAPH_ORIGINAL=node103
RTX_ORIGINAL=node024
GRAPH_NODES=node103,node104
RTX_NODES=node020,node021,node022,node024,node026

for required in "$AMENDMENT" "$IDENTITY" "$AUDIT"; do
  [[ -f "$required" ]] || {
    echo "Missing E66 amendment prerequisite: $required" >&2
    exit 1
  }
done
[[ ! -e "$OUT" ]] || {
  echo "Fresh E66 resume amendment artifact required: $OUT" >&2
  exit 1
}

released=0
rollback() {
  local status="$?"
  trap - EXIT
  if [[ "$released" != 1 ]]; then
    for job_id in "${GRAPH_IDS[@]}"; do
      scontrol update JobId="$job_id" Account=allcs Partition=lowprio \
        NodeList="$GRAPH_ORIGINAL" 2>/dev/null || true
    done
    for job_id in "${RTX_IDS[@]}"; do
      scontrol update JobId="$job_id" Account=allcs Partition=lowprio \
        NodeList="$RTX_ORIGINAL" 2>/dev/null || true
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
    'JobState=PENDING' 'Reason=JobHeldUser' 'Account=allcs' \
    'OAT_ZERO_VARIANT=verified_first_global_replay_canonical' \
    'OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.50' \
    'OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS=0'; do
    [[ "$record" == *"$required"* ]] || {
      echo "E66 resume hold audit failed for $job_id: $required" >&2
      exit 1
    }
  done
done

for job_id in "${GRAPH_IDS[@]}"; do
  scontrol update JobId="$job_id" Account=mltheory \
    Partition=pvl-lowprio NodeList="$GRAPH_NODES"
done
for job_id in "${RTX_IDS[@]}"; do
  scontrol update JobId="$job_id" Account=mltheory \
    Partition=pvl-lowprio NodeList="$RTX_NODES"
done

for job_id in "${GRAPH_IDS[@]}"; do
  record="$(scontrol show job "$job_id" -o)"
  [[ "$record" == *'Partition=pvl-lowprio'* ]] || exit 1
  [[ "$record" == *'Account=mltheory'* ]] || exit 1
  [[ "$record" == *'TresPerNode=gres/gpu:a6000:1'* ]] || exit 1
done
for job_id in "${RTX_IDS[@]}"; do
  record="$(scontrol show job "$job_id" -o)"
  [[ "$record" == *'Partition=pvl-lowprio'* ]] || exit 1
  [[ "$record" == *'Account=mltheory'* ]] || exit 1
  [[ "$record" == *'TresPerNode=gres/gpu:rtx_3090:1'* ]] || exit 1
done

python - "$OUT" "$AMENDMENT" "$0" "$AUDIT" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


out = Path(sys.argv[1])
audit = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))
pre_amendment_steps = {}
for domain in ("graph_coloring", "countdown", "python_factor"):
    for run in audit["domains"][domain]["runs"]:
        pre_amendment_steps[str(run["job_id"])] = run["latest_step"]

payload = {
    "schema": "e66_same_family_resume_placement_amendment_v1",
    "amendment_sha256": digest(sys.argv[2]),
    "script_sha256": digest(sys.argv[3]),
    "affected_jobs": {
        "graph_coloring": [30128394, 30128395, 30128396],
        "countdown": [30128397, 30128398, 30128399],
        "python_factor": [30128400, 30128401, 30128402],
    },
    "unaffected_running_jobs": [30128403, 30128404, 30128405],
    "precondition": "all affected jobs held and pending",
    "pre_amendment_latest_steps": pre_amendment_steps,
    "mutation": {
        "graph_coloring": {
            "account": "mltheory",
            "partition": "pvl-lowprio",
            "nodes": ["node103", "node104"],
            "gres": "gpu:a6000:1",
        },
        "countdown_python": {
            "account": "mltheory",
            "partition": "pvl-lowprio",
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
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, out)
PY

for job_id in "${ALL_IDS[@]}"; do
  scontrol release "$job_id"
done
released=1
trap - EXIT
echo "[e66-resume-placement] released amended jobs: ${ALL_IDS[*]}"
