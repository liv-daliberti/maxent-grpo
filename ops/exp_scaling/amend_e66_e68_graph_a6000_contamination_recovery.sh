#!/usr/bin/env bash
# Recover paired E66/E68 Graph jobs from a foreign-process-contaminated A6000.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AMENDMENT="$ROOT_DIR/paper/preregistration/e66_e68_graph_a6000_contamination_recovery_amendment_20260727.md"
SCRIPT_PATH="$ROOT_DIR/ops/exp_scaling/amend_e66_e68_graph_a6000_contamination_recovery.sh"
E66_AUDIT="$ROOT_DIR/var/artifacts/e66_same_plumbing_actuator_ablation_audit_latest.json"
E68_AUDIT="$ROOT_DIR/var/artifacts/e68_separated_support_actuator_ablation_audit_latest.json"
OUT="$ROOT_DIR/var/artifacts/e66_e68_graph_a6000_contamination_recovery_amendment.json"
E66_IDS=(30128394 30128395 30128396)
E68_IDS=(30130469 30130470 30130471)
ALL_IDS=("${E66_IDS[@]}" "${E68_IDS[@]}")
OLD_NODES=node205,node206,node207
NEW_NODES=node103,node104,node805
CONTAMINATED_IDS=(30130469 30130470)

for required in "$AMENDMENT" "$SCRIPT_PATH" "$E66_AUDIT" "$E68_AUDIT"; do
  [[ -f "$required" ]] || {
    echo "Missing Graph contamination-recovery prerequisite: $required" >&2
    exit 1
  }
done
[[ ! -e "$OUT" ]] || {
  echo "Fresh Graph contamination-recovery artifact required: $OUT" >&2
  exit 1
}

released=0
rollback() {
  local status="$?"
  trap - EXIT
  if [[ "$released" != 1 ]]; then
    for job_id in "${ALL_IDS[@]}"; do
      scontrol update JobId="$job_id" Account=allcs Partition=lowprio \
        NodeList="$OLD_NODES" 2>/dev/null || true
      if [[ " ${CONTAMINATED_IDS[*]} " == *" $job_id "* ]]; then
        scontrol hold "$job_id" 2>/dev/null || true
      else
        scontrol release "$job_id" 2>/dev/null || true
      fi
    done
  fi
  exit "$status"
}
trap rollback EXIT

for job_id in "${CONTAMINATED_IDS[@]}"; do
  contaminated_record="$(scontrol show job "$job_id" -o)"
  if [[ "$contaminated_record" == *'JobState=RUNNING'* ]] \
    || [[ "$contaminated_record" == *'JobState=COMPLETING'* ]]; then
    scontrol requeuehold "$job_id"
  else
    scontrol hold "$job_id"
  fi
done
for job_id in "${ALL_IDS[@]}"; do
  if [[ " ${CONTAMINATED_IDS[*]} " != *" $job_id "* ]]; then
    scontrol hold "$job_id"
  fi
done

for attempt in $(seq 1 180); do
  all_held=1
  for job_id in "${ALL_IDS[@]}"; do
    record="$(scontrol show job "$job_id" -o)"
    if [[ "$record" != *'JobState=PENDING'* ]] \
      || { [[ "$record" != *'Reason=JobHeldUser'* ]] \
        && [[ "$record" != *'Reason=job_requeued_in_held_state'* ]]; }; then
      all_held=0
      break
    fi
  done
  [[ "$all_held" == 1 ]] && break
  sleep 1
done

for job_id in "${ALL_IDS[@]}"; do
  record="$(scontrol show job "$job_id" -o)"
  for required in \
    'JobState=PENDING' \
    'Account=allcs' \
    'Partition=lowprio' \
    'TresPerNode=gres/gpu:a6000:1'; do
    [[ "$record" == *"$required"* ]] || {
      echo "Graph contamination-recovery hold audit failed for $job_id: $required" >&2
      exit 1
    }
  done
  if [[ "$record" != *'Reason=JobHeldUser'* ]] \
    && [[ "$record" != *'Reason=job_requeued_in_held_state'* ]]; then
    echo "Graph contamination-recovery hold audit failed for $job_id: held reason" >&2
    exit 1
  fi
done

python - "$E66_AUDIT" "$E68_AUDIT" "${CONTAMINATED_IDS[@]}" <<'PY'
import json
from pathlib import Path
import sys

contaminated = {int(value) for value in sys.argv[3:]}
for audit_path in sys.argv[1:3]:
    payload = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    if payload.get("status") == "fail":
        allowed = {
            "CUDA out of memory",
            "[rank0]: Traceback (most recent call last)",
        }
        for violation in payload.get("violations", []):
            if not any(str(job_id) in violation for job_id in contaminated) or not any(
                marker in violation for marker in allowed
            ):
                raise SystemExit(
                    f"unrelated pre-amendment audit violation: {violation}"
                )

e68 = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
for run in e68["domains"]["graph_coloring"]["runs"]:
    if int(run["job_id"]) in contaminated:
        if int(run["latest_step"]) != -1 or int(run["metric_records"]) != 0:
            raise SystemExit("contaminated E68 job is not pre-optimizer")
PY

for job_id in "${ALL_IDS[@]}"; do
  scontrol update JobId="$job_id" Account=mltheory \
    Partition=pvl-lowprio NodeList="$NEW_NODES"
done

for job_id in "${ALL_IDS[@]}"; do
  record="$(scontrol show job "$job_id" -o)"
  for required in \
    'JobState=PENDING' \
    'Account=mltheory' \
    'Partition=pvl-lowprio' \
    'TresPerNode=gres/gpu:a6000:1'; do
    [[ "$record" == *"$required"* ]] || {
      echo "Graph contamination-recovery mutation audit failed for $job_id: $required" >&2
      exit 1
    }
  done
  if [[ "$record" != *'Reason=JobHeldUser'* ]] \
    && [[ "$record" != *'Reason=job_requeued_in_held_state'* ]]; then
    echo "Graph contamination-recovery mutation audit failed for $job_id: held reason" >&2
    exit 1
  fi
done

python - "$OUT" "$AMENDMENT" "$SCRIPT_PATH" "$E66_AUDIT" "$E68_AUDIT" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


root = Path(sys.argv[1]).resolve().parents[2]
out = Path(sys.argv[1])
e66 = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))
e68 = json.loads(Path(sys.argv[5]).read_text(encoding="utf-8"))
affected = {
    30128394, 30128395, 30128396,
    30130469, 30130470, 30130471,
}
steps = {}
for payload in (e66, e68):
    for run in payload["domains"]["graph_coloring"]["runs"]:
        job_id = int(run["job_id"])
        if job_id in affected:
            steps[str(job_id)] = int(run["latest_step"])
if set(map(int, steps)) != affected:
    raise SystemExit("could not bind all paired Graph jobs")
for job_id in ("30130469", "30130470"):
    if steps[job_id] != -1:
        raise SystemExit("registered contaminated jobs must remain pre-optimizer")

log_prefix_bytes = {}
for job_id in (30130469, 30130470):
    log_prefix_bytes[str(job_id)] = {}
    for suffix in ("out", "err"):
        path = root / f"var/artifacts/logs/xdr_train-{job_id}.{suffix}"
        log_prefix_bytes[str(job_id)][suffix] = (
            path.stat().st_size if path.is_file() else 0
        )

payload = {
    "schema": "e66_e68_graph_a6000_contamination_recovery_amendment_v1",
    "amendment_sha256": digest(sys.argv[2]),
    "script_sha256": digest(sys.argv[3]),
    "affected_jobs": {
        "e66_graph_coloring": [30128394, 30128395, 30128396],
        "e68_graph_coloring": [30130469, 30130470, 30130471],
    },
    "paired_arms_moved_together": True,
    "pre_amendment_latest_steps": steps,
    "registered_contamination": {
        "job_ids": [30130469, 30130470],
        "node": "node206",
        "slurm_physical_gpu_index": 6,
        "observed_memory_used_mib": 48323,
        "foreign_process_users": ["mi9937", "rj5498"],
        "optimizer_metric_records": {
            "30130469": 0,
            "30130470": 0,
        },
        "log_prefix_bytes": log_prefix_bytes,
    },
    "mutation": {
        "account": "mltheory",
        "partition": "pvl-lowprio",
        "nodes": ["node103", "node104", "node805"],
        "gres": "gpu:a6000:1",
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
echo "[graph-a6000-contamination-recovery] released paired jobs: ${ALL_IDS[*]}"
