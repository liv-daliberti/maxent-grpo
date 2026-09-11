#!/usr/bin/env bash
# Move newly preempted E61-R1 MathIR A6000 jobs to same-family pvl capacity.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AMENDMENT="$ROOT_DIR/paper/preregistration/e61r1_third_same_family_resume_placement_amendment_20260727.md"
IDENTITY="$ROOT_DIR/var/artifacts/e61r1_e58_vs_grpo_05b_12pass_identity.json"
AUDIT="$ROOT_DIR/var/artifacts/e61r1_e58_vs_grpo_12pass_audit_latest.json"
OUT="$ROOT_DIR/var/artifacts/e61r1_third_same_family_resume_placement_amendment.json"
ALL_IDS=(30126351 30126352 30126353 30126354)
ORIGINAL_NODES=node103,node104,node208
EXPANDED_NODES=node103,node104,node805

declare -A EXPECTED_ARM=(
  [30126351]=grpo
  [30126352]=verified_first_global_replay_canonical
  [30126353]=grpo
  [30126354]=verified_first_global_replay_canonical
)

for required in "$AMENDMENT" "$IDENTITY" "$AUDIT"; do
  [[ -f "$required" ]] || {
    echo "Missing E61-R1 third-amendment prerequisite: $required" >&2
    exit 1
  }
done
[[ ! -e "$OUT" ]] || {
  echo "Fresh E61-R1 third amendment artifact required: $OUT" >&2
  exit 1
}

released=0
rollback() {
  local status="$?"
  trap - EXIT
  if [[ "$released" != 1 ]]; then
    for job_id in "${ALL_IDS[@]}"; do
      scontrol update JobId="$job_id" Account=mltheory Partition=lowprio \
        NodeList="$ORIGINAL_NODES" 2>/dev/null || true
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
    'JobState=PENDING' \
    'Reason=JobHeldUser' \
    'Account=mltheory' \
    'Partition=lowprio' \
    'TresPerNode=gres/gpu:a6000:1' \
    "OAT_ZERO_VARIANT=${EXPECTED_ARM[$job_id]}"; do
    [[ "$record" == *"$required"* ]] || {
      echo "E61-R1 third-amendment hold audit failed for $job_id: $required" >&2
      exit 1
    }
  done
done

for job_id in "${ALL_IDS[@]}"; do
  scontrol update JobId="$job_id" Account=mltheory \
    Partition=pvl-lowprio NodeList="$EXPANDED_NODES"
done

for job_id in "${ALL_IDS[@]}"; do
  record="$(scontrol show job "$job_id" -o)"
  for required in \
    'JobState=PENDING' \
    'Reason=JobHeldUser' \
    'Account=mltheory' \
    'Partition=pvl-lowprio' \
    'TresPerNode=gres/gpu:a6000:1'; do
    [[ "$record" == *"$required"* ]] || {
      echo "E61-R1 third-amendment mutation audit failed for $job_id: $required" >&2
      exit 1
    }
  done
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
affected = {30126351, 30126352, 30126353, 30126354}
pre_amendment_steps = {}
for domain in audit["domains"].values():
    for run in domain["runs"]:
        job_id = int(run["job_id"])
        if job_id in affected:
            pre_amendment_steps[str(job_id)] = int(run["latest_step"])
if set(map(int, pre_amendment_steps)) != affected:
    raise SystemExit("E61-R1 third amendment could not bind every affected job")
if min(pre_amendment_steps.values()) < 0:
    raise SystemExit("E61-R1 third amendment requires materialized checkpoints")

payload = {
    "schema": "e61r1_third_same_family_resume_placement_amendment_v1",
    "amendment_sha256": digest(sys.argv[2]),
    "script_sha256": digest(sys.argv[3]),
    "affected_jobs": {
        "mathir": [30126351, 30126352, 30126353, 30126354],
    },
    "unaffected_job_count": 20,
    "precondition": "all affected jobs held and pending",
    "pre_amendment_latest_steps": pre_amendment_steps,
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
echo "[e61r1-third-resume-placement] released amended jobs: ${ALL_IDS[*]}"
