#!/usr/bin/env bash
# Move paired E66/E68 Graph jobs off the drained A6000 pool.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AMENDMENT="$ROOT_DIR/paper/preregistration/e66_e68_graph_a6000_drain_recovery_amendment_20260727.md"
E66_AUDIT="$ROOT_DIR/var/artifacts/e66_same_plumbing_actuator_ablation_audit_latest.json"
E68_AUDIT="$ROOT_DIR/var/artifacts/e68_separated_support_actuator_ablation_audit_latest.json"
OUT="$ROOT_DIR/var/artifacts/e66_e68_graph_a6000_drain_recovery_amendment.json"
CONTROL_IDS=(30128394 30128395 30128396)
REPAIR_IDS=(30130469 30130470 30130471)
ALL_IDS=("${CONTROL_IDS[@]}" "${REPAIR_IDS[@]}")
ORIGINAL_NODES=node103,node104
TARGET_NODES=node205,node206,node207

for required in "$AMENDMENT" "$E66_AUDIT" "$E68_AUDIT"; do
  [[ -f "$required" ]] || {
    echo "Missing Graph drain-recovery prerequisite: $required" >&2
    exit 1
  }
done
[[ ! -e "$OUT" ]] || {
  echo "Fresh Graph drain-recovery artifact required: $OUT" >&2
  exit 1
}

released=0
rollback() {
  local status="$?"
  trap - EXIT
  if [[ "$released" != 1 ]]; then
    for job_id in "${ALL_IDS[@]}"; do
      scontrol update JobId="$job_id" Account=mltheory \
        Partition=pvl-lowprio NodeList="$ORIGINAL_NODES" 2>/dev/null || true
      scontrol release "$job_id" 2>/dev/null || true
    done
  fi
  exit "$status"
}
trap rollback EXIT

for job_id in "${ALL_IDS[@]}"; do
  scontrol hold "$job_id"
done

for job_id in "${CONTROL_IDS[@]}"; do
  record="$(scontrol show job "$job_id" -o)"
  for required in \
    'JobState=PENDING' 'Reason=JobHeldUser' 'Account=mltheory' \
    'Partition=pvl-lowprio' \
    'OAT_ZERO_VARIANT=verified_first_global_replay_canonical' \
    'OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.50' \
    'TresPerNode=gres/gpu:a6000:1'; do
    [[ "$record" == *"$required"* ]] || {
      echo "E66 Graph hold audit failed for $job_id: $required" >&2
      exit 1
    }
  done
done

for job_id in "${REPAIR_IDS[@]}"; do
  record="$(scontrol show job "$job_id" -o)"
  for required in \
    'JobState=PENDING' 'Reason=JobHeldUser' 'Account=mltheory' \
    'Partition=pvl-lowprio' \
    'OAT_ZERO_VARIANT=verified_entropy_gated_singleton_escape_canonical' \
    'OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.50' \
    'OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SEPARATE_OBJECTIVE_SUPPORT=1' \
    'TresPerNode=gres/gpu:a6000:1'; do
    [[ "$record" == *"$required"* ]] || {
      echo "E68 Graph hold audit failed for $job_id: $required" >&2
      exit 1
    }
  done
done

for job_id in "${ALL_IDS[@]}"; do
  scontrol update JobId="$job_id" Account=allcs \
    Partition=lowprio NodeList="$TARGET_NODES"
done

for job_id in "${ALL_IDS[@]}"; do
  record="$(scontrol show job "$job_id" -o)"
  for required in \
    'Account=allcs' 'Partition=lowprio' \
    'ReqNodeList=node205,node206,node207' \
    'TresPerNode=gres/gpu:a6000:1'; do
    [[ "$record" == *"$required"* ]] || {
      echo "Graph placement audit failed for $job_id: $required" >&2
      exit 1
    }
  done
done

python - "$OUT" "$AMENDMENT" "$0" "$E66_AUDIT" "$E68_AUDIT" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


out = Path(sys.argv[1])
audits = {
    "e66": json.loads(Path(sys.argv[4]).read_text(encoding="utf-8")),
    "e68": json.loads(Path(sys.argv[5]).read_text(encoding="utf-8")),
}
affected = {
    "e66": {30128394, 30128395, 30128396},
    "e68": {30130469, 30130470, 30130471},
}
pre_steps = {}
for cohort, audit in audits.items():
    for domain in audit["domains"].values():
        for run in domain["runs"]:
            job_id = int(run["job_id"])
            if job_id in affected[cohort]:
                pre_steps[str(job_id)] = int(run["latest_step"])

expected = set().union(*affected.values())
if set(map(int, pre_steps)) != expected:
    raise SystemExit(
        f"Graph drain-recovery step surface mismatch: {sorted(pre_steps)}"
    )

payload = {
    "schema": "e66_e68_graph_a6000_drain_recovery_amendment_v1",
    "amendment_sha256": digest(sys.argv[2]),
    "script_sha256": digest(sys.argv[3]),
    "affected_jobs": {
        "e66_graph_coloring": sorted(affected["e66"]),
        "e68_graph_coloring": sorted(affected["e68"]),
    },
    "precondition": "all affected jobs held and pending",
    "pre_amendment_latest_steps": pre_steps,
    "source_pool_health": {
        "node103": "draining: NHC reported 9 overheated GPUs",
        "node104": "draining: NHC reported 7 overheated GPUs",
    },
    "scheduler_probe": {
        "kind": "srun --test-only",
        "account": "allcs",
        "partition": "lowprio",
        "nodes": ["node205", "node206", "node207"],
        "gres": "gpu:a6000:1",
        "result": "schedulable with estimated start",
    },
    "mutation": {
        "account": "allcs",
        "partition": "lowprio",
        "nodes": ["node205", "node206", "node207"],
        "gres": "gpu:a6000:1",
    },
    "scientific_settings_changed": False,
    "paired_arms_moved_together": True,
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
echo "[e66-e68-graph-drain-recovery] released amended jobs: ${ALL_IDS[*]}"
