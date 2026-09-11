#!/usr/bin/env bash
# Submit the fail-closed post-v13 AntMaze paired-smoke launcher dependency.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
V13_IDENTITY="$ROOT_DIR/var/artifacts/ant_maze_interactive_warmstart_v13_identity.json"
LAUNCHER="$ROOT_DIR/ops/exp_scaling/launch_ant_maze_interactive_paired_smoke_v13.py"
BATCH="$ROOT_DIR/ops/slurm/launch_ant_maze_paired_smoke_after_v13.slurm"
OUTPUT="$ROOT_DIR/var/artifacts/ant_maze_paired_smoke_v13_dependency_identity.json"
phase="${1:-}"
case "$phase" in
  config|run) ;;
  *) echo "Usage: $0 {config|run}" >&2; exit 1 ;;
esac
for required in "$PYTHON_BIN" "$V13_IDENTITY" "$LAUNCHER" "$BATCH"; do
  [[ -e "$required" ]] || {
    echo "Missing AntMaze v13 dependency prerequisite: $required" >&2
    exit 1
  }
done
bash -n "$BATCH"
"$PYTHON_BIN" "$LAUNCHER" config
v13_job_id="$(jq -er '.job_id' "$V13_IDENTITY")"
[[ "$v13_job_id" == 30202183 ]] || {
  echo "AntMaze v13 job identity drift" >&2
  exit 1
}
launcher_sha256="$(sha256sum "$LAUNCHER" | cut -d' ' -f1)"
if [[ "$phase" == config ]]; then
  echo "[ant-paired-v13-dependency] configuration passed; no job submitted"
  exit 0
fi
[[ ! -e "$OUTPUT" ]] || {
  echo "Fresh AntMaze v13 paired-smoke dependency identity required" >&2
  exit 1
}
job_id="$(sbatch --parsable --dependency="afterok:$v13_job_id" \
  --export="ALL,ROOT_DIR=$ROOT_DIR,OAT_ZERO_EXPECTED_LAUNCHER_SHA256=$launcher_sha256,OAT_ZERO_V13_JOB_ID=$v13_job_id" \
  "$BATCH")"
job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || {
  echo "Invalid AntMaze v13 paired-smoke dependency job ID" >&2
  exit 1
}
"$PYTHON_BIN" - \
  "$OUTPUT" "$job_id" "$v13_job_id" "$launcher_sha256" "$BATCH" "$V13_IDENTITY" <<'PY'
import hashlib
import json
import os
import pathlib
import sys
import tempfile

def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "ant-maze-paired-smoke-v13-dependency-identity-v1",
    "job_id": int(sys.argv[2]),
    "dependency": f"afterok:{sys.argv[3]}",
    "v13_job_id": int(sys.argv[3]),
    "expected_paired_smoke_launcher_sha256": sys.argv[4],
    "dependency_batch_sha256": sha(sys.argv[5]),
    "v13_identity_sha256": sha(sys.argv[6]),
    "fail_closed": True,
    "online_training_before_v13_pass": False,
    "paper_jobs_before_paired_smoke_pass": False,
}
path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
with os.fdopen(fd, "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY
echo "[ant-paired-v13-dependency] submitted job $job_id afterok:$v13_job_id"
