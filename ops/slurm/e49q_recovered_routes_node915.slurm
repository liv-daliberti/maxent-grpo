#!/usr/bin/env bash
#SBATCH --job-name=e49q-recover
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --nodelist=node915
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail
ROOT_DIR="${OAT_ZERO_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
SCRIPT="${E49Q_FROZEN_SCRIPT:-}"
SOURCE="${E49Q_SOURCE:-}"
EVIDENCE_DIRS="${E49Q_EVIDENCE_DIRS:-}"
ENDPOINT="${E49Q_ENDPOINT:-}"
CALIBRATION="${E49Q_CALIBRATION:-}"
OUTPUT="${E49Q_OUTPUT:-}"
IDENTITY="${E49Q_IDENTITY:-}"
IDENTITY_HASH="${E49Q_IDENTITY_HASH:-}"
if [[ "$ROOT_DIR" != "/n/fs/similarity/maxent-grpo" ]] \
  || [[ "$(hostname -s)" != node915 ]] \
  || [[ -z "$SCRIPT" || -z "$SOURCE" || -z "$EVIDENCE_DIRS" ]] \
  || [[ -z "$ENDPOINT" || -z "$CALIBRATION" || -z "$OUTPUT" ]] \
  || [[ -z "$IDENTITY" || -z "$IDENTITY_HASH" ]]; then
  echo "invalid E49Q Slurm environment" >&2
  exit 2
fi

source "$ROOT_DIR/ops/repo_env.sh"
export PYTHONDONTWRITEBYTECODE=1
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
export LD_LIBRARY_PATH="$ROOT_DIR/var/seed_paper_eval/paper310/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$(dirname "$(dirname "$(dirname "$SCRIPT")")")/src${PYTHONPATH:+:$PYTHONPATH}"
IFS=: read -r -a evidence <<<"$EVIDENCE_DIRS"
for required in "$SCRIPT" "$SOURCE/train" "$SOURCE/eval" "$ENDPOINT" \
  "$CALIBRATION" "$IDENTITY" "${evidence[@]}"; do
  if [[ ! -e "$required" ]]; then
    echo "missing E49Q frozen prerequisite: $required" >&2
    exit 3
  fi
done
if [[ "$(sha256sum "$IDENTITY" | cut -d' ' -f1)" != "$IDENTITY_HASH" ]]; then
  echo "E49Q frozen identity hash changed" >&2
  exit 3
fi

"$PYTHON_BIN" - "$IDENTITY" "$SCRIPT" "$ENDPOINT" "$CALIBRATION" \
  "$(dirname "${evidence[0]}")" "$SOURCE/train" "$SOURCE/eval" <<'PY'
import hashlib
import json
import pathlib
import sys

def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

def tree(path):
    root = pathlib.Path(path)
    digest = hashlib.sha256()
    for item in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(item.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()

identity = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
checks = {
    "script_sha256": sha(sys.argv[2]),
    "endpoint_sha256": sha(sys.argv[3]),
    "e49j_calibration_sha256": sha(sys.argv[4]),
    "frozen_evidence_tree_sha256": tree(sys.argv[5]),
    "source_train_tree_sha256": tree(sys.argv[6]),
    "source_eval_tree_sha256": tree(sys.argv[7]),
}
for key, observed in checks.items():
    if identity.get(key) != observed:
        raise SystemExit(f"E49Q frozen binding failed: {key}")
if identity.get("requests_at_freeze") != 0:
    raise SystemExit("E49Q identity was not frozen before requests")
PY

run_args=()
for item in "${evidence[@]}"; do
  run_args+=(--evidence "$item")
done
exec "$PYTHON_BIN" "$SCRIPT" run \
  --source "$SOURCE" \
  "${run_args[@]}" \
  --endpoint "$ENDPOINT" \
  --e49j-calibration "$CALIBRATION" \
  --output "$OUTPUT" \
  --workers 4 \
  --timeout 900
