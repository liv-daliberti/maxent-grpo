#!/usr/bin/env bash
#SBATCH --job-name=e49h-curated
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
SCRIPT="${E49H_FROZEN_SCRIPT:-}"
SOURCE="${E49H_SOURCE:-}"
CONTRACTS="${E49H_CONTRACTS:-}"
ENDPOINT="${E49H_ENDPOINT:-}"
CALIBRATION="${E49H_CALIBRATION:-}"
OUTPUT="${E49H_OUTPUT:-}"
IDENTITY="${E49H_IDENTITY:-}"
IDENTITY_HASH="${E49H_IDENTITY_HASH:-}"
EXPERIMENT="${E49H_EXPERIMENT:-e49h}"
EXPECTED_TRAIN="${E49H_EXPECTED_TRAIN:-12}"
EXPECTED_EVAL="${E49H_EXPECTED_EVAL:-10}"
if [[ "$ROOT_DIR" != "/n/fs/similarity/maxent-grpo" ]] \
  || [[ "$(hostname -s)" != node915 ]] \
  || [[ -z "$SCRIPT" || -z "$SOURCE" || -z "$CONTRACTS" ]] \
  || [[ -z "$ENDPOINT" || -z "$CALIBRATION" || -z "$OUTPUT" ]] \
  || [[ -z "$IDENTITY" || -z "$IDENTITY_HASH" ]]; then
  echo "invalid E49H Slurm environment" >&2
  exit 2
fi

source "$ROOT_DIR/ops/repo_env.sh"
export PYTHONDONTWRITEBYTECODE=1
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
export LD_LIBRARY_PATH="$ROOT_DIR/var/seed_paper_eval/paper310/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
for required in "$SCRIPT" "$SOURCE/train" "$SOURCE/eval" "$CONTRACTS" \
  "$ENDPOINT" "$CALIBRATION" "$IDENTITY"; do
  if [[ ! -e "$required" ]]; then
    echo "missing E49H frozen prerequisite: $required" >&2
    exit 3
  fi
done
if [[ "$(sha256sum "$IDENTITY" | cut -d' ' -f1)" != "$IDENTITY_HASH" ]]; then
  echo "E49H frozen identity hash changed" >&2
  exit 3
fi

"$PYTHON_BIN" - "$IDENTITY" "$SCRIPT" "$CONTRACTS" "$ENDPOINT" \
  "$CALIBRATION" <<'PY'
import hashlib
import json
import pathlib
import sys

identity = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
for key, path in (
    ("script_sha256", sys.argv[2]),
    ("contracts_sha256", sys.argv[3]),
    ("endpoint_sha256", sys.argv[4]),
    ("e49j_calibration_sha256", sys.argv[5]),
):
    observed = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    if identity.get(key) != observed:
        raise SystemExit(f"E49H frozen binding failed: {key}")
if identity.get("requests_at_freeze") != 0:
    raise SystemExit("E49H identity was not frozen before requests")
PY

exec "$PYTHON_BIN" "$SCRIPT" run \
  --source "$SOURCE" \
  --contracts "$CONTRACTS" \
  --endpoint "$ENDPOINT" \
  --e49j-calibration "$CALIBRATION" \
  --output "$OUTPUT" \
  --experiment "$EXPERIMENT" \
  --expected-train "$EXPECTED_TRAIN" \
  --expected-eval "$EXPECTED_EVAL" \
  --workers 4 \
  --timeout 900
