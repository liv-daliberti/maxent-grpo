#!/usr/bin/env bash
#SBATCH --job-name=e49g-pair-veto
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --nodelist=node915
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00

set -euo pipefail
ROOT_DIR="${OAT_ZERO_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
SCRIPT="${E49G_FROZEN_SCRIPT:-}"
PACKET="${E49G_PACKET:-}"
ENDPOINT="${E49G_ENDPOINT:-}"
OUTPUT="${E49G_OUTPUT:-}"
IDENTITY="${E49G_IDENTITY:-}"
IDENTITY_HASH="${E49G_IDENTITY_HASH:-}"
if [[ "$ROOT_DIR" != "/n/fs/similarity/maxent-grpo" ]] \
  || [[ "$(hostname -s)" != node915 ]] \
  || [[ -z "$SCRIPT" || -z "$PACKET" || -z "$ENDPOINT" ]] \
  || [[ -z "$OUTPUT" || -z "$IDENTITY" || -z "$IDENTITY_HASH" ]]; then
  echo "invalid E49G Slurm environment" >&2
  exit 2
fi

source "$ROOT_DIR/ops/repo_env.sh"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
export LD_LIBRARY_PATH="$ROOT_DIR/var/seed_paper_eval/paper310/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ ! -f "$SCRIPT" || ! -f "$PACKET" || ! -f "$ENDPOINT" ]] \
  || [[ ! -f "$IDENTITY" ]] \
  || [[ "$(sha256sum "$IDENTITY" | cut -d' ' -f1)" != "$IDENTITY_HASH" ]]; then
  echo "E49G frozen prerequisite is missing" >&2
  exit 3
fi

"$PYTHON_BIN" - "$IDENTITY" "$SCRIPT" "$PACKET" "$ENDPOINT" <<'PY'
import hashlib
import json
import pathlib
import sys
identity = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
for key, path in (
    ("script_sha256", sys.argv[2]),
    ("packet_sha256", sys.argv[3]),
    ("endpoint_sha256", sys.argv[4]),
):
    observed = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    if identity.get(key) != observed:
        raise SystemExit(f"E49G frozen binding failed: {key}")
if identity.get("requests_at_freeze") != 0:
    raise SystemExit("E49G identity was not frozen before requests")
PY

exec "$PYTHON_BIN" "$SCRIPT" run \
  --packet "$PACKET" \
  --endpoint "$ENDPOINT" \
  --output "$OUTPUT" \
  --workers 4 \
  --timeout 900
