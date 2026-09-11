#!/usr/bin/env bash
#SBATCH --job-name=e49e-kernel-augmentation
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --nodelist=node302
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

set -euo pipefail

ROOT_DIR="${OAT_ZERO_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
STAGE="${E49E_KERNEL_STAGE:-}"
VARIANT="${E49E_KERNEL_VARIANT:-v1}"
EVIDENCE="${E49E_KERNEL_EVIDENCE:-}"
AUGMENT="${E49E_KERNEL_AUGMENT:-}"
SNAPSHOT_ROOT="${E49E_KERNEL_SOURCE_SNAPSHOT:-}"
SNAPSHOT_HASH="${E49E_KERNEL_SOURCE_HASH:-}"
FROZEN_E49D="${E49E_KERNEL_FROZEN_E49D:-}"
FROZEN_E49E="${E49E_KERNEL_FROZEN_E49E:-}"
ENDPOINT="${E49E_KERNEL_ENDPOINT:-}"
INVALID="${E49E_KERNEL_INVALID_CONTROLS:-}"
EQUIVALENT="${E49E_KERNEL_EQUIVALENT_CONTROLS:-}"
IDENTITY="${E49E_KERNEL_IDENTITY:-}"
IDENTITY_HASH="${E49E_KERNEL_IDENTITY_HASH:-}"
if [[ "$ROOT_DIR" != "/n/fs/similarity/maxent-grpo" ]] \
  || [[ "$STAGE" != toy && "$STAGE" != full ]] \
  || [[ "$VARIANT" != v1 && "$VARIANT" != v2 ]] \
  || [[ "$(hostname -s)" != node302 ]] \
  || [[ -z "$EVIDENCE" || -z "$AUGMENT" ]] \
  || [[ -z "$SNAPSHOT_ROOT" || -z "$SNAPSHOT_HASH" ]] \
  || [[ -z "$FROZEN_E49D" || -z "$FROZEN_E49E" ]] \
  || [[ -z "$ENDPOINT" || -z "$INVALID" || -z "$EQUIVALENT" ]] \
  || [[ -z "$IDENTITY" || -z "$IDENTITY_HASH" ]]; then
  echo "invalid E49E kernel-augmentation environment" >&2
  exit 2
fi

source "$ROOT_DIR/ops/repo_env.sh"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
export PYTHONPATH="$SNAPSHOT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ROOT_DIR/var/seed_paper_eval/paper310/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ "$STAGE" == toy ]]; then
  SOURCE="$ROOT_DIR/var/data/e49b_math_strategy_toy"
  EXPECTED_EVIDENCE="$ROOT_DIR/var/artifacts/e49e_kernel_augmentation_math_toy_${VARIANT}"
else
  SOURCE="$ROOT_DIR/var/data/math12k_384_math500"
  EXPECTED_EVIDENCE="$ROOT_DIR/var/artifacts/e49e_kernel_augmentation_math_full_${VARIANT}"
fi

hash_tree() {
  "$PYTHON_BIN" - "$1" <<'PY'
import hashlib
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
digest = hashlib.sha256()
for item in sorted(path for path in root.rglob("*") if path.is_file()):
    digest.update(str(item.relative_to(root)).encode("utf-8"))
    digest.update(b"\0")
    digest.update(hashlib.sha256(item.read_bytes()).digest())
print(digest.hexdigest())
PY
}

if [[ "$EVIDENCE" != "$EXPECTED_EVIDENCE" ]] \
  || [[ "$AUGMENT" != "$SNAPSHOT_ROOT/ops/math_strategy_calibration/"* ]] \
  || [[ ! -f "$AUGMENT" ]] \
  || [[ ! -f "$FROZEN_E49D/menu_records.jsonl" ]] \
  || [[ ! -f "$FROZEN_E49E/trace_bank_records.jsonl" ]] \
  || [[ ! -f "$ENDPOINT" || ! -f "$INVALID" || ! -f "$EQUIVALENT" ]] \
  || [[ ! -f "$IDENTITY" ]] \
  || [[ "$(hash_tree "$SNAPSHOT_ROOT")" != "$SNAPSHOT_HASH" ]] \
  || [[ "$(sha256sum "$IDENTITY" | cut -d' ' -f1)" != "$IDENTITY_HASH" ]]; then
  echo "E49E kernel-augmentation frozen identity failed" >&2
  exit 3
fi

exec "$PYTHON_BIN" "$AUGMENT" \
  --source "$SOURCE" \
  --input-e49d "$FROZEN_E49D" \
  --input-e49e "$FROZEN_E49E" \
  --evidence "$EVIDENCE" \
  --endpoint "$ENDPOINT" \
  --known-invalid-controls "$INVALID" \
  --known-equivalent-controls "$EQUIVALENT" \
  --workers 8 --timeout 900
