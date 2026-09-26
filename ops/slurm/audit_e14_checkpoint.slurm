#!/usr/bin/env bash
# GPU entry point for E14's exact 27-action checkpoint audit. Resource and
# placement flags are supplied by sbatch so this remains cluster-portable.
set -euo pipefail

ROOT_DIR="${E14_REPO_ROOT:?set E14_REPO_ROOT}"
source "$ROOT_DIR/ops/repo_env.sh"

PYTHON_BIN="${E14_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
PYTHON_LIB_DIR="${E14_PYTHON_LIB_DIR:-$ROOT_DIR/var/seed_paper_eval/paper310/lib}"
CHECKPOINT="${E14_CHECKPOINT:?set E14_CHECKPOINT}"
TOKENIZER="${E14_TOKENIZER:?set E14_TOKENIZER}"
DATA_ROOT="${E14_DATA_ROOT:?set E14_DATA_ROOT}"
OUTPUT="${E14_AUDIT_OUTPUT:?set E14_AUDIT_OUTPUT}"

export LD_LIBRARY_PATH="$PYTHON_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_TF=1
export USE_TF=0
export USE_FLAX=0

cd "$ROOT_DIR"
exec "$PYTHON_BIN" "$ROOT_DIR/ops/exp_scaling/audit_e14_checkpoint.py" \
  --checkpoint "$CHECKPOINT" \
  --tokenizer "$TOKENIZER" \
  --data-root "$DATA_ROOT" \
  --output "$OUTPUT" \
  --device cuda \
  --dtype bfloat16 \
  --batch-size "${E14_AUDIT_BATCH_SIZE:-8}" \
  --expected-optimizer-updates 128
