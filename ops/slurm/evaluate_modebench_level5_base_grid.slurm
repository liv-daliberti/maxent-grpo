#!/bin/bash
# Frozen base-model evaluation of ModeBench Level 5, one array task per domain.
#
# Submitted by ops/submit_modebench_level5_base_grid.sh, which sets --gres,
# --mem and --time per model scale and passes MODEL_LABEL/MODEL_PATH/TP.
# Refuses to run unless the domain carries an admitted Level-5 dataset binding:
# evaluating prompts that are still being fitted would burn the allocation and
# produce numbers that cannot be reported.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=all
#SBATCH --account=allcs
#SBATCH --output=var/logs/modebench_level5_base_grid/%x-%A_%a.out
#SBATCH --error=var/logs/modebench_level5_base_grid/%x-%A_%a.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"          # artifacts/<dated run dir>
MODEL_LABEL="${MODEL_LABEL:?MODEL_LABEL is required}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
TP="${TP:-1}"
read -r -a DOMAINS <<< "${DOMAIN_LIST:?DOMAIN_LIST is required}"
DOMAIN="${DOMAINS[${SLURM_ARRAY_TASK_ID:?}]}"

BINDING="$RUN_ROOT/bindings/level5/$DOMAIN.json"
[[ -f "$BINDING" ]] || { echo "no Level-5 dataset binding for $DOMAIN: $BINDING" >&2; exit 78; }
STATUS="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status"))' "$BINDING")"
[[ "$STATUS" == "admitted" ]] || { echo "Level-5 $DOMAIN binding is '$STATUS', not 'admitted'" >&2; exit 78; }
# The evaluator takes the frozen rows from --rows-jsonl and cross-checks it against
# the binding, so both must be passed and must agree.
ROWS_JSONL="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["rows_jsonl"])' "$BINDING")"
[[ -f "$ROWS_JSONL" ]] || { echo "Level-5 $DOMAIN rows_jsonl missing: $ROWS_JSONL" >&2; exit 78; }

OUT="$RUN_ROOT/receipts/${MODEL_LABEL}_level5_${DOMAIN}.json"
mkdir -p "$(dirname "$OUT")" "$ROOT_DIR/var/logs/modebench_level5_base_grid"

# The independent-seed contract the evaluator enforces: vLLM V0 with the XFormers
# backend, offline hub, and no bytecode writes into the shared tree.
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$(mktemp -d /tmp/mb-l5-grid-pycache-XXXXXXXX)/pycache"

exec "$PYTHON_BIN" "$ROOT_DIR/ops/evaluate_modebench_base_grid.py" \
  --model "$MODEL_PATH" --model-label "$MODEL_LABEL" \
  --domain "$DOMAIN" --level level5 --split eval \
  --dataset-binding "$BINDING" --rows-jsonl "$ROWS_JSONL" --output "$OUT" \
  --seeds 6600500 6600501 6600502 6600503 \
  --batch-size 8 --tensor-parallel-size "$TP" \
  --gpu-memory-utilization .82 --max-model-len 2048 \
  --enable-prefix-caching --confirm-eval --resume
