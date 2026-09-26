#!/bin/bash
#SBATCH --job-name=constructive-v5-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=all
#SBATCH --account=allcs
#SBATCH --time=7-00:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
SOURCE_ROOT="${OAT_ZERO_SOURCE_ROOT:?source root is required}"
GATE_OPS_ROOT="${OAT_ZERO_GATE_OPS_ROOT:?gate ops root is required}"
EXECUTION_ROOT="${OAT_ZERO_EXECUTION_ROOT:?execution root is required}"
SOURCE_HASH="${OAT_ZERO_SOURCE_HASH:?source hash is required}"
EXECUTION_HASH="${OAT_ZERO_EXECUTION_HASH:?execution hash is required}"
MODE="${OAT_ZERO_CONSTRUCTIVE_MODE:?mode is required}"
ARM="${OAT_ZERO_CONSTRUCTIVE_ARM:?arm is required}"
SEED="${OAT_ZERO_CONSTRUCTIVE_SEED:?seed is required}"
PROTOCOL="${OAT_ZERO_CONSTRUCTIVE_PROTOCOL:?protocol is required}"
IDENTITY="${OAT_ZERO_CONSTRUCTIVE_IDENTITY:?identity is required}"
RECEIPT="${OAT_ZERO_CONSTRUCTIVE_RECEIPT:?receipt is required}"
CANDIDATES="${OAT_ZERO_CONSTRUCTIVE_CANDIDATES:?candidate ledger is required}"

source "$ROOT_DIR/ops/repo_env.sh"
export PYTHONPATH="$EXECUTION_ROOT:$GATE_OPS_ROOT:$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_TESTLIB_ROOT="$GATE_OPS_ROOT/testlib"
export OAT_ZERO_SANDBOX_SOURCE="$GATE_OPS_ROOT/constructive_code_sandbox.c"
export OAT_ZERO_CONSTRUCTIVE_V5_PROTOCOL="$GATE_OPS_ROOT/protocol/constructive_code_executable_slate_v5_20260730.md"
export OAT_ZERO_CONSTRUCTIVE_V5_INDEX="$GATE_OPS_ROOT/inputs/candidate_source_index.json"
export OAT_ZERO_CONSTRUCTIVE_V5_V1_ROOT="$GATE_OPS_ROOT/inputs/v1"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

RUN_ROOT="${TMPDIR:-/tmp}/constructive-v5-${MODE}-${ARM}-s${SEED}-${SLURM_JOB_ID:?job ID required}"
mkdir -p "$RUN_ROOT/build" "$RUN_ROOT/scratch"
if [[ -n "${OAT_ZERO_CONSTRUCTIVE_METRICS:-}" ]]; then
  METRICS="$OAT_ZERO_CONSTRUCTIVE_METRICS"
else
  RUN_DATA_DIR="${OAT_ZERO_CONSTRUCTIVE_RUN_DIR:?run directory is required when metrics path is omitted}"
  METRICS="$RUN_DATA_DIR/debug_job${SLURM_JOB_ID}/train_metrics.jsonl"
fi
extra=()
if [[ "$MODE" == "stage_b" ]]; then
  extra+=(--paired-audit "${OAT_ZERO_CONSTRUCTIVE_PAIRED_AUDIT:?paired audit is required}")
  extra+=(--output-evaluations "${OAT_ZERO_CONSTRUCTIVE_EVALUATIONS:?evaluation ledger is required}")
elif [[ "$MODE" != "paired_smoke" ]]; then
  echo "Invalid ConstructiveCode mode: $MODE" >&2
  exit 1
fi

exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$EXECUTION_ROOT/train_constructive_code_v5.py" \
  --mode "$MODE" --arm "$ARM" --seed "$SEED" \
  --model "$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-Coder-0.5B-Instruct/snapshots/ea3f2471cf1b1f0db85067f1ef93848e38e88c25" \
  --slate-root "$ROOT_DIR/var/data/constructive_code_v5" \
  --v1-root "$GATE_OPS_ROOT/inputs/v1" \
  --image "$ROOT_DIR/var/images/python-3.10-slim-c1e4e6c01eb4.sqsh" \
  --runtime-root "$RUN_ROOT/runtime" --launcher "$RUN_ROOT/constructive_code_sandbox" \
  --build-root "$RUN_ROOT/build" --scratch-root "$RUN_ROOT/scratch" \
  --gate-audit "$ROOT_DIR/var/artifacts/constructive_code_v5_gate_audit.json" \
  --gate-identity "$ROOT_DIR/var/artifacts/constructive_code_v5_gate_identity.json" \
  --viability-receipt "$ROOT_DIR/var/artifacts/constructive_code_v5_coder_05b_viability.json" \
  --protocol "$PROTOCOL" --identity "$IDENTITY" \
  --output-receipt "$RECEIPT" --output-metrics "$METRICS" \
  --output-candidates "$CANDIDATES" \
  --learning-rate 2e-7 --max-model-len 8192 --execution-workers 16 \
  --source-hash "$SOURCE_HASH" --execution-hash "$EXECUTION_HASH" \
  --job-id "$SLURM_JOB_ID" "${extra[@]}"
