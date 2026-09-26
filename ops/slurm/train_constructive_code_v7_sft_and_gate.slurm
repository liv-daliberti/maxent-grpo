#!/bin/bash
#SBATCH --job-name=constructive-v7-sft-gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=all
#SBATCH --account=allcs
#SBATCH --time=04:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail
ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
SOURCE_ROOT="${OAT_ZERO_SOURCE_ROOT:?source root is required}"
GATE_OPS_ROOT="${OAT_ZERO_GATE_OPS_ROOT:?gate ops root is required}"
EXECUTION_ROOT="${OAT_ZERO_EXECUTION_ROOT:?execution root is required}"
SOURCE_HASH="${OAT_ZERO_SOURCE_HASH:?source hash is required}"
EXECUTION_HASH="${OAT_ZERO_EXECUTION_HASH:?execution hash is required}"
IDENTITY="${OAT_ZERO_PROTOCOL_IDENTITY:?protocol identity is required}"

source "$ROOT_DIR/ops/repo_env.sh"
export PYTHONPATH="$EXECUTION_ROOT:$GATE_OPS_ROOT:$SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OAT_ZERO_REPO_ROOT="$ROOT_DIR" OAT_ZERO_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_TESTLIB_ROOT="$GATE_OPS_ROOT/testlib"
export OAT_ZERO_SANDBOX_SOURCE="$GATE_OPS_ROOT/constructive_code_sandbox.c"
export OAT_ZERO_CONSTRUCTIVE_V5_PROTOCOL="$GATE_OPS_ROOT/protocol/constructive_code_executable_slate_v5_20260730.md"
export OAT_ZERO_CONSTRUCTIVE_V5_INDEX="$GATE_OPS_ROOT/inputs/candidate_source_index.json"
export OAT_ZERO_CONSTRUCTIVE_V5_V1_ROOT="$GATE_OPS_ROOT/inputs/v1"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V1=0 TOKENIZERS_PARALLELISM=false

RUN_ROOT="${TMPDIR:-/tmp}/constructive-v7-sft-gate-${SLURM_JOB_ID:?job ID required}"
mkdir -p "$RUN_ROOT/build" "$RUN_ROOT/scratch"
OUTPUT_MODEL="$ROOT_DIR/var/models/constructive_code_v7_sft"
PROTOCOL="$EXECUTION_ROOT/constructive_code_v7_train_only_sft_20260730.md"

"$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$EXECUTION_ROOT/train_constructive_code_v7_sft.py" \
  --base-model "$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-Coder-0.5B-Instruct/snapshots/ea3f2471cf1b1f0db85067f1ef93848e38e88c25" \
  --examples "$ROOT_DIR/var/data/constructive_code_v7_sft/examples.jsonl" \
  --manifest "$ROOT_DIR/var/data/constructive_code_v7_sft/manifest.json" \
  --protocol "$PROTOCOL" --identity "$IDENTITY" \
  --output-model "$OUTPUT_MODEL" \
  --output-receipt "$ROOT_DIR/var/artifacts/constructive_code_v7_sft.json" \
  --job-id "$SLURM_JOB_ID"

"$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$EXECUTION_ROOT/evaluate_constructive_code_v7_coder_viability.py" \
  --slate-root "$ROOT_DIR/var/data/constructive_code_v5" \
  --v1-root "$GATE_OPS_ROOT/inputs/v1" \
  --image "$ROOT_DIR/var/images/python-3.10-slim-c1e4e6c01eb4.sqsh" \
  --runtime-root "$RUN_ROOT/runtime" --launcher "$RUN_ROOT/constructive_code_sandbox" \
  --build-root "$RUN_ROOT/build" --scratch-root "$RUN_ROOT/scratch" \
  --model "$OUTPUT_MODEL" \
  --gate-audit "$ROOT_DIR/var/artifacts/constructive_code_v6_gate_audit.json" \
  --gate-identity "$ROOT_DIR/var/artifacts/constructive_code_v6_gate_identity.json" \
  --protocol "$PROTOCOL" --identity "$IDENTITY" \
  --output "$ROOT_DIR/var/artifacts/constructive_code_v7_post_sft_viability.json" \
  --sample-count 64 --prefix-count 16 --seed 77102 \
  --temperature 1.0 --top-p 1.0 --max-tokens 1024 \
  --max-model-len 8192 --generation-batch-size 16 --execution-workers 16 \
  --source-hash "$SOURCE_HASH" --execution-hash "$EXECUTION_HASH" --job-id "$SLURM_JOB_ID"
