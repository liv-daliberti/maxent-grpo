#!/bin/bash
# Run sampled exact answer-mode coverage evaluation for tiny OAT probes.

#SBATCH --job-name=answer_mode_cov
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --nodelist=node302
#SBATCH --time=06:00:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON_BIN="${OAT_ZERO_PYTHON:-${ROOT_DIR}/var/seed_paper_eval/paper310/bin/python}"

source "${ROOT_DIR}/ops/repo_env.sh"

STAMP_PREFIX="${RUN_STAMP_PREFIX:?RUN_STAMP_PREFIX is required}"
DATA_ROOT="${OAT_ZERO_ANSWER_MODE_DATA_ROOT:-${ROOT_DIR}/var/data/exact_answer_mode_probe}"
OUTPUT_ROOT="${OAT_ZERO_COVERAGE_OUTPUT_ROOT:-${ROOT_DIR}/var/artifacts}"

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false
export MAXENT_SEED_PAPER_SUPPRESS_GRADER_LOGS=1

echo "[coverage-slurm] host=$(hostname)"
echo "[coverage-slurm] job_id=${SLURM_JOB_ID:-}"
echo "[coverage-slurm] stamp_prefix=${STAMP_PREFIX}"
echo "[coverage-slurm] data_root=${DATA_ROOT}"
echo "[coverage-slurm] output_root=${OUTPUT_ROOT}"
echo "[coverage-slurm] seeds=${OAT_ZERO_SEEDS:-43,44,45}"
echo "[coverage-slurm] sample_count=${OAT_ZERO_COVERAGE_SAMPLE_COUNT:-32}"

exec "$PYTHON_BIN" "$ROOT_DIR/ops/eval_exact_answer_mode_coverage.py" \
  --stamp-prefix "$STAMP_PREFIX" \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --run-data-root "${OAT_ZERO_RUN_DATA_ROOT:-${ROOT_DIR}/var/data}" \
  --seeds "${OAT_ZERO_SEEDS:-43,44,45}" \
  --variants "${OAT_ZERO_COVERAGE_VARIANTS:-grpo,answer_maxent}" \
  --sample-count "${OAT_ZERO_COVERAGE_SAMPLE_COUNT:-32}" \
  --temperature "${OAT_ZERO_COVERAGE_TEMPERATURE:-1.0}" \
  --top-p "${OAT_ZERO_COVERAGE_TOP_P:-1.0}" \
  --max-tokens "${OAT_ZERO_COVERAGE_MAX_TOKENS:-192}" \
  --max-model-len "${OAT_ZERO_COVERAGE_MAX_MODEL_LEN:-1024}" \
  --batch-size "${OAT_ZERO_COVERAGE_BATCH_SIZE:-8}" \
  --seed "${OAT_ZERO_COVERAGE_SEED:-20260529}" \
  --gpu-memory-utilization "${OAT_ZERO_COVERAGE_GPU_MEMORY_UTILIZATION:-0.82}" \
  --swap-space "${OAT_ZERO_COVERAGE_SWAP_SPACE:-16}" \
  ${OAT_ZERO_COVERAGE_INCLUDE_TEXT:+--include-text}
