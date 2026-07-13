#!/bin/bash
# Run exact answer-mode coverage/correctness Pareto evaluation.

#SBATCH --job-name=answer_mode_pareto
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
OUTPUT_ROOT="${OAT_ZERO_PARETO_OUTPUT_ROOT:-${ROOT_DIR}/var/artifacts}"

export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}/ops${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false
export MAXENT_SEED_PAPER_SUPPRESS_GRADER_LOGS=1

echo "[pareto-slurm] host=$(hostname)"
echo "[pareto-slurm] job_id=${SLURM_JOB_ID:-}"
echo "[pareto-slurm] stamp_prefix=${STAMP_PREFIX}"
echo "[pareto-slurm] data_root=${DATA_ROOT}"
echo "[pareto-slurm] output_root=${OUTPUT_ROOT}"
echo "[pareto-slurm] seeds=${OAT_ZERO_SEEDS:-43,44,45}"
echo "[pareto-slurm] sample_counts=${OAT_ZERO_PARETO_SAMPLE_COUNTS:-1,4,8,16,32}"
echo "[pareto-slurm] temperatures=${OAT_ZERO_PARETO_TEMPERATURES:-0.6,0.8,1.0,1.2}"
echo "[pareto-slurm] checkpoint_steps=${OAT_ZERO_PARETO_CHECKPOINT_STEPS:-final}"
echo "[pareto-slurm] explicit_checkpoints=${OAT_ZERO_PARETO_CHECKPOINTS:-}"
echo "[pareto-slurm] explicit_checkpoints_file=${OAT_ZERO_PARETO_CHECKPOINTS_FILE:-}"

checkpoint_args=()
if [[ -n "${OAT_ZERO_PARETO_CHECKPOINTS_FILE:-}" ]]; then
  while IFS= read -r checkpoint_spec; do
    checkpoint_spec="$(echo "$checkpoint_spec" | xargs)"
    if [[ -n "$checkpoint_spec" ]]; then
      checkpoint_args+=(--checkpoint "$checkpoint_spec")
    fi
  done < "$OAT_ZERO_PARETO_CHECKPOINTS_FILE"
fi
if [[ -n "${OAT_ZERO_PARETO_CHECKPOINTS:-}" ]]; then
  IFS=',' read -r -a checkpoint_specs <<< "$OAT_ZERO_PARETO_CHECKPOINTS"
  for checkpoint_spec in "${checkpoint_specs[@]}"; do
    checkpoint_spec="$(echo "$checkpoint_spec" | xargs)"
    if [[ -n "$checkpoint_spec" ]]; then
      checkpoint_args+=(--checkpoint "$checkpoint_spec")
    fi
  done
fi

exec "$PYTHON_BIN" "$ROOT_DIR/ops/eval_exact_answer_mode_pareto.py" \
  --stamp-prefix "$STAMP_PREFIX" \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --run-data-root "${OAT_ZERO_RUN_DATA_ROOT:-${ROOT_DIR}/var/data}" \
  "${checkpoint_args[@]}" \
  --seeds "${OAT_ZERO_SEEDS:-43,44,45}" \
  --variants "${OAT_ZERO_PARETO_VARIANTS:-grpo,answer_maxent}" \
  --checkpoint-steps "${OAT_ZERO_PARETO_CHECKPOINT_STEPS:-final}" \
  --splits "${OAT_ZERO_PARETO_SPLITS:-multi_answer,unique_answer}" \
  --sample-counts "${OAT_ZERO_PARETO_SAMPLE_COUNTS:-1,4,8,16,32}" \
  --temperatures "${OAT_ZERO_PARETO_TEMPERATURES:-0.6,0.8,1.0,1.2}" \
  --top-p "${OAT_ZERO_PARETO_TOP_P:-1.0}" \
  --max-tokens "${OAT_ZERO_PARETO_MAX_TOKENS:-192}" \
  --max-model-len "${OAT_ZERO_PARETO_MAX_MODEL_LEN:-1024}" \
  --dtype "${OAT_ZERO_PARETO_DTYPE:-bfloat16}" \
  --batch-size "${OAT_ZERO_PARETO_BATCH_SIZE:-8}" \
  --seed "${OAT_ZERO_PARETO_SEED:-20260529}" \
  --gpu-memory-utilization "${OAT_ZERO_PARETO_GPU_MEMORY_UTILIZATION:-0.82}" \
  --swap-space "${OAT_ZERO_PARETO_SWAP_SPACE:-16}" \
  --plot-splits "${OAT_ZERO_PARETO_PLOT_SPLITS:-multi_answer,unique_answer}" \
  --trajectory-temperature "${OAT_ZERO_PARETO_TRAJECTORY_TEMPERATURE:-1.0}" \
  --trajectory-sample-count "${OAT_ZERO_PARETO_TRAJECTORY_SAMPLE_COUNT:-32}" \
  --budget-batches-per-epoch "${OAT_ZERO_PARETO_BUDGET_BATCHES_PER_EPOCH:-192}" \
  ${OAT_ZERO_PARETO_INCLUDE_TEXT:+--include-text}
