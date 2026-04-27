#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${PASSK_SWEEP_OUTPUT_ROOT:-${ROOT_DIR}/var/artifacts/passk_eval_node302_${RUN_STAMP}}"
PYTHON_BIN="${PASSK_SWEEP_PYTHON:-${ROOT_DIR}/var/seed_paper_eval/paper310/bin/python}"

TEMPLATES="${PASSK_SWEEP_TEMPLATES:-no,qwen_math,r1}"
MODEL_FAMILIES="${PASSK_SWEEP_MODEL_FAMILIES:-drgrpo,drx_grpo}"
MODEL_SEEDS="${PASSK_SWEEP_MODEL_SEEDS:-42,43,44}"
MODEL_RUN_ORDER="${PASSK_SWEEP_MODEL_RUN_ORDER:-seed_then_template_then_family}"
INFERENCE_SEEDS="${PASSK_SWEEP_INFERENCE_SEEDS:-1,2,3,4,5}"
PASS_KS="${PASSK_SWEEP_PASS_KS:-1,8}"
MEAN_K="${PASSK_SWEEP_MEAN_K:-8}"
SAMPLE_COUNT="${PASSK_SWEEP_SAMPLE_COUNT:-8}"
SAMPLE_TEMPERATURE="${PASSK_SWEEP_SAMPLE_TEMPERATURE:-1.0}"
SAMPLE_TOP_P="${PASSK_SWEEP_SAMPLE_TOP_P:-1.0}"
PROMPT_BATCHES_PER_EPOCH="${PASSK_SWEEP_PROMPT_BATCHES_PER_EPOCH:-94}"
INFER_EPOCH_CHECKPOINTS="${PASSK_SWEEP_INFER_EPOCH_CHECKPOINTS:-3}"
MAX_WORKERS="${PASSK_SWEEP_MAX_WORKERS:-8}"
MAX_TOKENS="${PASSK_SWEEP_MAX_TOKENS:-3000}"
MAX_MODEL_LEN="${PASSK_SWEEP_MAX_MODEL_LEN:-4096}"
GREEDY_BATCH_SIZE="${PASSK_SWEEP_GREEDY_BATCH_SIZE:-8}"
SAMPLED_BATCH_SIZE="${PASSK_SWEEP_SAMPLED_BATCH_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${PASSK_SWEEP_GPU_MEMORY_UTILIZATION:-0.90}"
SWAP_SPACE="${PASSK_SWEEP_SWAP_SPACE:-32}"
SEMANTIC_SIMILARITY_THRESHOLD="${PASSK_SWEEP_SEMANTIC_SIMILARITY_THRESHOLD:-0.75}"
DATASET_ROOT="${PASSK_SWEEP_DATASET_ROOT:-${ROOT_DIR}/datasets/evaluation_suite}"
CACHE_ROOT="${PASSK_SWEEP_CACHE_ROOT:-${ROOT_DIR}/var/cache/passk_eval_checkpoints}"

EXTRA_ARGS=()
if [[ "${PASSK_SWEEP_USE_WANDB:-0}" == "1" ]]; then
  EXTRA_ARGS+=("--use-wandb" "--wandb-project=${PASSK_SWEEP_WANDB_PROJECT:-oat-zero}")
  if [[ -n "${PASSK_SWEEP_WANDB_ENTITY:-}" ]]; then
    EXTRA_ARGS+=("--wandb-entity=${PASSK_SWEEP_WANDB_ENTITY}")
  fi
else
  EXTRA_ARGS+=("--no-use-wandb")
fi

if [[ "${PASSK_SWEEP_INCLUDE_TOKEN_IDS:-0}" == "1" ]]; then
  EXTRA_ARGS+=("--include-token-ids")
fi
if [[ "${PASSK_SWEEP_FORCE_DOWNLOAD:-0}" == "1" ]]; then
  EXTRA_ARGS+=("--force-download")
fi
if [[ "${PASSK_SWEEP_LOCAL_FILES_ONLY:-1}" == "1" ]]; then
  EXTRA_ARGS+=("--local-files-only")
fi
if [[ "${PASSK_SWEEP_SKIP_EXISTING:-1}" == "0" ]]; then
  EXTRA_ARGS+=("--no-skip-existing")
fi
if [[ "${PASSK_SWEEP_DRY_RUN:-0}" == "1" ]]; then
  EXTRA_ARGS+=("--dry-run")
fi

echo "[passk-sweep] host=$(hostname)"
echo "[passk-sweep] output_root=${OUTPUT_ROOT}"
echo "[passk-sweep] model_families=${MODEL_FAMILIES}"
echo "[passk-sweep] model_seeds=${MODEL_SEEDS}"
echo "[passk-sweep] model_run_order=${MODEL_RUN_ORDER}"
echo "[passk-sweep] templates=${TEMPLATES}"
echo "[passk-sweep] inference_seeds=${INFERENCE_SEEDS}"
echo "[passk-sweep] pass_ks=${PASS_KS}"
echo "[passk-sweep] mean_k=${MEAN_K}"
echo "[passk-sweep] sample_count=${SAMPLE_COUNT}"
echo "[passk-sweep] sample_temperature=${SAMPLE_TEMPERATURE}"
echo "[passk-sweep] sample_top_p=${SAMPLE_TOP_P}"
echo "[passk-sweep] infer_epoch_checkpoints=${INFER_EPOCH_CHECKPOINTS}"
echo "[passk-sweep] prompt_batches_per_epoch=${PROMPT_BATCHES_PER_EPOCH}"
echo "[passk-sweep] max_workers=${MAX_WORKERS}"
echo "[passk-sweep] semantic_similarity_threshold=${SEMANTIC_SIMILARITY_THRESHOLD}"
echo "[passk-sweep] dataset_root=${DATASET_ROOT}"
echo "[passk-sweep] cache_root=${CACHE_ROOT}"
echo "[passk-sweep] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

exec "${PYTHON_BIN}" "${ROOT_DIR}/ops/eval_model_grid_node302.py" \
  --output-root "${OUTPUT_ROOT}" \
  --model-families "${MODEL_FAMILIES}" \
  --model-seeds "${MODEL_SEEDS}" \
  --model-run-order "${MODEL_RUN_ORDER}" \
  --templates "${TEMPLATES}" \
  --dataset-root "${DATASET_ROOT}" \
  --checkpoint-cache-root "${CACHE_ROOT}" \
  --infer-epoch-checkpoints "${INFER_EPOCH_CHECKPOINTS}" \
  --prompt-batches-per-epoch "${PROMPT_BATCHES_PER_EPOCH}" \
  --pass-ks "${PASS_KS}" \
  --mean-k "${MEAN_K}" \
  --sample-count "${SAMPLE_COUNT}" \
  --sample-temperature "${SAMPLE_TEMPERATURE}" \
  --sample-top-p "${SAMPLE_TOP_P}" \
  --inference-seeds "${INFERENCE_SEEDS}" \
  --max-workers "${MAX_WORKERS}" \
  --max-tokens "${MAX_TOKENS}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --greedy-batch-size "${GREEDY_BATCH_SIZE}" \
  --sampled-batch-size "${SAMPLED_BATCH_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --swap-space "${SWAP_SPACE}" \
  --semantic-similarity-threshold "${SEMANTIC_SIMILARITY_THRESHOLD}" \
  "${EXTRA_ARGS[@]}"
