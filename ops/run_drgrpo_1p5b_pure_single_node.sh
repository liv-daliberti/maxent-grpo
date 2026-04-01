#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

PURE_DRGRPO_ROOT="${PURE_DRGRPO_ROOT:-$ROOT_DIR/var/external/understand-r1-zero}"
PYTHON_BIN="${PURE_DRGRPO_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
PYTHON_LIB_DIR="${PURE_DRGRPO_PYTHON_LIB_DIR:-$ROOT_DIR/var/seed_paper_eval/paper310/lib}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SAVE_PATH="${SAVE_PATH:-$ROOT_DIR/var/data/drgrpo_1p5b_pure_oat_r1_${RUN_STAMP}}"
SAVE_STEPS="${SAVE_STEPS:-50}"
MAX_SAVE_NUM="${MAX_SAVE_NUM:-10}"
WB_PROJECT="${WB_PROJECT:-oat-zero}"
WB_RUN_NAME="${WB_RUN_NAME:-qwen2.5-Math-1.5b-drgrpo-r1template}"
PURE_OBJECTIVE="${PURE_OBJECTIVE:-grpo}"
PURE_BETA="${PURE_BETA:-0.0}"
PURE_MAXENT_ALPHA="${PURE_MAXENT_ALPHA:-0.0}"
PURE_LISTWISE_TAU="${PURE_LISTWISE_TAU:-0.5}"
PURE_LISTWISE_Q_TEMPERATURE="${PURE_LISTWISE_Q_TEMPERATURE:-2.0}"
PURE_LISTWISE_Q_EPSILON="${PURE_LISTWISE_Q_EPSILON:-1e-6}"
PURE_MAXENT_LOGPROB_CHUNK_SIZE="${PURE_MAXENT_LOGPROB_CHUNK_SIZE:-0}"
PURE_MAXENT_ALPHA_RAISE_ON_LOW_KL="${PURE_MAXENT_ALPHA_RAISE_ON_LOW_KL:-0}"
PURE_MAXENT_ALPHA_LOWER_ON_HIGH_KL="${PURE_MAXENT_ALPHA_LOWER_ON_HIGH_KL:-0}"
PURE_MAXENT_ALPHA_KL_THRESHOLD="${PURE_MAXENT_ALPHA_KL_THRESHOLD:-0.07}"
PURE_MAXENT_ALPHA_KL_GAIN="${PURE_MAXENT_ALPHA_KL_GAIN:-0.5}"
PURE_MAXENT_ALPHA_DISABLE_OUTSIDE_TRUST_ZONE="${PURE_MAXENT_ALPHA_DISABLE_OUTSIDE_TRUST_ZONE:-0}"
PURE_MAXENT_ALPHA_KL_MIN_MULTIPLIER="${PURE_MAXENT_ALPHA_KL_MIN_MULTIPLIER:-0.5}"
PURE_MAXENT_ALPHA_KL_MAX_MULTIPLIER="${PURE_MAXENT_ALPHA_KL_MAX_MULTIPLIER:-1.5}"
PURE_SEED_GRPO_ENABLED="${PURE_SEED_GRPO_ENABLED:-0}"
PURE_SEED_GRPO_ALPHA="${PURE_SEED_GRPO_ALPHA:-0.0417}"
PURE_SEED_GRPO_ALPHA_NORMALIZE_BY_MAX_ENTROPY="${PURE_SEED_GRPO_ALPHA_NORMALIZE_BY_MAX_ENTROPY:-1}"
PURE_SEED_GRPO_LENGTH_NORMALIZE_LOGPROBS="${PURE_SEED_GRPO_LENGTH_NORMALIZE_LOGPROBS:-1}"
PROMPT_DATA="${PROMPT_DATA:-$ROOT_DIR/datasets/train/math_12k}"
EVAL_DATA="${EVAL_DATA:-$PURE_DRGRPO_ROOT/datasets/evaluation_suite}"
EVAL_STEPS="${EVAL_STEPS:-16}"
EVAL_N="${EVAL_N:-1}"
N_GPU="${N_GPU:-8}"
N_SAMPLE="${N_SAMPLE:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
TRAIN_BATCH_SIZE_PER_DEVICE="${TRAIN_BATCH_SIZE_PER_DEVICE:-1}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-128}"
ROLLOUT_BATCH_SIZE_PER_DEVICE="${ROLLOUT_BATCH_SIZE_PER_DEVICE:-$((ROLLOUT_BATCH_SIZE / N_GPU))}"
PI_BUFFER_MAXLEN_PER_DEVICE="${PI_BUFFER_MAXLEN_PER_DEVICE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLE / N_GPU))}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing pure Dr.GRPO python at $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$PURE_DRGRPO_ROOT/train_zero_math.py" ]]; then
  echo "Missing upstream pure Dr.GRPO checkout at $PURE_DRGRPO_ROOT" >&2
  exit 1
fi

if [[ ! -d "$PROMPT_DATA" ]]; then
  echo "Missing prompt dataset at $PROMPT_DATA" >&2
  exit 1
fi

if [[ ! -d "$EVAL_DATA" ]]; then
  echo "Missing eval dataset at $EVAL_DATA" >&2
  exit 1
fi

mkdir -p "$SAVE_PATH"

export CUDA_VISIBLE_DEVICES
export PATH="$(dirname "$PYTHON_BIN"):${PATH}"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export USE_TF="${USE_TF:-0}"
export TRANSFORMERS_NO_TF="${TRANSFORMERS_NO_TF:-1}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export TMPDIR="${PURE_DRGRPO_TMPDIR:-$ROOT_DIR/var/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$ROOT_DIR/var/cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ROOT_DIR/var/cache/torchinductor}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$ROOT_DIR/var/cache/xdg/torch_extensions}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"
unset PYTORCH_CUDA_ALLOC_CONF
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

if [[ -d "$PYTHON_LIB_DIR" ]]; then
  export LD_LIBRARY_PATH="${PYTHON_LIB_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

flash_attn_flag=(--no-flash-attn)
if [[ "${PURE_DRGRPO_ENABLE_FLASH_ATTN:-0}" == "1" ]]; then
  flash_attn_flag=(--flash-attn)
else
  echo "[pure-drgrpo] flash_attn is unavailable in the pure runtime; disabling the optional flag while keeping the upstream OAT/Dr.GRPO backend." >&2
fi

if [[ "${PURE_DRGRPO_PREBUILD_FUSED_ADAM:-1}" == "1" ]]; then
  echo "[pure-drgrpo] prebuilding DeepSpeed fused_adam extension..."
  "$PYTHON_BIN" - <<'PY'
from deepspeed.ops.op_builder import FusedAdamBuilder

FusedAdamBuilder().load(verbose=True)
print("[pure-drgrpo] fused_adam ready", flush=True)
PY
fi

if [[ "${PURE_DRGRPO_PREWARM_MODEL_CACHE:-1}" == "1" ]]; then
  echo "[pure-drgrpo] prewarming Hugging Face snapshot into ${HF_HOME}..."
  "$PYTHON_BIN" - <<'PY'
import os

from huggingface_hub import snapshot_download

repo_id = "Qwen/Qwen2.5-Math-1.5B"
snapshot_path = snapshot_download(
    repo_id=repo_id,
    cache_dir=os.environ["HF_HOME"],
)
print(f"[pure-drgrpo] hf snapshot ready at {snapshot_path}", flush=True)
PY
fi

save_ckpt_flag=()
if [[ "${SAVE_CKPT:-1}" == "1" ]]; then
  save_ckpt_flag=(--save-ckpt)
fi

maxent_alpha_low_kl_flag=(--no-maxent-alpha-raise-on-low-kl)
if [[ "$PURE_MAXENT_ALPHA_RAISE_ON_LOW_KL" == "1" ]]; then
  maxent_alpha_low_kl_flag=(--maxent-alpha-raise-on-low-kl)
fi

maxent_alpha_high_kl_flag=(--no-maxent-alpha-lower-on-high-kl)
if [[ "$PURE_MAXENT_ALPHA_LOWER_ON_HIGH_KL" == "1" ]]; then
  maxent_alpha_high_kl_flag=(--maxent-alpha-lower-on-high-kl)
fi

maxent_alpha_trust_zone_flag=(--no-maxent-alpha-disable-outside-trust-zone)
if [[ "$PURE_MAXENT_ALPHA_DISABLE_OUTSIDE_TRUST_ZONE" == "1" ]]; then
  maxent_alpha_trust_zone_flag=(--maxent-alpha-disable-outside-trust-zone)
fi

seed_grpo_flag=(--no-seed-grpo-enabled)
if [[ "$PURE_SEED_GRPO_ENABLED" == "1" ]]; then
  seed_grpo_flag=(--seed-grpo-enabled)
fi

seed_grpo_norm_flag=(--no-seed-grpo-alpha-normalize-by-max-entropy)
if [[ "$PURE_SEED_GRPO_ALPHA_NORMALIZE_BY_MAX_ENTROPY" == "1" ]]; then
  seed_grpo_norm_flag=(--seed-grpo-alpha-normalize-by-max-entropy)
fi

seed_grpo_len_norm_flag=(--no-seed-grpo-length-normalize-logprobs)
if [[ "$PURE_SEED_GRPO_LENGTH_NORMALIZE_LOGPROBS" == "1" ]]; then
  seed_grpo_len_norm_flag=(--seed-grpo-length-normalize-logprobs)
fi

cmd=(
  "$PYTHON_BIN" train_zero_math.py
  --critic_type drgrpo
  --gpus "$N_GPU"
  --enable_prefix_caching
  --collocate
  --vllm_sleep
  --vllm_gpu_ratio 0.35
  --gradient-checkpointing
  "${flash_attn_flag[@]}"
  --bf16
  --rnd-seed
  --learning_rate 0.000001
  --lr_scheduler constant
  --num_ppo_epochs 1
  --objective "$PURE_OBJECTIVE"
  --beta "$PURE_BETA"
  --maxent-alpha "$PURE_MAXENT_ALPHA"
  --maxent-tau "$PURE_LISTWISE_TAU"
  --maxent-q-temperature "$PURE_LISTWISE_Q_TEMPERATURE"
  --maxent-q-epsilon "$PURE_LISTWISE_Q_EPSILON"
  --maxent-alpha-kl-threshold "$PURE_MAXENT_ALPHA_KL_THRESHOLD"
  --maxent-alpha-kl-gain "$PURE_MAXENT_ALPHA_KL_GAIN"
  --maxent-alpha-kl-min-multiplier "$PURE_MAXENT_ALPHA_KL_MIN_MULTIPLIER"
  --maxent-alpha-kl-max-multiplier "$PURE_MAXENT_ALPHA_KL_MAX_MULTIPLIER"
  "${maxent_alpha_low_kl_flag[@]}"
  "${maxent_alpha_high_kl_flag[@]}"
  "${maxent_alpha_trust_zone_flag[@]}"
  "${seed_grpo_flag[@]}"
  --seed-grpo-alpha "$PURE_SEED_GRPO_ALPHA"
  "${seed_grpo_norm_flag[@]}"
  "${seed_grpo_len_norm_flag[@]}"
  --oracle_type reward
  --oracle math
  --pretrain Qwen/Qwen2.5-Math-1.5B
  --prompt_template r1
  --zero-stage 2
  --ref_offload
  --prompt_data "$PROMPT_DATA"
  --train_split train
  --input_key problem
  --output_key answer
  --max-train 9999999
  --num_prompt_epoch 20
  --prompt_max_length 1024
  --num_samples "$N_SAMPLE"
  --temperature 1
  --top_p 1
  --generate_max_length 3000
  --save_path "$SAVE_PATH"
  --save-steps "$SAVE_STEPS"
  --max-save-num "$MAX_SAVE_NUM"
  "${save_ckpt_flag[@]}"
  --train_batch_size "$TRAIN_BATCH_SIZE"
  --train_batch_size_per_device "$TRAIN_BATCH_SIZE_PER_DEVICE"
  --rollout_batch_size "$ROLLOUT_BATCH_SIZE"
  --rollout_batch_size_per_device "$ROLLOUT_BATCH_SIZE_PER_DEVICE"
  --pi_buffer_maxlen_per_device "$PI_BUFFER_MAXLEN_PER_DEVICE"
  --eval_batch_size 200
  --eval-steps "$EVAL_STEPS"
  --eval-n "$EVAL_N"
  --eval_temperature 0
  --eval_generate_max_length 3000
  --eval_data "$EVAL_DATA"
  --eval_input_key input
  --use-wb
  --wb_project "$WB_PROJECT"
  --wb-run-name "$WB_RUN_NAME"
)

if [[ "$PURE_MAXENT_LOGPROB_CHUNK_SIZE" -gt 0 ]]; then
  cmd+=(
    --maxent-logprob-chunk-size "$PURE_MAXENT_LOGPROB_CHUNK_SIZE"
  )
fi

echo "[pure-drgrpo] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[pure-drgrpo] python=${PYTHON_BIN}"
echo "[pure-drgrpo] checkout=${PURE_DRGRPO_ROOT}"
echo "[pure-drgrpo] save_path=${SAVE_PATH}"
echo "[pure-drgrpo] save_steps=${SAVE_STEPS}"
echo "[pure-drgrpo] save_ckpt=${SAVE_CKPT:-1}"
echo "[pure-drgrpo] max_save_num=${MAX_SAVE_NUM}"
echo "[pure-drgrpo] objective=${PURE_OBJECTIVE}"
echo "[pure-drgrpo] beta=${PURE_BETA}"
echo "[pure-drgrpo] maxent_alpha=${PURE_MAXENT_ALPHA}"
echo "[pure-drgrpo] listwise_tau=${PURE_LISTWISE_TAU}"
echo "[pure-drgrpo] listwise_q_temperature=${PURE_LISTWISE_Q_TEMPERATURE}"
echo "[pure-drgrpo] listwise_q_epsilon=${PURE_LISTWISE_Q_EPSILON}"
echo "[pure-drgrpo] maxent_logprob_chunk_size=${PURE_MAXENT_LOGPROB_CHUNK_SIZE}"
echo "[pure-drgrpo] maxent_alpha_raise_on_low_kl=${PURE_MAXENT_ALPHA_RAISE_ON_LOW_KL}"
echo "[pure-drgrpo] maxent_alpha_lower_on_high_kl=${PURE_MAXENT_ALPHA_LOWER_ON_HIGH_KL}"
echo "[pure-drgrpo] maxent_alpha_kl_threshold=${PURE_MAXENT_ALPHA_KL_THRESHOLD}"
echo "[pure-drgrpo] maxent_alpha_kl_gain=${PURE_MAXENT_ALPHA_KL_GAIN}"
echo "[pure-drgrpo] maxent_alpha_disable_outside_trust_zone=${PURE_MAXENT_ALPHA_DISABLE_OUTSIDE_TRUST_ZONE}"
echo "[pure-drgrpo] maxent_alpha_kl_min_multiplier=${PURE_MAXENT_ALPHA_KL_MIN_MULTIPLIER}"
echo "[pure-drgrpo] maxent_alpha_kl_max_multiplier=${PURE_MAXENT_ALPHA_KL_MAX_MULTIPLIER}"
echo "[pure-drgrpo] seed_grpo_enabled=${PURE_SEED_GRPO_ENABLED}"
echo "[pure-drgrpo] seed_grpo_alpha=${PURE_SEED_GRPO_ALPHA}"
echo "[pure-drgrpo] seed_grpo_alpha_normalize_by_max_entropy=${PURE_SEED_GRPO_ALPHA_NORMALIZE_BY_MAX_ENTROPY}"
echo "[pure-drgrpo] seed_grpo_length_normalize_logprobs=${PURE_SEED_GRPO_LENGTH_NORMALIZE_LOGPROBS}"
echo "[pure-drgrpo] wb_project=${WB_PROJECT}"
echo "[pure-drgrpo] wb_run_name=${WB_RUN_NAME}"
echo "[pure-drgrpo] prompt_data=${PROMPT_DATA}"
echo "[pure-drgrpo] eval_data=${EVAL_DATA}"
echo "[pure-drgrpo] eval_steps=${EVAL_STEPS}"
echo "[pure-drgrpo] eval_n=${EVAL_N} (pass@1)"
echo "[pure-drgrpo] train_batch_size=${TRAIN_BATCH_SIZE}"
echo "[pure-drgrpo] train_batch_size_per_device=${TRAIN_BATCH_SIZE_PER_DEVICE}"
echo "[pure-drgrpo] rollout_batch_size=${ROLLOUT_BATCH_SIZE}"
echo "[pure-drgrpo] rollout_batch_size_per_device=${ROLLOUT_BATCH_SIZE_PER_DEVICE}"
echo "[pure-drgrpo] pi_buffer_maxlen_per_device=${PI_BUFFER_MAXLEN_PER_DEVICE}"
echo "[pure-drgrpo] ${cmd[*]} $*"

cd "$PURE_DRGRPO_ROOT"
exec "${cmd[@]}" "$@"
