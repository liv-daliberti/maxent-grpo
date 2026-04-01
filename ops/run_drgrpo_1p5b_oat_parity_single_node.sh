#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

RECIPE="${GRPO_RECIPE:-$ROOT_DIR/configs/recipes/Qwen2.5-1.5B-Instruct/grpo/config_math_oat_parity_r1.yaml}"
ACCEL_CONFIG="${ACCEL_CONFIG:-$ROOT_DIR/configs/recipes/accelerate_configs/zero2_bf16.yaml}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-var/data/drgrpo_1p5b_oat_parity_trl_r1_${RUN_STAMP}}"
RUN_NAME="${RUN_NAME:-drgrpo-1p5b-oat-parity-trl-r1-${RUN_STAMP}}"
FLASH_ATTN_PROBE_TIMEOUT_S="${FLASH_ATTN_PROBE_TIMEOUT_S:-20}"
ATTN_IMPLEMENTATION_OVERRIDE="${ATTN_IMPLEMENTATION_OVERRIDE:-}"
SKIP_FLASH_ATTN_PROBE="${SKIP_FLASH_ATTN_PROBE:-0}"

export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export GRPO_RECIPE="$RECIPE"

# Closest TRL/vLLM runtime shape to the public OAT 1.5B command.
export MAXENT_VLLM_MODE=colocate
export MAXENT_VLLM_COLOCATE_SYNC="${MAXENT_VLLM_COLOCATE_SYNC:-1}"
export MAXENT_VLLM_COLOCATE_SYNC_INTERVAL="${MAXENT_VLLM_COLOCATE_SYNC_INTERVAL:-1}"
export MAXENT_VLLM_COLOCATE_GPU_UTIL="${MAXENT_VLLM_COLOCATE_GPU_UTIL:-0.35}"
export MAXENT_VLLM_COLOCATE_ENABLE_PREFIX_CACHING="${MAXENT_VLLM_COLOCATE_ENABLE_PREFIX_CACHING:-1}"
export MAXENT_VLLM_COLOCATE_ENABLE_SLEEP_MODE="${MAXENT_VLLM_COLOCATE_ENABLE_SLEEP_MODE:-1}"
export MAXENT_VLLM_COLOCATE_ENFORCE_EAGER="${MAXENT_VLLM_COLOCATE_ENFORCE_EAGER:-0}"
export MAXENT_VLLM_COLOCATE_ATTENTION_BACKEND="${MAXENT_VLLM_COLOCATE_ATTENTION_BACKEND:-FLASH_ATTN}"

extra_args=(
  --config "$RECIPE"
  --output_dir "$OUTPUT_DIR"
  --run_name "$RUN_NAME"
)

if [[ -n "$ATTN_IMPLEMENTATION_OVERRIDE" ]]; then
  extra_args+=(--attn_implementation "$ATTN_IMPLEMENTATION_OVERRIDE")
elif [[ "$SKIP_FLASH_ATTN_PROBE" == "1" ]]; then
  echo "[parity] skipping flash_attn probe; trainer will use the recipe/default attention path." >&2
elif timeout "${FLASH_ATTN_PROBE_TIMEOUT_S}s" python - <<'PY' >/dev/null 2>&1
import flash_attn  # noqa: F401
PY
then
  extra_args+=(--attn_implementation flash_attention_2)
else
  echo "[parity] flash_attn probe failed or timed out after ${FLASH_ATTN_PROBE_TIMEOUT_S}s; trainer will use the recipe default attention path." >&2
  echo "[parity] Set ATTN_IMPLEMENTATION_OVERRIDE=flash_attention_2 to force FA2, or SKIP_FLASH_ATTN_PROBE=1 to bypass this check entirely." >&2
fi

launch=(
  python -m accelerate.commands.launch
  --config_file "$ACCEL_CONFIG"
  --num_processes "$NUM_PROCESSES"
  "$ROOT_DIR/src/maxent_grpo/grpo.py"
)

echo "[parity] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[parity] recipe=${RECIPE}"
echo "[parity] accelerator=${ACCEL_CONFIG}"
echo "[parity] output_dir=${OUTPUT_DIR}"
echo "[parity] run_name=${RUN_NAME}"
echo "[parity] ${launch[*]} ${extra_args[*]} $*"

exec "${launch[@]}" "${extra_args[@]}" "$@"
