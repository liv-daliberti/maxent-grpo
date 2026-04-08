#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

CANONICAL_UPSTREAM_ROOT="${ROOT_DIR}/understand-r1-zero"
CANONICAL_PYTHON_BIN="${ROOT_DIR}/var/seed_paper_eval/paper310/bin/python"
CANONICAL_PYTHON_LIB_DIR="${ROOT_DIR}/var/seed_paper_eval/paper310/lib"

UPSTREAM_ROOT="${OAT_ZERO_UPSTREAM_ROOT:-$CANONICAL_UPSTREAM_ROOT}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SAVE_PATH="${SAVE_PATH:-$ROOT_DIR/var/data/oat_zero_exact_1p5b_${RUN_STAMP}}"
WB_PROJECT="${OAT_ZERO_WB_PROJECT:-oat-zero}"
WB_RUN_NAME="${OAT_ZERO_WB_RUN_NAME:-qwen2.5-Math-1.5b-r1-zero-exact-${RUN_STAMP}}"
USE_WB="${OAT_ZERO_USE_WB:-1}"
PROMPT_DATA="${OAT_ZERO_PROMPT_DATA:-$ROOT_DIR/datasets/train/math_12k}"
EVAL_DATA="${OAT_ZERO_EVAL_DATA:-$UPSTREAM_ROOT/datasets/evaluation_suite}"
PRETRAIN="${OAT_ZERO_PRETRAIN:-Qwen/Qwen2.5-Math-1.5B}"
VERIFIER_VERSION="${OAT_ZERO_VERIFIER_VERSION:-fast}"
INPUT_KEY="${OAT_ZERO_INPUT_KEY:-problem}"
OUTPUT_KEY="${OAT_ZERO_OUTPUT_KEY:-answer}"
N_GPU="${OAT_ZERO_N_GPU:-8}"
PROMPT_TEMPLATE="${OAT_ZERO_PROMPT_TEMPLATE:-r1}"
OBJECTIVE="${OAT_ZERO_OBJECTIVE:-grpo}"
COLLOCATE="${OAT_ZERO_COLLOCATE:-1}"
ZERO_STAGE="${OAT_ZERO_ZERO_STAGE:-2}"
ADAM_OFFLOAD="${OAT_ZERO_ADAM_OFFLOAD:-0}"
ACTIVATION_OFFLOADING="${OAT_ZERO_ACTIVATION_OFFLOADING:-0}"
DISABLE_TRACE_CACHE="${OAT_ZERO_DISABLE_TRACE_CACHE:-0}"
GRAD_ACCUM_DTYPE="${OAT_ZERO_GRAD_ACCUM_DTYPE:-}"
VLLM_SLEEP="${OAT_ZERO_VLLM_SLEEP:-1}"
LEARNING_RATE="${OAT_ZERO_LEARNING_RATE:-0.000001}"
NUM_PPO_EPOCHS="${OAT_ZERO_NUM_PPO_EPOCHS:-1}"
BETA="${OAT_ZERO_BETA:-0}"
NUM_PROMPT_EPOCH="${OAT_ZERO_NUM_PROMPT_EPOCH:-20}"
NUM_SAMPLES="${OAT_ZERO_NUM_SAMPLES:-8}"
TRAIN_BATCH_SIZE="${OAT_ZERO_TRAIN_BATCH_SIZE:-128}"
TRAIN_BATCH_SIZE_PER_DEVICE="${OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE:-1}"
ROLLOUT_BATCH_SIZE="${OAT_ZERO_ROLLOUT_BATCH_SIZE:-128}"
ROLLOUT_BATCH_SIZE_PER_DEVICE="${OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE:-16}"
PI_BUFFER_MAXLEN_PER_DEVICE="${OAT_ZERO_PI_BUFFER_MAXLEN_PER_DEVICE:-128}"
VLLM_GPU_RATIO="${OAT_ZERO_VLLM_GPU_RATIO:-0.35}"
EVAL_STEPS="${OAT_ZERO_EVAL_STEPS:-16}"
SAVE_STEPS="${OAT_ZERO_SAVE_STEPS:-$EVAL_STEPS}"
SAVE_FROM="${OAT_ZERO_SAVE_FROM:-0}"
SAVE_CKPT="${OAT_ZERO_SAVE_CKPT:-0}"
SAVE_INITIAL_MODEL="${OAT_ZERO_SAVE_INITIAL_MODEL:-1}"
ALLOW_NO_SAVE="${OAT_ZERO_ALLOW_NO_SAVE:-0}"
MAX_SAVE_NUM="${OAT_ZERO_MAX_SAVE_NUM:-999999}"
MAX_SAVE_MEM="${OAT_ZERO_MAX_SAVE_MEM:-99999999}"
ALLOW_NO_EVAL="${OAT_ZERO_ALLOW_NO_EVAL:-0}"
MAX_EVAL_STEPS_ALLOWED="${OAT_ZERO_MAX_EVAL_STEPS_ALLOWED:-1000}"
MAX_NORM="${OAT_ZERO_MAX_NORM:-1.0}"
IGNORE_NO_EOS="${OAT_ZERO_IGNORE_NO_EOS:-0}"
MAXENT_TAU="${OAT_ZERO_MAXENT_TAU:-0.5}"
MAXENT_Q_TEMPERATURE="${OAT_ZERO_MAXENT_Q_TEMPERATURE:-2.0}"
MAXENT_Q_EPSILON="${OAT_ZERO_MAXENT_Q_EPSILON:-1e-6}"
MAXENT_LOGPROB_CHUNK_SIZE="${OAT_ZERO_MAXENT_LOGPROB_CHUNK_SIZE:-2}"
MAXENT_BACKWARD_CHUNK_SIZE="${OAT_ZERO_MAXENT_BACKWARD_CHUNK_SIZE:-4}"
MAXENT_BACKWARD_TOKEN_BUDGET="${OAT_ZERO_MAXENT_BACKWARD_TOKEN_BUDGET:-8192}"
MAXENT_LENGTH_NORMALIZE_REF="${OAT_ZERO_MAXENT_LENGTH_NORMALIZE_REF:-1}"
MAXENT_LENGTH_NORMALIZE_POLICY="${OAT_ZERO_MAXENT_LENGTH_NORMALIZE_POLICY:-1}"
MAXENT_LISTWISE_SKIP_ZERO_VARIANCE_GROUPS="${OAT_ZERO_MAXENT_LISTWISE_SKIP_ZERO_VARIANCE_GROUPS:-1}"
MAXENT_USE_CLIP_OBJECTIVE="${OAT_ZERO_MAXENT_USE_CLIP_OBJECTIVE:-1}"
MAXENT_CLIP_OBJECTIVE_COEF="${OAT_ZERO_MAXENT_CLIP_OBJECTIVE_COEF:-1.0}"
MAXENT_CLIP_PRESERVE_REWARD_MASS="${OAT_ZERO_MAXENT_CLIP_PRESERVE_REWARD_MASS:-0}"
MAXENT_CLIP_MODE="${OAT_ZERO_MAXENT_CLIP_MODE:-sequence}"
MAXENT_TOKEN_SURROGATE_PRIMARY="${OAT_ZERO_MAXENT_TOKEN_SURROGATE_PRIMARY:-0}"
MAXENT_DRGRPO_TOKEN_PRIMARY="${OAT_ZERO_MAXENT_DRGRPO_TOKEN_PRIMARY:-0}"
MAXENT_SEQUENCE_AUX_COEF="${OAT_ZERO_MAXENT_SEQUENCE_AUX_COEF:-1.0}"
MAXENT_BRANCH_GRAD_DIAGNOSTICS="${OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS:-0}"
MAXENT_BRANCH_GRAD_DIAGNOSTICS_INTERVAL="${OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS_INTERVAL:-1}"
MAXENT_BRANCH_GRAD_DIAGNOSTICS_MAX_STEPS="${OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS_MAX_STEPS:-0}"
MAXENT_REFERENCE_LOGPROBS_SOURCE="${OAT_ZERO_MAXENT_REFERENCE_LOGPROBS_SOURCE:-model}"
MAXENT_TARGET_WEIGHT_ENTROPY="${OAT_ZERO_MAXENT_TARGET_WEIGHT_ENTROPY:-}"
MAXENT_TARGET_WEIGHT_ENTROPY_START="${OAT_ZERO_MAXENT_TARGET_WEIGHT_ENTROPY_START:-}"
MAXENT_TARGET_WEIGHT_ENTROPY_PEAK="${OAT_ZERO_MAXENT_TARGET_WEIGHT_ENTROPY_PEAK:-}"
MAXENT_TARGET_WEIGHT_ENTROPY_PEAK_STEP="${OAT_ZERO_MAXENT_TARGET_WEIGHT_ENTROPY_PEAK_STEP:-0}"
MAXENT_TARGET_WEIGHT_ENTROPY_FINAL="${OAT_ZERO_MAXENT_TARGET_WEIGHT_ENTROPY_FINAL:-}"
MAXENT_TARGET_WEIGHT_ENTROPY_HORIZON="${OAT_ZERO_MAXENT_TARGET_WEIGHT_ENTROPY_HORIZON:-0}"
MAXENT_TAU_LEARNABLE="${OAT_ZERO_MAXENT_TAU_LEARNABLE:-0}"
MAXENT_TAU_CONTROLLER_ENABLED="${OAT_ZERO_MAXENT_TAU_CONTROLLER_ENABLED:-1}"
MAXENT_TAU_LR="${OAT_ZERO_MAXENT_TAU_LR:-0.0}"
MAXENT_TAU_MIN="${OAT_ZERO_MAXENT_TAU_MIN:-0.0}"
MAXENT_TAU_MAX="${OAT_ZERO_MAXENT_TAU_MAX:-0.0}"
MAXENT_TAU_WARMUP_STEPS="${OAT_ZERO_MAXENT_TAU_WARMUP_STEPS:--1}"
MAXENT_BETA_CONTROLLER_ENABLED="${OAT_ZERO_MAXENT_BETA_CONTROLLER_ENABLED:-0}"
KL_TARGET="${OAT_ZERO_KL_TARGET:-0.0}"
KL_HORIZON="${OAT_ZERO_KL_HORIZON:-0}"
KL_CTL_STEP_SIZE="${OAT_ZERO_KL_CTL_STEP_SIZE:-0.0}"
FLASH_ATTN_VERSION="${OAT_ZERO_FLASH_ATTN_VERSION:-2.7.4.post1}"
FLASH_ATTN_MAX_JOBS="${OAT_ZERO_FLASH_ATTN_MAX_JOBS:-8}"
FLASH_ATTN_CACHE_ROOT="${OAT_ZERO_FLASH_ATTN_CACHE_ROOT:-$ROOT_DIR/var/cache/oat_zero_exact_flash_attn}"
ENABLE_FLASH_ATTN="${OAT_ZERO_ENABLE_FLASH_ATTN:-1}"
SMOKE_MODE="${OAT_ZERO_SMOKE:-0}"
SMOKE_SUCCESS_PATTERN="${OAT_ZERO_SMOKE_SUCCESS_PATTERN:-weights @version=0 broadcasted to actors}"
SMOKE_TIMEOUT_SECONDS="${OAT_ZERO_SMOKE_TIMEOUT_SECONDS:-3000}"
BOOTSTRAP_ONLY="${OAT_ZERO_BOOTSTRAP_ONLY:-0}"
SHM_SIZE_MB="${OAT_ZERO_SHM_SIZE_MB:-20000}"
EXPECTED_PYTHON_PREFIX="${OAT_ZERO_EXPECTED_PYTHON_PREFIX:-3.10.}"
EXPECTED_TORCH_VERSION="${OAT_ZERO_EXPECTED_TORCH_VERSION:-2.6.0}"
EXPECTED_TRANSFORMERS_VERSION="${OAT_ZERO_EXPECTED_TRANSFORMERS_VERSION:-4.51.3}"
EXPECTED_VLLM_VERSION="${OAT_ZERO_EXPECTED_VLLM_VERSION:-0.8.4}"
EXPECTED_OAT_VERSION="${OAT_ZERO_EXPECTED_OAT_VERSION:-0.1.3.post1}"
EXPECTED_DEEPSPEED_VERSION="${OAT_ZERO_EXPECTED_DEEPSPEED_VERSION:-0.16.8}"
EXPECTED_MATH_VERIFY_VERSION="${OAT_ZERO_EXPECTED_MATH_VERIFY_VERSION:-0.7.0}"
EXPECTED_FIRE_VERSION="${OAT_ZERO_EXPECTED_FIRE_VERSION:-0.7.0}"

choose_local_root() {
  if [[ -n "${OAT_ZERO_LOCAL_ROOT:-}" ]]; then
    printf '%s\n' "$OAT_ZERO_LOCAL_ROOT"
    return
  fi
  if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    printf '%s\n' "${SLURM_TMPDIR%/}/maxent-grpo-oat-zero"
    return
  fi
  printf '/tmp/%s/maxent-grpo-oat-zero\n' "${USER:-unknown}"
}

LOCAL_ROOT="$(choose_local_root)"
PYTHON_BIN="${OAT_ZERO_PYTHON:-$CANONICAL_PYTHON_BIN}"
PYTHON_LIB_DIR="${OAT_ZERO_PYTHON_LIB_DIR:-$CANONICAL_PYTHON_LIB_DIR}"
LOCAL_CACHE_ROOT="${OAT_ZERO_LOCAL_CACHE_ROOT:-$LOCAL_ROOT/cache}"
LOCAL_JOB_ROOT="${OAT_ZERO_LOCAL_JOB_ROOT:-$LOCAL_ROOT/job_${SLURM_JOB_ID:-manual}}"

if [[ ! -f "$UPSTREAM_ROOT/train_zero_math.py" ]]; then
  echo "Missing canonical upstream checkout at $UPSTREAM_ROOT" >&2
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

case "$OBJECTIVE" in
  grpo|maxent_listwise) ;;
  *)
    echo "OAT_ZERO_OBJECTIVE must be one of: grpo, maxent_listwise" >&2
    exit 1
    ;;
esac

case "$MAXENT_REFERENCE_LOGPROBS_SOURCE" in
  model|behavior) ;;
  *)
    echo "OAT_ZERO_MAXENT_REFERENCE_LOGPROBS_SOURCE must be one of: model, behavior" >&2
    exit 1
    ;;
esac

case "$MAXENT_BETA_CONTROLLER_ENABLED" in
  0|1) ;;
  *)
    echo "OAT_ZERO_MAXENT_BETA_CONTROLLER_ENABLED must be 0 or 1" >&2
    exit 1
    ;;
esac

case "$IGNORE_NO_EOS" in
  0|1) ;;
  *)
    echo "OAT_ZERO_IGNORE_NO_EOS must be 0 or 1" >&2
    exit 1
    ;;
esac

case "$MAXENT_TAU_LEARNABLE" in
  0|1) ;;
  *)
    echo "OAT_ZERO_MAXENT_TAU_LEARNABLE must be 0 or 1" >&2
    exit 1
    ;;
esac

case "$MAXENT_TAU_CONTROLLER_ENABLED" in
  0|1) ;;
  *)
    echo "OAT_ZERO_MAXENT_TAU_CONTROLLER_ENABLED must be 0 or 1" >&2
    exit 1
    ;;
esac

case "$MAXENT_BRANCH_GRAD_DIAGNOSTICS" in
  0|1) ;;
  *)
    echo "OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS must be 0 or 1" >&2
    exit 1
    ;;
esac

case "$MAXENT_CLIP_MODE" in
  sequence|token|none) ;;
  *)
    echo "OAT_ZERO_MAXENT_CLIP_MODE must be one of: sequence, token, none" >&2
    exit 1
    ;;
esac

case "$ALLOW_NO_EVAL" in
  0|1) ;;
  *)
    echo "OAT_ZERO_ALLOW_NO_EVAL must be 0 or 1" >&2
    exit 1
    ;;
esac

case "$SAVE_CKPT" in
  0|1) ;;
  *)
    echo "OAT_ZERO_SAVE_CKPT must be 0 or 1" >&2
    exit 1
    ;;
esac

case "$SAVE_INITIAL_MODEL" in
  0|1) ;;
  *)
    echo "OAT_ZERO_SAVE_INITIAL_MODEL must be 0 or 1" >&2
    exit 1
    ;;
esac

case "$ALLOW_NO_SAVE" in
  0|1) ;;
  *)
    echo "OAT_ZERO_ALLOW_NO_SAVE must be 0 or 1" >&2
    exit 1
    ;;
esac

case "$USE_WB" in
  0|1) ;;
  *)
    echo "OAT_ZERO_USE_WB must be 0 or 1" >&2
    exit 1
    ;;
esac

case "$COLLOCATE" in
  0|1) ;;
  *)
    echo "OAT_ZERO_COLLOCATE must be 0 or 1" >&2
    exit 1
    ;;
esac

case "$ZERO_STAGE" in
  0|1|2|3) ;;
  *)
    echo "OAT_ZERO_ZERO_STAGE must be 0, 1, 2, or 3" >&2
    exit 1
    ;;
esac

case "$ADAM_OFFLOAD" in
  0|1) ;;
  *)
    echo "OAT_ZERO_ADAM_OFFLOAD must be 0 or 1" >&2
    exit 1
    ;;
esac

case "$ACTIVATION_OFFLOADING" in
  0|1) ;;
  *)
    echo "OAT_ZERO_ACTIVATION_OFFLOADING must be 0 or 1" >&2
    exit 1
    ;;
esac

case "$DISABLE_TRACE_CACHE" in
  0|1) ;;
  *)
    echo "OAT_ZERO_DISABLE_TRACE_CACHE must be 0 or 1" >&2
    exit 1
    ;;
esac

case "$VLLM_SLEEP" in
  0|1) ;;
  *)
    echo "OAT_ZERO_VLLM_SLEEP must be 0 or 1" >&2
    exit 1
    ;;
esac

case "$EVAL_STEPS" in
  ''|*[!0-9]*)
    echo "OAT_ZERO_EVAL_STEPS must be a positive integer" >&2
    exit 1
    ;;
esac

case "$MAX_EVAL_STEPS_ALLOWED" in
  ''|*[!0-9]*)
    echo "OAT_ZERO_MAX_EVAL_STEPS_ALLOWED must be a positive integer" >&2
    exit 1
    ;;
esac

if (( EVAL_STEPS == 0 )) && [[ "$ALLOW_NO_EVAL" != "1" ]]; then
  echo "Eval must stay enabled: OAT_ZERO_EVAL_STEPS must be > 0. To override intentionally, set OAT_ZERO_ALLOW_NO_EVAL=1." >&2
  exit 1
fi

if (( EVAL_STEPS > MAX_EVAL_STEPS_ALLOWED )) && [[ "$ALLOW_NO_EVAL" != "1" ]]; then
  echo "Eval is effectively disabled: OAT_ZERO_EVAL_STEPS=${EVAL_STEPS} exceeds OAT_ZERO_MAX_EVAL_STEPS_ALLOWED=${MAX_EVAL_STEPS_ALLOWED}. To override intentionally, set OAT_ZERO_ALLOW_NO_EVAL=1." >&2
  exit 1
fi

if ! [[ "$SAVE_STEPS" =~ ^-?[0-9]+$ ]]; then
  echo "OAT_ZERO_SAVE_STEPS must be an integer" >&2
  exit 1
fi

if ! [[ "$SAVE_FROM" =~ ^[0-9]+$ ]]; then
  echo "OAT_ZERO_SAVE_FROM must be a non-negative integer" >&2
  exit 1
fi

if ! [[ "$MAX_SAVE_NUM" =~ ^[0-9]+$ ]] || (( MAX_SAVE_NUM <= 0 )); then
  echo "OAT_ZERO_MAX_SAVE_NUM must be a positive integer" >&2
  exit 1
fi

if ! [[ "$MAX_SAVE_MEM" =~ ^[0-9]+$ ]] || (( MAX_SAVE_MEM <= 0 )); then
  echo "OAT_ZERO_MAX_SAVE_MEM must be a positive integer number of GB" >&2
  exit 1
fi

if ! [[ "$MAXENT_BRANCH_GRAD_DIAGNOSTICS_INTERVAL" =~ ^[0-9]+$ ]] || (( MAXENT_BRANCH_GRAD_DIAGNOSTICS_INTERVAL <= 0 )); then
  echo "OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS_INTERVAL must be a positive integer" >&2
  exit 1
fi

if ! [[ "$MAXENT_BRANCH_GRAD_DIAGNOSTICS_MAX_STEPS" =~ ^[0-9]+$ ]]; then
  echo "OAT_ZERO_MAXENT_BRANCH_GRAD_DIAGNOSTICS_MAX_STEPS must be a non-negative integer" >&2
  exit 1
fi

if (( SAVE_STEPS <= 0 )) && [[ "$ALLOW_NO_SAVE" != "1" ]]; then
  echo "Checkpoint saving must stay enabled: OAT_ZERO_SAVE_STEPS must be > 0. To override intentionally, set OAT_ZERO_ALLOW_NO_SAVE=1." >&2
  exit 1
fi

if [[ "$OBJECTIVE" == "maxent_listwise" ]]; then
  if (( NUM_SAMPLES <= 1 )); then
    echo "Listwise MaxEnt requires OAT_ZERO_NUM_SAMPLES > 1" >&2
    exit 1
  fi
  if (( TRAIN_BATCH_SIZE_PER_DEVICE <= 0 )); then
    echo "Listwise MaxEnt requires OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE > 0" >&2
    exit 1
  fi
  if (( TRAIN_BATCH_SIZE_PER_DEVICE % NUM_SAMPLES != 0 )); then
    echo "Listwise MaxEnt requires OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE to be divisible by OAT_ZERO_NUM_SAMPLES" >&2
    exit 1
  fi
fi

mkdir -p \
  "$SAVE_PATH" \
  "$ROOT_DIR/var/artifacts/logs" \
  "$ROOT_DIR/var/cache" \
  "$ROOT_DIR/var/tmp" \
  "$LOCAL_ROOT" \
  "$LOCAL_CACHE_ROOT" \
  "$LOCAL_JOB_ROOT"

runtime_probe() {
  local python_bin="$1"
  "$python_bin" - <<'PY'
import importlib.metadata as md
import json
import sys

packages = [
    "torch",
    "transformers",
    "vllm",
    "oat-llm",
    "deepspeed",
    "flash-attn",
    "math-verify",
    "fire",
]
data = {"python": sys.version.split()[0]}
for name in packages:
    try:
        data[name] = md.version(name)
    except Exception:
        data[name] = "MISSING"
print(json.dumps(data, sort_keys=True))
PY
}

runtime_matches_expected() {
  local python_bin="$1"
  [[ -x "$python_bin" ]] || return 1
  OAT_ZERO_EXPECTED_PYTHON_PREFIX="$EXPECTED_PYTHON_PREFIX" \
  OAT_ZERO_EXPECTED_TORCH_VERSION="$EXPECTED_TORCH_VERSION" \
  OAT_ZERO_EXPECTED_TRANSFORMERS_VERSION="$EXPECTED_TRANSFORMERS_VERSION" \
  OAT_ZERO_EXPECTED_VLLM_VERSION="$EXPECTED_VLLM_VERSION" \
  OAT_ZERO_EXPECTED_OAT_VERSION="$EXPECTED_OAT_VERSION" \
  OAT_ZERO_EXPECTED_DEEPSPEED_VERSION="$EXPECTED_DEEPSPEED_VERSION" \
  OAT_ZERO_EXPECTED_MATH_VERIFY_VERSION="$EXPECTED_MATH_VERIFY_VERSION" \
  OAT_ZERO_EXPECTED_FIRE_VERSION="$EXPECTED_FIRE_VERSION" \
  "$python_bin" - <<'PY'
import importlib.metadata as md
import os
import sys

expected = {
    "torch": os.environ["OAT_ZERO_EXPECTED_TORCH_VERSION"],
    "transformers": os.environ["OAT_ZERO_EXPECTED_TRANSFORMERS_VERSION"],
    "vllm": os.environ["OAT_ZERO_EXPECTED_VLLM_VERSION"],
    "oat-llm": os.environ["OAT_ZERO_EXPECTED_OAT_VERSION"],
    "deepspeed": os.environ["OAT_ZERO_EXPECTED_DEEPSPEED_VERSION"],
    "math-verify": os.environ["OAT_ZERO_EXPECTED_MATH_VERIFY_VERSION"],
    "fire": os.environ["OAT_ZERO_EXPECTED_FIRE_VERSION"],
}
python_prefix = os.environ["OAT_ZERO_EXPECTED_PYTHON_PREFIX"]
if not sys.version.startswith(python_prefix):
    raise SystemExit(1)
for name, version in expected.items():
    if md.version(name) != version:
        raise SystemExit(1)
PY
}

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing canonical OAT python at $PYTHON_BIN" >&2
  exit 1
fi

if ! runtime_matches_expected "$PYTHON_BIN"; then
  echo "[oat-zero-exact] runtime mismatch: expected the canonical README-flash paper310 stack." >&2
  runtime_probe "$PYTHON_BIN" >&2
  exit 1
fi

export PATH="$(dirname "$PYTHON_BIN"):${PATH}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TRANSFORMERS_NO_TF="${TRANSFORMERS_NO_TF:-1}"
export USE_TF="${USE_TF:-0}"
export USE_FLAX="${USE_FLAX:-0}"
export TMPDIR="${TMPDIR:-$LOCAL_JOB_ROOT/tmp}"
export TMP="${TMP:-$TMPDIR}"
export TEMP="${TEMP:-$TMPDIR}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$LOCAL_CACHE_ROOT/triton_${SLURM_JOB_ID:-manual}}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$LOCAL_CACHE_ROOT/torch_extensions_${SLURM_JOB_ID:-manual}}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$LOCAL_CACHE_ROOT/torchinductor_${SLURM_JOB_ID:-manual}}"
export XDG_CACHE_HOME="${LOCAL_CACHE_ROOT}/xdg_${SLURM_JOB_ID:-manual}"
export PIP_CACHE_DIR="${LOCAL_CACHE_ROOT}/pip_${SLURM_JOB_ID:-manual}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"
mkdir -p \
  "$TMPDIR" \
  "$TRITON_CACHE_DIR" \
  "$TORCH_EXTENSIONS_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$XDG_CACHE_HOME" \
  "$PIP_CACHE_DIR"

if [[ -d "$PYTHON_LIB_DIR" ]]; then
  export LD_LIBRARY_PATH="${PYTHON_LIB_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

echo "[oat-zero-exact] probing python libdir"
python_libdir="$("$PYTHON_BIN" - <<'PY' | tail -n 1 | tr -d '\r'
import sysconfig

print(sysconfig.get_config_var("LIBDIR") or "")
PY
)"
if [[ -n "$python_libdir" ]]; then
  export LD_LIBRARY_PATH="${python_libdir}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

echo "[oat-zero-exact] probing torch cuda version"
torch_cuda_version="$("$PYTHON_BIN" - <<'PY' | tail -n 1 | tr -d '\r'
import torch

print(torch.version.cuda or "")
PY
)"
echo "[oat-zero-exact] torch_cuda_version=${torch_cuda_version:-cpu}"

if [[ -z "${CUDA_HOME:-}" ]]; then
  candidate_nvcc=""
  if [[ -n "${OAT_ZERO_CUDA_HOME:-}" && -x "${OAT_ZERO_CUDA_HOME}/bin/nvcc" ]]; then
    candidate_nvcc="${OAT_ZERO_CUDA_HOME}/bin/nvcc"
  fi
  if [[ -z "$candidate_nvcc" ]] && [[ -n "$torch_cuda_version" ]] && [[ -x "/usr/local/cuda-${torch_cuda_version}/bin/nvcc" ]]; then
    candidate_nvcc="/usr/local/cuda-${torch_cuda_version}/bin/nvcc"
  fi
  if [[ -z "$candidate_nvcc" ]] && command -v nvcc >/dev/null 2>&1; then
    candidate_nvcc="$(command -v nvcc)"
  fi
  if [[ -z "$candidate_nvcc" ]]; then
    for fallback_nvcc in \
      /usr/local/cuda-12.4/bin/nvcc \
      /usr/local/cuda/bin/nvcc \
      /usr/local/cuda-13.1/bin/nvcc
    do
      if [[ -x "$fallback_nvcc" ]]; then
        candidate_nvcc="$fallback_nvcc"
        break
      fi
    done
  fi
  if [[ -n "$candidate_nvcc" ]]; then
    export CUDA_HOME="$(cd "$(dirname "$candidate_nvcc")/.." && pwd)"
  fi
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  export CUDACXX="$CUDA_HOME/bin/nvcc"
  export PATH="$CUDA_HOME/bin:${PATH}"
  if [[ -d "$CUDA_HOME/lib64" ]]; then
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
fi

echo "[oat-zero-exact] probing torch runtime key"
flash_attn_runtime_key="$("$PYTHON_BIN" - <<'PY' | tail -n 1 | tr -d '\r'
import re
import sys

import torch

torch_version = re.sub(r"[^0-9A-Za-z._-]+", "_", torch.__version__)
cuda_version = re.sub(r"[^0-9A-Za-z._-]+", "_", torch.version.cuda or "cpu")
print(f"py{sys.version_info.major}{sys.version_info.minor}_torch{torch_version}_cu{cuda_version}")
PY
)"
FLASH_ATTN_OVERLAY_DIR="${OAT_ZERO_FLASH_ATTN_OVERLAY_DIR:-$FLASH_ATTN_CACHE_ROOT/$flash_attn_runtime_key}"

prepend_pythonpath() {
  local new_path="$1"
  if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${new_path}:$PYTHONPATH"
  else
    export PYTHONPATH="${new_path}"
  fi
}

prepend_pythonpath "$UPSTREAM_ROOT"

flash_attn_importable() {
  "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import flash_attn
PY
}

flash_attn_location() {
  "$PYTHON_BIN" - <<'PY' | tail -n 1 | tr -d '\r'
import flash_attn

print(flash_attn.__file__)
PY
}

bootstrap_flash_attn() {
  local install_lock="$FLASH_ATTN_OVERLAY_DIR.install.lock"

  mkdir -p "$FLASH_ATTN_OVERLAY_DIR"
  prepend_pythonpath "$FLASH_ATTN_OVERLAY_DIR"

  if flash_attn_importable; then
    return 0
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
    echo "[oat-zero-exact] flash-attn bootstrap requires a visible GPU." >&2
    return 1
  fi

  if [[ -z "${CUDA_HOME:-}" || ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
    echo "[oat-zero-exact] missing CUDA_HOME nvcc for flash-attn build." >&2
    return 1
  fi

  exec 9>"$install_lock"
  flock 9

  if flash_attn_importable; then
    return 0
  fi

  echo "[oat-zero-exact] flash-attn build using CUDA_HOME=${CUDA_HOME}"
  "${CUDA_HOME}/bin/nvcc" --version | tail -n 1
  echo "[oat-zero-exact] installing flash-attn==${FLASH_ATTN_VERSION} into ${FLASH_ATTN_OVERLAY_DIR}"
  MAX_JOBS="$FLASH_ATTN_MAX_JOBS" \
    "$PYTHON_BIN" -m pip install \
      --upgrade \
      --no-build-isolation \
      --no-deps \
      --target "$FLASH_ATTN_OVERLAY_DIR" \
      "flash-attn==${FLASH_ATTN_VERSION}"

  flash_attn_importable
}

if [[ "$ENABLE_FLASH_ATTN" == "1" ]]; then
  echo "[oat-zero-exact] checking flash-attn availability"
  if ! flash_attn_importable; then
    bootstrap_flash_attn
  fi
  flash_attn_path="$(flash_attn_location)"
  flash_attn_flag=(--flash-attn)
else
  echo "[oat-zero-exact] flash-attn disabled"
  flash_attn_path="disabled"
  flash_attn_flag=(--no-flash-attn)
fi

echo "[oat-zero-exact] host=$(hostname)"
echo "[oat-zero-exact] upstream_root=${UPSTREAM_ROOT}"
echo "[oat-zero-exact] python=${PYTHON_BIN}"
echo "[oat-zero-exact] runtime_probe=$(runtime_probe "$PYTHON_BIN")"
echo "[oat-zero-exact] local_root=${LOCAL_ROOT}"
echo "[oat-zero-exact] local_job_root=${LOCAL_JOB_ROOT}"
echo "[oat-zero-exact] save_path=${SAVE_PATH}"
echo "[oat-zero-exact] wb_project=${WB_PROJECT}"
echo "[oat-zero-exact] wb_run_name=${WB_RUN_NAME}"
echo "[oat-zero-exact] use_wb=${USE_WB}"
echo "[oat-zero-exact] prompt_data=${PROMPT_DATA}"
echo "[oat-zero-exact] eval_data=${EVAL_DATA}"
echo "[oat-zero-exact] pretrain=${PRETRAIN}"
echo "[oat-zero-exact] verifier_version=${VERIFIER_VERSION}"
echo "[oat-zero-exact] input_key=${INPUT_KEY}"
echo "[oat-zero-exact] output_key=${OUTPUT_KEY}"
echo "[oat-zero-exact] prompt_template=${PROMPT_TEMPLATE}"
echo "[oat-zero-exact] objective=${OBJECTIVE}"
echo "[oat-zero-exact] collocate=${COLLOCATE}"
echo "[oat-zero-exact] zero_stage=${ZERO_STAGE}"
echo "[oat-zero-exact] adam_offload=${ADAM_OFFLOAD}"
echo "[oat-zero-exact] activation_offloading=${ACTIVATION_OFFLOADING}"
echo "[oat-zero-exact] disable_trace_cache=${DISABLE_TRACE_CACHE}"
echo "[oat-zero-exact] grad_accum_dtype=${GRAD_ACCUM_DTYPE:-<unset>}"
echo "[oat-zero-exact] vllm_sleep=${VLLM_SLEEP}"
echo "[oat-zero-exact] beta=${BETA}"
echo "[oat-zero-exact] num_samples=${NUM_SAMPLES}"
echo "[oat-zero-exact] train_batch_size=${TRAIN_BATCH_SIZE}"
echo "[oat-zero-exact] train_batch_size_per_device=${TRAIN_BATCH_SIZE_PER_DEVICE}"
echo "[oat-zero-exact] rollout_batch_size=${ROLLOUT_BATCH_SIZE}"
echo "[oat-zero-exact] rollout_batch_size_per_device=${ROLLOUT_BATCH_SIZE_PER_DEVICE}"
echo "[oat-zero-exact] pi_buffer_maxlen_per_device=${PI_BUFFER_MAXLEN_PER_DEVICE}"
echo "[oat-zero-exact] vllm_gpu_ratio=${VLLM_GPU_RATIO}"
echo "[oat-zero-exact] max_norm=${MAX_NORM}"
echo "[oat-zero-exact] pytorch_cuda_alloc_conf=${PYTORCH_CUDA_ALLOC_CONF:-<unset>}"
echo "[oat-zero-exact] eval_steps=${EVAL_STEPS}"
echo "[oat-zero-exact] save_steps=${SAVE_STEPS}"
echo "[oat-zero-exact] save_from=${SAVE_FROM}"
echo "[oat-zero-exact] save_ckpt=${SAVE_CKPT}"
echo "[oat-zero-exact] save_initial_model=${SAVE_INITIAL_MODEL}"
echo "[oat-zero-exact] allow_no_save=${ALLOW_NO_SAVE}"
echo "[oat-zero-exact] max_save_num=${MAX_SAVE_NUM}"
echo "[oat-zero-exact] max_save_mem=${MAX_SAVE_MEM}"
echo "[oat-zero-exact] allow_no_eval=${ALLOW_NO_EVAL}"
echo "[oat-zero-exact] max_eval_steps_allowed=${MAX_EVAL_STEPS_ALLOWED}"
echo "[oat-zero-exact] maxent_tau=${MAXENT_TAU}"
echo "[oat-zero-exact] maxent_tau_learnable=${MAXENT_TAU_LEARNABLE}"
echo "[oat-zero-exact] maxent_tau_controller_enabled=${MAXENT_TAU_CONTROLLER_ENABLED}"
echo "[oat-zero-exact] ignore_no_eos=${IGNORE_NO_EOS}"
echo "[oat-zero-exact] maxent_target_weight_entropy=${MAXENT_TARGET_WEIGHT_ENTROPY:-<unset>}"
echo "[oat-zero-exact] maxent_target_weight_entropy_start=${MAXENT_TARGET_WEIGHT_ENTROPY_START:-<unset>}"
echo "[oat-zero-exact] maxent_target_weight_entropy_peak=${MAXENT_TARGET_WEIGHT_ENTROPY_PEAK:-<unset>}"
echo "[oat-zero-exact] maxent_target_weight_entropy_peak_step=${MAXENT_TARGET_WEIGHT_ENTROPY_PEAK_STEP}"
echo "[oat-zero-exact] maxent_target_weight_entropy_final=${MAXENT_TARGET_WEIGHT_ENTROPY_FINAL:-<unset>}"
echo "[oat-zero-exact] maxent_target_weight_entropy_horizon=${MAXENT_TARGET_WEIGHT_ENTROPY_HORIZON}"
echo "[oat-zero-exact] maxent_q_temperature=${MAXENT_Q_TEMPERATURE}"
echo "[oat-zero-exact] maxent_q_epsilon=${MAXENT_Q_EPSILON}"
echo "[oat-zero-exact] maxent_logprob_chunk_size=${MAXENT_LOGPROB_CHUNK_SIZE}"
echo "[oat-zero-exact] maxent_backward_chunk_size=${MAXENT_BACKWARD_CHUNK_SIZE}"
echo "[oat-zero-exact] maxent_backward_token_budget=${MAXENT_BACKWARD_TOKEN_BUDGET}"
echo "[oat-zero-exact] maxent_length_normalize_ref=${MAXENT_LENGTH_NORMALIZE_REF}"
echo "[oat-zero-exact] maxent_length_normalize_policy=${MAXENT_LENGTH_NORMALIZE_POLICY}"
echo "[oat-zero-exact] maxent_listwise_skip_zero_variance_groups=${MAXENT_LISTWISE_SKIP_ZERO_VARIANCE_GROUPS}"
echo "[oat-zero-exact] maxent_use_clip_objective=${MAXENT_USE_CLIP_OBJECTIVE}"
echo "[oat-zero-exact] maxent_clip_objective_coef=${MAXENT_CLIP_OBJECTIVE_COEF}"
echo "[oat-zero-exact] maxent_clip_preserve_reward_mass=${MAXENT_CLIP_PRESERVE_REWARD_MASS}"
echo "[oat-zero-exact] maxent_clip_mode=${MAXENT_CLIP_MODE}"
echo "[oat-zero-exact] maxent_token_surrogate_primary=${MAXENT_TOKEN_SURROGATE_PRIMARY}"
echo "[oat-zero-exact] maxent_drgrpo_token_primary=${MAXENT_DRGRPO_TOKEN_PRIMARY}"
echo "[oat-zero-exact] maxent_sequence_aux_coef=${MAXENT_SEQUENCE_AUX_COEF}"
echo "[oat-zero-exact] maxent_branch_grad_diagnostics=${MAXENT_BRANCH_GRAD_DIAGNOSTICS}"
echo "[oat-zero-exact] maxent_branch_grad_diagnostics_interval=${MAXENT_BRANCH_GRAD_DIAGNOSTICS_INTERVAL}"
echo "[oat-zero-exact] maxent_branch_grad_diagnostics_max_steps=${MAXENT_BRANCH_GRAD_DIAGNOSTICS_MAX_STEPS}"
echo "[oat-zero-exact] maxent_reference_logprobs_source=${MAXENT_REFERENCE_LOGPROBS_SOURCE}"
echo "[oat-zero-exact] shm_size_mb=${SHM_SIZE_MB}"
echo "[oat-zero-exact] cuda_home=${CUDA_HOME:-unset}"
echo "[oat-zero-exact] enable_flash_attn=${ENABLE_FLASH_ATTN}"
echo "[oat-zero-exact] flash_attn_path=${flash_attn_path}"
echo "[oat-zero-exact] smoke_mode=${SMOKE_MODE}"
echo "[oat-zero-exact] bootstrap_only=${BOOTSTRAP_ONLY}"

if [[ "$BOOTSTRAP_ONLY" == "1" ]]; then
  exit 0
fi

save_initial_model_snapshot() {
  local save_dir="$SAVE_PATH/saved_models/step_00000"
  if [[ -f "$save_dir/config.json" ]]; then
    echo "[oat-zero-exact] initial model snapshot already exists at $save_dir"
    return 0
  fi

  echo "[oat-zero-exact] saving initial model snapshot to $save_dir"
  PRETRAIN="$PRETRAIN" INITIAL_SAVE_DIR="$save_dir" "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

pretrain = os.environ["PRETRAIN"]
save_dir = Path(os.environ["INITIAL_SAVE_DIR"])
save_dir.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(pretrain)
model = AutoModelForCausalLM.from_pretrained(
    pretrain,
    low_cpu_mem_usage=True,
    torch_dtype="auto",
)
model.save_pretrained(save_dir, safe_serialization=True)
tokenizer.save_pretrained(save_dir)
print(save_dir)
PY
}

if [[ "$SAVE_INITIAL_MODEL" == "1" ]]; then
  save_initial_model_snapshot
fi

maxent_length_normalize_ref_flag=(--no-maxent-length-normalize-ref)
if [[ "$MAXENT_LENGTH_NORMALIZE_REF" == "1" ]]; then
  maxent_length_normalize_ref_flag=(--maxent-length-normalize-ref)
fi

maxent_length_normalize_policy_flag=(--no-maxent-length-normalize-policy)
if [[ "$MAXENT_LENGTH_NORMALIZE_POLICY" == "1" ]]; then
  maxent_length_normalize_policy_flag=(--maxent-length-normalize-policy)
fi

maxent_skip_zero_variance_flag=(--no-maxent-listwise-skip-zero-variance-groups)
if [[ "$MAXENT_LISTWISE_SKIP_ZERO_VARIANCE_GROUPS" == "1" ]]; then
  maxent_skip_zero_variance_flag=(--maxent-listwise-skip-zero-variance-groups)
fi

maxent_use_clip_objective_flag=(--no-maxent-use-clip-objective)
if [[ "$MAXENT_USE_CLIP_OBJECTIVE" == "1" ]]; then
  maxent_use_clip_objective_flag=(--maxent-use-clip-objective)
fi

maxent_clip_preserve_reward_mass_flag=(--no-maxent-clip-preserve-reward-mass)
if [[ "$MAXENT_CLIP_PRESERVE_REWARD_MASS" == "1" ]]; then
  maxent_clip_preserve_reward_mass_flag=(--maxent-clip-preserve-reward-mass)
fi

maxent_token_surrogate_primary_flag=(--no-maxent-token-surrogate-primary)
if [[ "$MAXENT_TOKEN_SURROGATE_PRIMARY" == "1" ]]; then
  maxent_token_surrogate_primary_flag=(--maxent-token-surrogate-primary)
fi

maxent_drgrpo_token_primary_flag=(--no-maxent-drgrpo-token-primary)
if [[ "$MAXENT_DRGRPO_TOKEN_PRIMARY" == "1" ]]; then
  maxent_drgrpo_token_primary_flag=(--maxent-drgrpo-token-primary)
fi

maxent_branch_grad_diagnostics_flag=(--no-maxent-branch-grad-diagnostics)
if [[ "$MAXENT_BRANCH_GRAD_DIAGNOSTICS" == "1" ]]; then
  maxent_branch_grad_diagnostics_flag=(--maxent-branch-grad-diagnostics)
fi

maxent_beta_controller_flag=(--no-maxent-beta-controller-enabled)
if [[ "$MAXENT_BETA_CONTROLLER_ENABLED" == "1" ]]; then
  maxent_beta_controller_flag=(--maxent-beta-controller-enabled)
fi

maxent_tau_learnable_flag=(--no-maxent-tau-learnable)
if [[ "$MAXENT_TAU_LEARNABLE" == "1" ]]; then
  maxent_tau_learnable_flag=(--maxent-tau-learnable)
fi

maxent_tau_controller_flag=(--no-maxent-tau-controller-enabled)
if [[ "$MAXENT_TAU_CONTROLLER_ENABLED" == "1" ]]; then
  maxent_tau_controller_flag=(--maxent-tau-controller-enabled)
fi

ignore_no_eos_flag=(--no-ignore-no-eos)
if [[ "$IGNORE_NO_EOS" == "1" ]]; then
  ignore_no_eos_flag=(--ignore-no-eos)
fi

objective_args=(
  --objective "$OBJECTIVE"
  --beta "$BETA"
)

if [[ "$OBJECTIVE" == "maxent_listwise" ]]; then
  objective_args+=(
    --maxent-tau "$MAXENT_TAU"
    "${maxent_tau_learnable_flag[@]}"
    "${maxent_tau_controller_flag[@]}"
    --maxent-q-temperature "$MAXENT_Q_TEMPERATURE"
    --maxent-q-epsilon "$MAXENT_Q_EPSILON"
    --maxent-clip-objective-coef "$MAXENT_CLIP_OBJECTIVE_COEF"
    --maxent-clip-mode "$MAXENT_CLIP_MODE"
    --maxent-reference-logprobs-source "$MAXENT_REFERENCE_LOGPROBS_SOURCE"
    --maxent-tau-lr "$MAXENT_TAU_LR"
    --maxent-tau-min "$MAXENT_TAU_MIN"
    --maxent-tau-max "$MAXENT_TAU_MAX"
    --maxent-tau-warmup-steps "$MAXENT_TAU_WARMUP_STEPS"
    --maxent-target-weight-entropy-horizon "$MAXENT_TARGET_WEIGHT_ENTROPY_HORIZON"
    --kl_target "$KL_TARGET"
    --kl_horizon "$KL_HORIZON"
    --kl_ctl_step_size "$KL_CTL_STEP_SIZE"
    "${maxent_length_normalize_ref_flag[@]}"
    "${maxent_length_normalize_policy_flag[@]}"
    "${maxent_skip_zero_variance_flag[@]}"
    "${maxent_use_clip_objective_flag[@]}"
    "${maxent_clip_preserve_reward_mass_flag[@]}"
    "${maxent_token_surrogate_primary_flag[@]}"
    "${maxent_drgrpo_token_primary_flag[@]}"
    --maxent-sequence-aux-coef "$MAXENT_SEQUENCE_AUX_COEF"
    "${maxent_branch_grad_diagnostics_flag[@]}"
    --maxent-branch-grad-diagnostics-interval "$MAXENT_BRANCH_GRAD_DIAGNOSTICS_INTERVAL"
    --maxent-branch-grad-diagnostics-max-steps "$MAXENT_BRANCH_GRAD_DIAGNOSTICS_MAX_STEPS"
    "${maxent_beta_controller_flag[@]}"
  )
  if [[ -n "$MAXENT_TARGET_WEIGHT_ENTROPY" ]]; then
    objective_args+=(
      --maxent-target-weight-entropy "$MAXENT_TARGET_WEIGHT_ENTROPY"
    )
  fi
  if [[ -n "$MAXENT_TARGET_WEIGHT_ENTROPY_START" ]]; then
    objective_args+=(
      --maxent-target-weight-entropy-start "$MAXENT_TARGET_WEIGHT_ENTROPY_START"
    )
  fi
  if [[ -n "$MAXENT_TARGET_WEIGHT_ENTROPY_PEAK" ]]; then
    objective_args+=(
      --maxent-target-weight-entropy-peak "$MAXENT_TARGET_WEIGHT_ENTROPY_PEAK"
      --maxent-target-weight-entropy-peak-step "$MAXENT_TARGET_WEIGHT_ENTROPY_PEAK_STEP"
    )
  fi
  if [[ -n "$MAXENT_TARGET_WEIGHT_ENTROPY_FINAL" ]]; then
    objective_args+=(
      --maxent-target-weight-entropy-final "$MAXENT_TARGET_WEIGHT_ENTROPY_FINAL"
    )
  fi
  if [[ "$MAXENT_LOGPROB_CHUNK_SIZE" -gt 0 ]]; then
    objective_args+=(
      --maxent-logprob-chunk-size "$MAXENT_LOGPROB_CHUNK_SIZE"
    )
  fi
  if [[ "$MAXENT_BACKWARD_CHUNK_SIZE" -gt 0 ]]; then
    objective_args+=(
      --maxent-backward-chunk-size "$MAXENT_BACKWARD_CHUNK_SIZE"
    )
  fi
  if [[ -n "${OAT_ZERO_MAXENT_BACKWARD_TOKEN_BUDGET+x}" ]] || [[ "$MAXENT_BACKWARD_TOKEN_BUDGET" -gt 0 ]]; then
    objective_args+=(
      --maxent-backward-token-budget "$MAXENT_BACKWARD_TOKEN_BUDGET"
    )
  fi
fi

cmd=(
  "$PYTHON_BIN" train_zero_math.py
  --critic_type drgrpo
  --gpus "$N_GPU"
  --enable_prefix_caching
  --vllm_gpu_ratio "$VLLM_GPU_RATIO"
  "${flash_attn_flag[@]}"
  --shm_size_mb "$SHM_SIZE_MB"
  --gradient-checkpointing
  --bf16
  --rnd-seed
  --learning_rate "$LEARNING_RATE"
  --lr_scheduler constant
  --num_ppo_epochs "$NUM_PPO_EPOCHS"
  --max_norm "$MAX_NORM"
  "${ignore_no_eos_flag[@]}"
  "${objective_args[@]}"
  --oracle_type reward
  --oracle math
  --pretrain "$PRETRAIN"
  --prompt_template "$PROMPT_TEMPLATE"
  --verifier_version "$VERIFIER_VERSION"
  --zero-stage "$ZERO_STAGE"
  --ref_offload
  --prompt_data "$PROMPT_DATA"
  --train_split train
  --input_key "$INPUT_KEY"
  --output_key "$OUTPUT_KEY"
  --max-train 9999999
  --num_prompt_epoch "$NUM_PROMPT_EPOCH"
  --prompt_max_length 1024
  --num_samples "$NUM_SAMPLES"
  --temperature 1
  --top_p 1
  --generate_max_length 3000
  --save_path "$SAVE_PATH"
  --save_steps "$SAVE_STEPS"
  --save_from "$SAVE_FROM"
  --max_save_num "$MAX_SAVE_NUM"
  --max_save_mem "$MAX_SAVE_MEM"
  --train_batch_size "$TRAIN_BATCH_SIZE"
  --train_batch_size_per_device "$TRAIN_BATCH_SIZE_PER_DEVICE"
  --rollout_batch_size "$ROLLOUT_BATCH_SIZE"
  --rollout_batch_size_per_device "$ROLLOUT_BATCH_SIZE_PER_DEVICE"
  --pi_buffer_maxlen_per_device "$PI_BUFFER_MAXLEN_PER_DEVICE"
  --eval_batch_size 200
  --eval_steps "$EVAL_STEPS"
  --eval_temperature 0
  --eval_generate_max_length 3000
  --eval_data "$EVAL_DATA"
  --eval_input_key input
)

if [[ "$VLLM_SLEEP" == "1" ]]; then
  cmd+=(--vllm_sleep)
fi

if [[ "$COLLOCATE" == "1" ]]; then
  cmd+=(--collocate)
fi

if [[ "$ADAM_OFFLOAD" == "1" ]]; then
  cmd+=(--adam_offload)
fi

if [[ "$ACTIVATION_OFFLOADING" == "1" ]]; then
  cmd+=(--activation_offloading)
fi

if [[ "$DISABLE_TRACE_CACHE" == "1" ]]; then
  cmd+=(--disable_trace_cache)
fi

if [[ "$SAVE_CKPT" == "1" ]]; then
  cmd+=(--save-ckpt)
fi

if [[ -n "$GRAD_ACCUM_DTYPE" ]]; then
  cmd+=(--grad_accum_dtype "$GRAD_ACCUM_DTYPE")
fi

if [[ "$USE_WB" == "1" ]]; then
  cmd+=(
    --use-wb
    --wb_project "$WB_PROJECT"
    --wb-run-name "$WB_RUN_NAME"
  )
fi

echo "[oat-zero-exact] ${cmd[*]}"

cd "$UPSTREAM_ROOT"

if [[ "$SMOKE_MODE" != "1" ]]; then
  exec "${cmd[@]}"
fi

RUN_LOG="${ROOT_DIR}/var/artifacts/logs/${WB_RUN_NAME}.inner.log"
rm -f "$RUN_LOG"
touch "$RUN_LOG"

setsid bash -lc 'exec "$@"' _ "${cmd[@]}" \
  > >(tee -a "$RUN_LOG") \
  2> >(tee -a "$RUN_LOG" >&2) &
child_pid=$!
child_pgid=$child_pid

start_ts="$(date +%s)"
smoke_success=0

while kill -0 "$child_pid" >/dev/null 2>&1; do
  if rg -q --fixed-strings "$SMOKE_SUCCESS_PATTERN" "$RUN_LOG"; then
    smoke_success=1
    break
  fi
  now_ts="$(date +%s)"
  if (( now_ts - start_ts >= SMOKE_TIMEOUT_SECONDS )); then
    break
  fi
  sleep 10
done

if (( smoke_success == 1 )); then
  echo "[oat-zero-exact] smoke success pattern observed; stopping exact upstream run cleanly."
  kill -INT -- "-${child_pgid}" >/dev/null 2>&1 || true
  wait "$child_pid" || true
  exit 0
fi

if kill -0 "$child_pid" >/dev/null 2>&1; then
  echo "[oat-zero-exact] smoke run timed out before success pattern." >&2
  kill -TERM -- "-${child_pgid}" >/dev/null 2>&1 || true
  wait "$child_pid" || true
  exit 1
fi

wait "$child_pid"
