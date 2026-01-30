#!/usr/bin/env bash
# MATH-500 pass@1 (det) for:
#   1) baseline: Qwen/Qwen2.5-1.5B-Instruct + your training system prompt
#   2) finetuned: od2961/Qwen2.5-1.5B-Open-R1-GRPO-math-2k + same system prompt
#
# Notes:
# - LightEval model configs support system_prompt + cache_dir. :contentReference[oaicite:2]{index=2}
# - Chat templating/system prompts are meant for instruct models; base models may do poorly. :contentReference[oaicite:3]{index=3}

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Keep caches local (no ~/.local, no ~/.cache).
export PYTHONNOUSERSITE=1
export PIP_NO_USER_CONFIG=1
export XDG_CACHE_HOME="${REPO_ROOT}/.cache"
export PIP_CACHE_DIR="${REPO_ROOT}/.cache/pip"
export HF_HOME="${REPO_ROOT}/.cache/huggingface"
# Avoid noisy (and occasionally failing) background usage/telemetry threads in vLLM.
export VLLM_NO_USAGE_STATS=1
export VLLM_DO_NOT_TRACK=1
export DO_NOT_TRACK=1
# Avoid FlashInfer warnings/fallbacks unless explicitly enabled.
export VLLM_USE_FLASHINFER_SAMPLER=0

LIGHEVAL_BIN="${LIGHEVAL_BIN:-}"
if [[ -z "$LIGHEVAL_BIN" ]]; then
  # Prefer the legacy venv-lighteval path but fall back to the repo-local var/lighteval env.
  for candidate in \
    "${REPO_ROOT}/venv-lighteval/bin/lighteval" \
    "${REPO_ROOT}/var/lighteval/bin/lighteval"
  do
    if [[ -x "$candidate" ]]; then
      LIGHEVAL_BIN="$candidate"
      break
    fi
  done
fi
CUSTOM_TASKS="${REPO_ROOT}/custom_tasks/math500_passk.py"
CUSTOM_TASKS_MODULE="custom_tasks.math500_passk"
PASS1_CFG_TEMPLATE="${REPO_ROOT}/custom_tasks/pass1.yaml"
SYSTEM_PROMPT_PATH="${REPO_ROOT}/configs/prompts/math_system_prompt.txt"

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/var/artifacts/results}"
LE_CACHE_DIR="${REPO_ROOT}/.cache/huggingface/lighteval"

mkdir -p "${OUT_DIR}" "${LE_CACHE_DIR}" "${PIP_CACHE_DIR}" "${HF_HOME}"

if [[ ! -x "${LIGHEVAL_BIN}" ]]; then
  echo "error: ${LIGHEVAL_BIN:-<unset>} not found or not executable. Set LIGHEVAL_BIN or create venv-lighteval/var/lighteval." >&2
  exit 1
fi
if [[ ! -f "${CUSTOM_TASKS}" ]]; then
  echo "error: custom task file not found: ${CUSTOM_TASKS}" >&2
  exit 1
fi
if [[ ! -f "${PASS1_CFG_TEMPLATE}" ]]; then
  echo "error: pass1 config template not found: ${PASS1_CFG_TEMPLATE}" >&2
  exit 1
fi
if [[ ! -f "${SYSTEM_PROMPT_PATH}" ]]; then
  echo "error: system prompt file not found: ${SYSTEM_PROMPT_PATH}" >&2
  echo "       Put your training system prompt into ${SYSTEM_PROMPT_PATH}" >&2
  exit 1
fi

# baseline + finetuned
MODEL_VARIANTS=(
  "baseline:Qwen/Qwen2.5-1.5B-Instruct:main"
  "finetuned:od2961/Qwen2.5-1.5B-Open-R1-GRPO-math-2k:main"
)

tmp_files=()
cleanup() {
  local status=$?
  for f in "${tmp_files[@]:-}"; do
    [[ -f "${f}" ]] && rm -f "${f}"
  done
  return "${status}"
}
trap cleanup EXIT

rewrite_config() {
  local src="$1"
  local dst="$2"
  local model_name="$3"
  local revision="$4"

  python - <<'PY' "${src}" "${dst}" "${model_name}" "${revision}"
import pathlib
import sys
import yaml

src_path = pathlib.Path(sys.argv[1])
dst_path = pathlib.Path(sys.argv[2])
model_name = sys.argv[3]
revision = sys.argv[4]

data = yaml.safe_load(src_path.read_text()) or {}
mp = data.setdefault("model_parameters", {})

mp["model_name"] = model_name
if revision:
    mp["revision"] = revision

# Note: this repo pins a LightEval build whose VLLMModelConfig forbids extra keys
# such as `system_prompt` / `cache_dir`. Pass prompts via `--system-prompt` and
# keep HF caches local via env vars (HF_HOME/XDG_CACHE_HOME) instead.

dst_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
PY
}

run_pass1() {
  local slug="$1"
  local model_name="$2"
  local model_rev="$3"

  local tmp_cfg
  tmp_cfg="$(mktemp "${TMPDIR:-/tmp}/lighteval_${slug}_pass1.XXXX.yaml")"
  tmp_files+=("${tmp_cfg}")

  rewrite_config "${PASS1_CFG_TEMPLATE}" "${tmp_cfg}" "${model_name}" "${model_rev}"

  echo ">>> Running math_500_pass1_det (pass1) for ${slug} model ${model_name} (rev=${model_rev})"

  unset PYTHONPATH
  conda deactivate 2>/dev/null || true

  "${LIGHEVAL_BIN}" vllm \
    "${tmp_cfg}" \
    "custom|math_500_pass1_det|0|0" \
    --custom-tasks "${CUSTOM_TASKS_MODULE}" \
    --output-dir "${OUT_DIR}" \
    --save-details \
    --system-prompt "$(cat "${SYSTEM_PROMPT_PATH}")"
}

for v in "${MODEL_VARIANTS[@]}"; do
  IFS=":" read -r slug model_name model_rev <<<"${v}"
  run_pass1 "${slug}" "${model_name}" "${model_rev}"
done
