#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/ops/repo_env.sh"

MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
RESULTS_DIR="${RESULTS_DIR:?RESULTS_DIR is required}"
TASKS="${TASKS:-aime,amc,math,minerva,olympiad_bench}"
TEMPLATE="${TEMPLATE:-qwen_math}"
VLLM_PORT="${VLLM_PORT:-8123}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-true}"
PASS_AT_8_SAMPLES="${PASS_AT_8_SAMPLES:-8}"
PASS_AT_8_TEMPERATURE="${PASS_AT_8_TEMPERATURE:-1.0}"
PASS_AT_8_TOP_P="${PASS_AT_8_TOP_P:-1.0}"
VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-32}"
VLLM_LOG="${VLLM_LOG:-$RESULTS_DIR/vllm_server.log}"
EVAL_STDOUT_LOG="${EVAL_STDOUT_LOG:-$RESULTS_DIR/eval_stdout.log}"

mkdir -p "$RESULTS_DIR"

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

python "$ROOT_DIR/tools/vllm_serve_compat.py" \
  --model "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port "$VLLM_PORT" \
  --tensor_parallel_size 1 \
  --data_parallel_size 1 \
  --max_model_len "$VLLM_MAX_MODEL_LEN" \
  --dtype "$VLLM_DTYPE" \
  --enforce_eager "$VLLM_ENFORCE_EAGER" \
  >"$VLLM_LOG" 2>&1 &
VLLM_PID="$!"

for attempt in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${VLLM_PORT}/health/" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM server exited during startup" >&2
    tail -n 120 "$VLLM_LOG" >&2 || true
    exit 1
  fi
  sleep 5
done

if ! curl -fsS "http://127.0.0.1:${VLLM_PORT}/health/" >/dev/null 2>&1; then
  echo "vLLM server failed health check" >&2
  tail -n 120 "$VLLM_LOG" >&2 || true
  exit 1
fi

python "$ROOT_DIR/tools/seed_paper_eval.py" \
  --model-name "$MODEL_PATH" \
  --template "$TEMPLATE" \
  --tasks "$TASKS" \
  --results-dir "$RESULTS_DIR" \
  --vllm-url "http://127.0.0.1:${VLLM_PORT}/generate" \
  --vllm-batch-size "$VLLM_BATCH_SIZE" \
  --pass-at-8 \
  --pass-at-8-samples "$PASS_AT_8_SAMPLES" \
  --pass-at-8-temperature "$PASS_AT_8_TEMPERATURE" \
  --pass-at-8-top-p "$PASS_AT_8_TOP_P" \
  --save-outputs \
  2>&1 | tee "$EVAL_STDOUT_LOG"
