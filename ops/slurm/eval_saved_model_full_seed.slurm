#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-}"
if [[ -z "$ROOT_DIR" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/ops/repo_env.sh" ]]; then
    ROOT_DIR="$SLURM_SUBMIT_DIR"
  else
    ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
  fi
fi
cd "$ROOT_DIR"
source "$ROOT_DIR/ops/repo_env.sh"

if ! command -v ninja >/dev/null 2>&1; then
  FALLBACK_NINJA_BIN="/n/fs/similarity/kalshi/markets_pipeline/.runtime/qwen_vllm/venv/bin"
  if [[ -x "$FALLBACK_NINJA_BIN/ninja" ]]; then
    export PATH="$PATH:$FALLBACK_NINJA_BIN"
  fi
fi

MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
RESULTS_DIR="${RESULTS_DIR:?RESULTS_DIR is required}"
TASKS="${TASKS:-aime,amc,math,minerva,olympiad_bench}"
TEMPLATE="${TEMPLATE:-no}"
PORT_SEED="${SLURM_JOB_ID:-$$}"
VLLM_PORT="${VLLM_PORT:-$((18000 + (PORT_SEED % 20000)))}"
VLLM_INTERNAL_PORT="${VLLM_INTERNAL_PORT:-$((VLLM_PORT + 1000))}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-true}"
PASS_AT_8_SAMPLES="${PASS_AT_8_SAMPLES:-8}"
PASS_AT_8_TEMPERATURE="${PASS_AT_8_TEMPERATURE:-1.0}"
PASS_AT_8_TOP_P="${PASS_AT_8_TOP_P:-1.0}"
VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-32}"
VLLM_LOG="${VLLM_LOG:-$RESULTS_DIR/vllm_server.log}"
EVAL_STDOUT_LOG="${EVAL_STDOUT_LOG:-$RESULTS_DIR/eval_stdout.log}"
EVAL_PYTHON="${EVAL_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper_venv/bin/python}"
VLLM_HEALTH_WAIT_LOOPS="${VLLM_HEALTH_WAIT_LOOPS:-240}"
VLLM_HEALTH_WAIT_SLEEP_S="${VLLM_HEALTH_WAIT_SLEEP_S:-5}"
PROMPT_START="${PROMPT_START:-}"
PROMPT_END="${PROMPT_END:-}"
SAMPLING_SEED="${SAMPLING_SEED:-}"

mkdir -p "$RESULTS_DIR"

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

pick_free_port() {
  # Use -S so repo-local sitecustomize.py cannot pollute stdout with
  # compatibility-import warnings; this function must print only the port.
  python -S - "$1" <<'PY'
import socket
import sys

preferred = int(sys.argv[1]) if len(sys.argv) > 1 else 0

def bindable(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
        return True

if preferred > 0 and bindable(preferred):
    print(preferred)
    raise SystemExit

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
}

VLLM_PORT="$(pick_free_port "$VLLM_PORT")"
VLLM_INTERNAL_PORT="$(pick_free_port "$VLLM_INTERNAL_PORT")"
if [[ "$VLLM_INTERNAL_PORT" == "$VLLM_PORT" ]]; then
  VLLM_INTERNAL_PORT="$(pick_free_port 0)"
fi

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_DP_MASTER_PORT="$VLLM_INTERNAL_PORT"
export MAXENT_DISABLE_BUILTIN_WEIGHT_TRANSFER=1

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

for attempt in $(seq 1 "$VLLM_HEALTH_WAIT_LOOPS"); do
  if curl -fsS "http://127.0.0.1:${VLLM_PORT}/health/" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM server exited during startup" >&2
    tail -n 120 "$VLLM_LOG" >&2 || true
    exit 1
  fi
  sleep "$VLLM_HEALTH_WAIT_SLEEP_S"
done

if ! curl -fsS "http://127.0.0.1:${VLLM_PORT}/health/" >/dev/null 2>&1; then
  echo "vLLM server failed health check after $((VLLM_HEALTH_WAIT_LOOPS * VLLM_HEALTH_WAIT_SLEEP_S))s" >&2
  tail -n 120 "$VLLM_LOG" >&2 || true
  exit 1
fi

eval_cmd=(
  env
  PYTHONNOUSERSITE=1
  PYTHONPATH="$ROOT_DIR/src"
  "$EVAL_PYTHON" "$ROOT_DIR/tools/seed_paper_eval.py"
  --model-name "$MODEL_PATH"
  --template "$TEMPLATE"
  --tasks "$TASKS"
  --results-dir "$RESULTS_DIR"
  --vllm-url "http://127.0.0.1:${VLLM_PORT}/generate"
  --vllm-batch-size "$VLLM_BATCH_SIZE"
  --pass-at-8
  --pass-at-8-samples "$PASS_AT_8_SAMPLES"
  --pass-at-8-temperature "$PASS_AT_8_TEMPERATURE"
  --pass-at-8-top-p "$PASS_AT_8_TOP_P"
  --save-outputs
)

if [[ -n "$PROMPT_START" ]]; then
  eval_cmd+=(--prompt-start "$PROMPT_START")
fi
if [[ -n "$PROMPT_END" ]]; then
  eval_cmd+=(--prompt-end "$PROMPT_END")
fi
if [[ -n "$SAMPLING_SEED" ]]; then
  eval_cmd+=(--sampling-seed "$SAMPLING_SEED")
fi

"${eval_cmd[@]}" 2>&1 | tee "$EVAL_STDOUT_LOG"
