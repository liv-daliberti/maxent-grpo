#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-}"
if [[ -z "$ROOT_DIR" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/ops/repo_env.sh" ]]; then
    ROOT_DIR="$SLURM_SUBMIT_DIR"
  else
    ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
DATASET_DIR="${DATASET_DIR:-}"
TASKS="${TASKS:-aime,amc,math,minerva,olympiad_bench}"
TEMPLATE="${TEMPLATE:-no}"
PORT_SEED="${SLURM_JOB_ID:-$$}"
VLLM_PORT="${VLLM_PORT:-$((18000 + (PORT_SEED % 20000)))}"
VLLM_INTERNAL_PORT="${VLLM_INTERNAL_PORT:-$((VLLM_PORT + 1000))}"
VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-32}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
PASS_AT_8_SAMPLES="${PASS_AT_8_SAMPLES:-8}"
PASS_AT_8_TEMPERATURE="${PASS_AT_8_TEMPERATURE:-1.0}"
PASS_AT_8_TOP_P="${PASS_AT_8_TOP_P:-1.0}"
VLLM_LOG="${VLLM_LOG:-$RESULTS_DIR/vllm_server.log}"
EVAL_STDOUT_LOG="${EVAL_STDOUT_LOG:-$RESULTS_DIR/eval_stdout.log}"
EVAL_PYTHON="${EVAL_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper_venv/bin/python}"
RUNTIME_SCRATCH_ROOT="${RUNTIME_SCRATCH_ROOT:-$ROOT_DIR/var/tmp/manual_seed_eval}"
RUNTIME_TMPDIR="${RUNTIME_TMPDIR:-$RUNTIME_SCRATCH_ROOT/tmp_${SLURM_JOB_ID:-$$}}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$RUNTIME_SCRATCH_ROOT/triton_${SLURM_JOB_ID:-$$}}"

mkdir -p "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR" "$RUNTIME_TMPDIR" "$TRITON_CACHE_DIR"

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

echo "START $(date -Is) host=$(hostname)"
echo "MODEL_PATH=$MODEL_PATH"
echo "RESULTS_DIR=$RESULTS_DIR"
echo "DATASET_DIR=$DATASET_DIR"
echo "TEMPLATE=$TEMPLATE"
echo "TASKS=$TASKS"
echo "VLLM_PORT=$VLLM_PORT"
echo "VLLM_INTERNAL_PORT=$VLLM_INTERNAL_PORT"
echo "VLLM_DTYPE=$VLLM_DTYPE"

export VLLM_DP_MASTER_PORT="$VLLM_INTERNAL_PORT"
export MAXENT_DISABLE_BUILTIN_WEIGHT_TRANSFER=1
export TMPDIR="$RUNTIME_TMPDIR"
export TMP="$RUNTIME_TMPDIR"
export TEMP="$RUNTIME_TMPDIR"
export TRITON_CACHE_DIR
python tools/vllm_serve_compat.py \
  --model "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port "$VLLM_PORT" \
  --tensor_parallel_size 1 \
  --data_parallel_size 1 \
  --max_model_len 4096 \
  --dtype "$VLLM_DTYPE" \
  --enforce_eager true \
  >"$VLLM_LOG" 2>&1 &
VLLM_PID="$!"
echo "VLLM_PID=$VLLM_PID"

for attempt in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${VLLM_PORT}/health/" >/dev/null 2>&1; then
    echo "HEALTHY attempt=$attempt"
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "VLLM_EXITED"
    tail -n 200 "$VLLM_LOG" || true
    exit 1
  fi
  if (( attempt % 5 == 0 )); then
    echo "WAIT attempt=$attempt"
    ls -l "$VLLM_LOG" || true
    tail -n 50 "$VLLM_LOG" || true
  fi
  sleep 2
done

eval_cmd=(
  "$EVAL_PYTHON" tools/seed_paper_eval.py
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
if [[ -n "$DATASET_DIR" ]]; then
  eval_cmd+=(--dataset-dir "$DATASET_DIR")
fi
env \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="$ROOT_DIR/src" \
  "${eval_cmd[@]}" 2>&1 | tee "$EVAL_STDOUT_LOG"
