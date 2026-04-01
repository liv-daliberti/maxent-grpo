#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PROMPT_SOURCE="${PROMPT_SOURCE:?PROMPT_SOURCE is required}"
RESULTS_DIR="${RESULTS_DIR:?RESULTS_DIR is required}"
STEPS="${STEPS:-200,300}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-true}"

cd "$ROOT_DIR"
source "$ROOT_DIR/ops/repo_env.sh"

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export MAXENT_DISABLE_BUILTIN_WEIGHT_TRANSFER=1
EVAL_PY="${EVAL_PY:-$ROOT_DIR/var/seed_paper_eval/paper_venv/bin/python}"

mkdir -p "$RESULTS_DIR"
export ROOT_DIR PROMPT_SOURCE RESULTS_DIR EVAL_PY

python - <<'PY'
import json
import os

src = os.environ["PROMPT_SOURCE"]
out = os.path.join(os.environ["RESULTS_DIR"], "prompt.json")
with open(src, encoding="utf-8") as f:
    data = json.load(f)
item = data[0]
with open(out, "w", encoding="utf-8") as f:
    json.dump(
        {
            "prompt": item["prompt"],
            "gt": item["gt"],
            "prompt_index": item["prompt_index"],
        },
        f,
        indent=2,
    )
print(f"prompt_json {out}")
PY

pick_free_port() {
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

run_probe() {
  local step="$1"
  local model_path="$ROOT_DIR/var/data/drgrpo_1p5b_oat_parity_trl_r1_20260331_144832/checkpoint-${step}"
  local step_dir="$RESULTS_DIR/step${step}"
  mkdir -p "$step_dir"

  local port
  port="$(pick_free_port $((18000 + step)))"
  local internal_port
  internal_port="$(pick_free_port $((19000 + step)))"
  if [[ "$internal_port" == "$port" ]]; then
    internal_port="$(pick_free_port 0)"
  fi

  export VLLM_DP_MASTER_PORT="$internal_port"
  python "$ROOT_DIR/tools/vllm_serve_compat.py" \
    --model "$model_path" \
    --host 127.0.0.1 \
    --port "$port" \
    --tensor_parallel_size 1 \
    --data_parallel_size 1 \
    --max_model_len "$VLLM_MAX_MODEL_LEN" \
    --dtype "$VLLM_DTYPE" \
    --enforce_eager "$VLLM_ENFORCE_EAGER" \
    >"$step_dir/vllm_server.log" 2>&1 &
  local vllm_pid=$!

  cleanup_step() {
    kill "$vllm_pid" 2>/dev/null || true
    wait "$vllm_pid" 2>/dev/null || true
  }
  trap cleanup_step RETURN

  for _attempt in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${port}/health/" >/dev/null 2>&1; then
      break
    fi
    if ! kill -0 "$vllm_pid" 2>/dev/null; then
      echo "vLLM exited during startup for step ${step}" >&2
      tail -n 120 "$step_dir/vllm_server.log" >&2 || true
      return 1
    fi
    sleep 2
  done

  curl -fsS "http://127.0.0.1:${port}/health/" >/dev/null

  STEP="$step" \
  PORT="$port" \
  MODEL_PATH="$model_path" \
  PROMPT_JSON="$RESULTS_DIR/prompt.json" \
  STEP_DIR="$step_dir" \
  "$EVAL_PY" - <<'PY'
import json
import os
import urllib.request

from transformers import AutoTokenizer

step = int(os.environ["STEP"])
port = int(os.environ["PORT"])
model_path = os.environ["MODEL_PATH"]
prompt_json = os.environ["PROMPT_JSON"]
step_dir = os.environ["STEP_DIR"]

with open(prompt_json, encoding="utf-8") as f:
    prompt_payload = json.load(f)

payload = {
    "prompts": [prompt_payload["prompt"]],
    "temperature": 0.0,
    "top_p": 1.0,
    "n": 1,
    "max_tokens": 3000,
    "stream": False,
    "return_logprobs": True,
    "logprobs": 1,
}

request = urllib.request.Request(
    f"http://127.0.0.1:{port}/generate",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=900) as response:
    raw = response.read()

raw_json = json.loads(raw.decode("utf-8"))
raw_path = os.path.join(step_dir, "raw_generate_response.json")
with open(raw_path, "w", encoding="utf-8") as f:
    json.dump(raw_json, f, indent=2)

completion_ids = raw_json["completion_ids"][0]
tokenizer = AutoTokenizer.from_pretrained(model_path)
decoded = tokenizer.decode(completion_ids, skip_special_tokens=True)

decoded_path = os.path.join(step_dir, "decoded_text.txt")
with open(decoded_path, "w", encoding="utf-8") as f:
    f.write(decoded)

summary = {
    "step": step,
    "prompt_index": prompt_payload["prompt_index"],
    "gt": prompt_payload["gt"],
    "completion_token_ids_len": len(completion_ids),
    "decoded_text_len": len(decoded),
    "decoded_text_head": decoded[:2000],
    "decoded_text_full_path": decoded_path,
    "raw_response_path": raw_path,
}
summary_path = os.path.join(step_dir, "probe_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
PY

  trap - RETURN
}

IFS=',' read -r -a step_values <<< "$STEPS"
for raw_step in "${step_values[@]}"; do
  step="${raw_step//[[:space:]]/}"
  [[ -n "$step" ]] || continue
  run_probe "$step"
done
