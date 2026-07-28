#!/usr/bin/env bash
# Dedicated E47 Qwen2.5-72B-AWQ judge server on four node105 A5000s.
# Four-way sharding leaves room for the checkpoint's 32,768-token KV cache.
#SBATCH --job-name=e47-qwen72
#SBATCH --output=/n/fs/similarity/maxent-grpo/var/artifacts/e47_math_strategy_calibration_v1/qwen72-server-%j.out
#SBATCH --error=/n/fs/similarity/maxent-grpo/var/artifacts/e47_math_strategy_calibration_v1/qwen72-server-%j.out
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --nodelist=node105
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a5000:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=12:00:00

set -euo pipefail

ROOT_DIR="${OAT_ZERO_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ "$ROOT_DIR" != "/n/fs/similarity/maxent-grpo" ]]; then
  echo "E47 Qwen72 server requires the frozen repository root" >&2
  exit 2
fi
if [[ "${SLURM_JOB_PARTITION:-}" != "mltheory" ]] \
  || [[ "${SLURM_JOB_ACCOUNT:-}" != "mltheory" ]] \
  || [[ "$(hostname -s)" != "node105" ]]; then
  echo "E47 Qwen72 server requires mltheory/mltheory on node105" >&2
  exit 2
fi

MODEL="/n/fs/similarity/tokenomics/.hf_cache/hub/models--Qwen--Qwen2.5-72B-Instruct-AWQ/snapshots/698703eae6604af048a3d2f509995dc302088217"
VLLM_PY="/n/fs/similarity/Illusion-of-Reasoning/openr1/bin/python"
OUTPUT="${OAT_ZERO_QWEN72_OUTPUT:-$ROOT_DIR/var/artifacts/e47_math_strategy_calibration_v1}"
ENDPOINT="$OUTPUT/qwen72_endpoint.json"
ENDPOINT_TMP="$ENDPOINT.${SLURM_JOB_ID}.tmp"
VLLM_LOG="$OUTPUT/qwen72-vllm-${SLURM_JOB_ID}.log"
PORT="${OAT_ZERO_QWEN72_PORT:-8766}"
TP=4
MAX_LEN=32768
MAX_NUM_SEQS=8

for frozen in \
  "config.json:ec4813b56d971f7a9c4490c7c148f997c468e97e9e8ae39e80f380d51a6dda1f" \
  "tokenizer.json:22bfef58b74d2fa40f402ac0c8d638d9e54d12137da2e2c7e9f32a1e7534e42f" \
  "model.safetensors.index.json:9f9e509271b4a2da37444f8a7ac3b3758279f6044c9c2f6eb40edc171329a65a"; do
  file="${frozen%%:*}"
  expected="${frozen#*:}"
  observed="$(sha256sum "$MODEL/$file" | cut -d' ' -f1)"
  if [[ "$observed" != "$expected" ]]; then
    echo "Frozen Qwen72 identity mismatch for $file" >&2
    exit 2
  fi
done

export LD_LIBRARY_PATH="/n/fs/similarity/Illusion-of-Reasoning/openr1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME="/n/fs/similarity/tokenomics/.hf_cache"
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_CACHE_ROOT="/tmp/${USER}/e47-qwen72-${SLURM_JOB_ID}"
mkdir -p "$OUTPUT" "$VLLM_CACHE_ROOT"

cleanup() {
  kill "${VLLM_PID:-}" 2>/dev/null || true
  if [[ -f "$ENDPOINT" ]] \
    && grep -q "\"job_id\": \"${SLURM_JOB_ID}\"" "$ENDPOINT"; then
    rm -f "$ENDPOINT"
  fi
  rm -f "$ENDPOINT_TMP"
}
trap cleanup EXIT

echo "=== E47 Qwen72 | node105 | A5000x4 | TP=4 | max_len=$MAX_LEN | max_num_seqs=$MAX_NUM_SEQS ==="
"$VLLM_PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name qwen2.5-72b \
  --tensor-parallel-size "$TP" \
  --enforce-eager \
  --disable-custom-all-reduce \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --guided-decoding-backend guidance \
  --max-model-len "$MAX_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization 0.88 \
  --disable-log-requests \
  --host 0.0.0.0 \
  --port "$PORT" >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

ready=0
for _ in $(seq 1 180); do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM exited before becoming ready" >&2
    tail -n 80 "$VLLM_LOG"
    exit 1
  fi
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 10
done
if [[ "$ready" != 1 ]]; then
  echo "vLLM did not become ready within 30 minutes" >&2
  exit 1
fi

printf '{"job_id": "%s", "model": "qwen2.5-72b", "base_url": "slurm://%s:%s/v1", "node": "node105", "port": %s, "tensor_parallel_size": %s, "max_model_len": %s, "max_num_seqs": %s, "enforce_eager": true, "structured_output_backend": "guidance", "gpu_memory_utilization": 0.88, "checkpoint_revision": "698703eae6604af048a3d2f509995dc302088217"}\n' \
  "$SLURM_JOB_ID" "$SLURM_JOB_ID" "$PORT" "$PORT" "$TP" "$MAX_LEN" "$MAX_NUM_SEQS" \
  >"$ENDPOINT_TMP"
mv "$ENDPOINT_TMP" "$ENDPOINT"
echo "=== READY: slurm://${SLURM_JOB_ID}:${PORT}/v1 ==="
wait "$VLLM_PID"
