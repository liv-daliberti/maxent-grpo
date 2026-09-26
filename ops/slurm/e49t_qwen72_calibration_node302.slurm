#!/usr/bin/env bash
# E49T Qwen2.5-72B-AWQ auditor plus the frozen route-confusion calibration.
# The server remains alive after a passing calibration so matched toy training
# can use the exact endpoint and checkpoint that were calibrated.
#SBATCH --job-name=e49t-qwen72
#SBATCH --output=/n/fs/similarity/maxent-grpo/var/artifacts/e49t_qwen72_node302_v1/server-%j.out
#SBATCH --error=/n/fs/similarity/maxent-grpo/var/artifacts/e49t_qwen72_node302_v1/server-%j.out
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --nodelist=node302
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=24:00:00

set -euo pipefail

ROOT_DIR="${OAT_ZERO_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
EXPECTED_ROOT="/n/fs/similarity/maxent-grpo"
if [[ "$ROOT_DIR" != "$EXPECTED_ROOT" ]]; then
  echo "E49T Qwen72 server requires $EXPECTED_ROOT" >&2
  exit 2
fi
if [[ "${SLURM_JOB_PARTITION:-}" != "mltheory" ]] \
  || [[ "${SLURM_JOB_ACCOUNT:-}" != "mltheory" ]] \
  || [[ "$(hostname -s)" != "node302" ]]; then
  echo "E49T Qwen72 server requires mltheory/mltheory on node302" >&2
  exit 2
fi

MODEL="/n/fs/similarity/tokenomics/.hf_cache/hub/models--Qwen--Qwen2.5-72B-Instruct-AWQ/snapshots/698703eae6604af048a3d2f509995dc302088217"
VLLM_PY="/n/fs/similarity/Illusion-of-Reasoning/openr1/bin/python"
SCORE_PY="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
OUTPUT="$ROOT_DIR/var/artifacts/e49t_qwen72_node302_v1"
CALIBRATION="$ROOT_DIR/var/artifacts/e49t_route_confusion_calibration_v1"
ENDPOINT="$OUTPUT/qwen72_endpoint.json"
ENDPOINT_TMP="$ENDPOINT.${SLURM_JOB_ID}.tmp"
RESULT="$CALIBRATION/result.json"
VLLM_LOG="$OUTPUT/qwen72-vllm-${SLURM_JOB_ID}.log"
SCORER="$ROOT_DIR/ops/math_strategy_calibration/score_e49t_route_confusion_calibration.py"
PORT="${OAT_ZERO_QWEN72_PORT:-8770}"
TP=4
MAX_LEN=32768
MAX_NUM_SEQS=8

for required in \
  "$SCORE_PY" \
  "$SCORER" \
  "$CALIBRATION/cohort.jsonl" \
  "$CALIBRATION/private/labels.jsonl" \
  "$CALIBRATION/frozen_identity.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing E49T frozen prerequisite: $required" >&2
    exit 2
  fi
done
if [[ -e "$RESULT" ]]; then
  echo "Fresh E49T route-confusion result required: $RESULT" >&2
  exit 2
fi
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
export VLLM_CACHE_ROOT="/tmp/${USER}/e49t-qwen72-${SLURM_JOB_ID}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
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

echo "=== E49T Qwen72 | node302 | A100x4 | TP=4 | max_len=$MAX_LEN ==="
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
    tail -n 100 "$VLLM_LOG"
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

printf '{"job_id": "%s", "model": "qwen2.5-72b", "base_url": "slurm://%s:%s/v1", "node": "node302", "port": %s, "tensor_parallel_size": %s, "max_model_len": %s, "max_num_seqs": %s, "enforce_eager": true, "structured_output_backend": "guidance", "gpu_memory_utilization": 0.88, "checkpoint_revision": "698703eae6604af048a3d2f509995dc302088217"}\n' \
  "$SLURM_JOB_ID" "$SLURM_JOB_ID" "$PORT" "$PORT" "$TP" "$MAX_LEN" "$MAX_NUM_SEQS" \
  >"$ENDPOINT_TMP"
mv "$ENDPOINT_TMP" "$ENDPOINT"
echo "=== READY: slurm://${SLURM_JOB_ID}:${PORT}/v1 ==="

"$SCORE_PY" "$SCORER" \
  --cohort-dir "$CALIBRATION" \
  --endpoint-record "$ENDPOINT" \
  --output "$RESULT"

echo "=== E49T ROUTE-CONFUSION CALIBRATION PASSED; SERVER RETAINED ==="
wait "$VLLM_PID"
