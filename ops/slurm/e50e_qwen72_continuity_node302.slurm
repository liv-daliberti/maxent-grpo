#!/usr/bin/env bash
# Fresh 48-hour Qwen72 service for E50E. It must pass both frozen finite-menu
# controls before it can submit materialization or training.
#SBATCH --job-name=e50e-qwen72
#SBATCH --output=/n/fs/similarity/maxent-grpo/var/artifacts/e50e_qwen72_node302_v1/server-%j.out
#SBATCH --error=/n/fs/similarity/maxent-grpo/var/artifacts/e50e_qwen72_node302_v1/server-%j.out
#SBATCH --partition=mltheory
#SBATCH --account=mltheory
#SBATCH --nodelist=node302
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=2-00:00:00

set -euo pipefail

ROOT_DIR="${OAT_ZERO_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
EXPECTED_ROOT=/n/fs/similarity/maxent-grpo
if [[ "$ROOT_DIR" != "$EXPECTED_ROOT" ]]; then
  echo "E50E continuity server requires $EXPECTED_ROOT" >&2
  exit 2
fi
if [[ "${SLURM_JOB_PARTITION:-}" != mltheory ]] \
  || [[ "${SLURM_JOB_ACCOUNT:-}" != mltheory ]] \
  || [[ "$(hostname -s)" != node302 ]]; then
  echo "E50E continuity server requires mltheory/mltheory on node302" >&2
  exit 2
fi

source "$ROOT_DIR/ops/repo_env.sh"
MODEL="/n/fs/similarity/tokenomics/.hf_cache/hub/models--Qwen--Qwen2.5-72B-Instruct-AWQ/snapshots/698703eae6604af048a3d2f509995dc302088217"
VLLM_PY=/n/fs/similarity/Illusion-of-Reasoning/openr1/bin/python
SCORE_PY="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
OUTPUT="$ROOT_DIR/var/artifacts/e50e_qwen72_node302_v1"
ENDPOINT="$OUTPUT/qwen72_endpoint.json"
ENDPOINT_TMP="$ENDPOINT.${SLURM_JOB_ID}.tmp"
VLLM_LOG="$OUTPUT/qwen72-vllm-${SLURM_JOB_ID}.log"
ROUTE_RESULT="$ROOT_DIR/var/artifacts/e50e_route_confusion_continuity_v1/result.json"
DECLARATION_RESULT="$ROOT_DIR/var/artifacts/e50e_declaration_mismatch_continuity_v1/result.json"
CERTIFICATE="$ROOT_DIR/var/artifacts/e50e_qwen72_continuity_v1/certificate.json"
PORT=8771
TP=4
MAX_LEN=32768
MAX_NUM_SEQS=8

for required in \
  "$SCORE_PY" \
  "$ROOT_DIR/ops/math_strategy_calibration/score_e49t_route_confusion_calibration.py" \
  "$ROOT_DIR/ops/math_strategy_calibration/score_e49t_declaration_mismatch_calibration.py" \
  "$ROOT_DIR/ops/math_strategy_calibration/certify_e50e_qwen72_continuity.py" \
  "$ROOT_DIR/var/artifacts/e49t_route_confusion_calibration_v1/cohort.jsonl" \
  "$ROOT_DIR/var/artifacts/e49t_route_confusion_calibration_v1/private/labels.jsonl" \
  "$ROOT_DIR/var/artifacts/e49t_declaration_mismatch_calibration_v1/cohort.jsonl" \
  "$ROOT_DIR/var/artifacts/e49t_declaration_mismatch_calibration_v1/private/labels.jsonl"; do
  if [[ ! -e "$required" ]]; then
    echo "missing E50E continuity prerequisite: $required" >&2
    exit 2
  fi
done
for fresh in "$ENDPOINT" "$ROUTE_RESULT" "$DECLARATION_RESULT" "$CERTIFICATE"; do
  if [[ -e "$fresh" ]]; then
    echo "fresh E50E continuity artifact required: $fresh" >&2
    exit 2
  fi
done
for frozen in \
  "config.json:ec4813b56d971f7a9c4490c7c148f997c468e97e9e8ae39e80f380d51a6dda1f" \
  "tokenizer.json:22bfef58b74d2fa40f402ac0c8d638d9e54d12137da2e2c7e9f32a1e7534e42f" \
  "model.safetensors.index.json:9f9e509271b4a2da37444f8a7ac3b3758279f6044c9c2f6eb40edc171329a65a"; do
  file="${frozen%%:*}"
  expected="${frozen#*:}"
  observed="$(sha256sum "$MODEL/$file" | cut -d' ' -f1)"
  if [[ "$observed" != "$expected" ]]; then
    echo "frozen Qwen72 identity mismatch for $file" >&2
    exit 2
  fi
done

export LD_LIBRARY_PATH="/n/fs/similarity/Illusion-of-Reasoning/openr1/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HF_HOME=/n/fs/similarity/tokenomics/.hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_CACHE_ROOT="/tmp/${USER}/e50e-qwen72-${SLURM_JOB_ID}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p \
  "$OUTPUT" \
  "$VLLM_CACHE_ROOT" \
  "$(dirname "$ROUTE_RESULT")" \
  "$(dirname "$DECLARATION_RESULT")" \
  "$(dirname "$CERTIFICATE")"

cleanup() {
  kill "${VLLM_PID:-}" 2>/dev/null || true
  if [[ -f "$ENDPOINT" ]] \
    && grep -q "\"job_id\": \"${SLURM_JOB_ID}\"" "$ENDPOINT"; then
    rm -f "$ENDPOINT"
  fi
  rm -f "$ENDPOINT_TMP"
}
trap cleanup EXIT

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
    echo "E50E Qwen72 exited before becoming ready" >&2
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
  echo "E50E Qwen72 did not become ready within 30 minutes" >&2
  exit 1
fi

printf '{"job_id": "%s", "model": "qwen2.5-72b", "base_url": "slurm://%s:%s/v1", "node": "node302", "port": %s, "tensor_parallel_size": %s, "max_model_len": %s, "max_num_seqs": %s, "enforce_eager": true, "structured_output_backend": "guidance", "gpu_memory_utilization": 0.88, "checkpoint_revision": "698703eae6604af048a3d2f509995dc302088217"}\n' \
  "$SLURM_JOB_ID" "$SLURM_JOB_ID" "$PORT" "$PORT" "$TP" "$MAX_LEN" "$MAX_NUM_SEQS" \
  >"$ENDPOINT_TMP"
mv "$ENDPOINT_TMP" "$ENDPOINT"

"$SCORE_PY" \
  "$ROOT_DIR/ops/math_strategy_calibration/score_e49t_route_confusion_calibration.py" \
  --cohort-dir "$ROOT_DIR/var/artifacts/e49t_route_confusion_calibration_v1" \
  --endpoint-record "$ENDPOINT" \
  --output "$ROUTE_RESULT"
"$SCORE_PY" \
  "$ROOT_DIR/ops/math_strategy_calibration/score_e49t_declaration_mismatch_calibration.py" \
  --artifact-dir "$ROOT_DIR/var/artifacts/e49t_declaration_mismatch_calibration_v1" \
  --endpoint-record "$ENDPOINT" \
  --output "$DECLARATION_RESULT"
"$SCORE_PY" \
  "$ROOT_DIR/ops/math_strategy_calibration/certify_e50e_qwen72_continuity.py" \
  --endpoint-record "$ENDPOINT" \
  --route-result "$ROUTE_RESULT" \
  --declaration-result "$DECLARATION_RESULT" \
  --expected-node node302 \
  --expected-port "$PORT" \
  --output "$CERTIFICATE"

materialization_job="$(sbatch --parsable \
  --export="ALL,E50E_ENDPOINT_RECORD=$ENDPOINT,E50E_ROUTE_CALIBRATION=$ROUTE_RESULT,E50E_DECLARATION_CALIBRATION=$DECLARATION_RESULT,E50E_CONTINUITY_CERTIFICATE=$CERTIFICATE" \
  "$ROOT_DIR/ops/slurm/e50e_materialize_exact_oat_node302.slurm")"
printf '%s\n' "$materialization_job" \
  >"$ROOT_DIR/var/artifacts/e50e_exact_oat_materialization_job.txt"
echo "submitted E50E materialization job $materialization_job"
wait "$VLLM_PID"
