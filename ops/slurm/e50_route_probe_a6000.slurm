#!/usr/bin/env bash
#SBATCH --job-name=e50-route-probe
#SBATCH --partition=lowprio
#SBATCH --account=allcs
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=var/artifacts/e50-route-probe-%j.out
#SBATCH --error=var/artifacts/e50-route-probe-%j.err

set -euo pipefail

: "${E50_ROUTE_DATA:?missing E50_ROUTE_DATA}"
: "${E50_ROUTE_MODEL:?missing E50_ROUTE_MODEL}"
: "${E50_ROUTE_OUT:?missing E50_ROUTE_OUT}"
: "${E50_ROUTE_SCHEMA:?missing E50_ROUTE_SCHEMA}"
: "${E50_ROUTE_LABEL:?missing E50_ROUTE_LABEL}"
: "${E50_ROUTE_ENDPOINT_RECORD:?missing E50_ROUTE_ENDPOINT_RECORD}"
: "${E50_ROUTE_ENDPOINT_SHA256:?missing E50_ROUTE_ENDPOINT_SHA256}"
: "${E50_ROUTE_ENDPOINT_NODE:?missing E50_ROUTE_ENDPOINT_NODE}"
: "${E50_ROUTE_ENDPOINT_PORT:?missing E50_ROUTE_ENDPOINT_PORT}"

ROOT_DIR=/n/fs/similarity/maxent-grpo
cd "$ROOT_DIR"
source "$ROOT_DIR/ops/repo_env.sh"

args=(
  --data-root "$E50_ROUTE_DATA"
  --model-root "$E50_ROUTE_MODEL"
  --out "$E50_ROUTE_OUT"
  --expected-data-schema "$E50_ROUTE_SCHEMA"
  --label "$E50_ROUTE_LABEL"
  --endpoint-record "$E50_ROUTE_ENDPOINT_RECORD"
  --expected-endpoint-sha256 "$E50_ROUTE_ENDPOINT_SHA256"
  --expected-endpoint-node "$E50_ROUTE_ENDPOINT_NODE"
  --expected-endpoint-port "$E50_ROUTE_ENDPOINT_PORT"
  --workers 4
  --timeout 1200
)
if [[ "${E50_ROUTE_REQUIRE_BASELINE:-0}" == 1 ]]; then
  args+=(--require-baseline-gate)
fi

exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$ROOT_DIR/ops/math_strategy_calibration/score_e50_route_probe.py" \
  "${args[@]}"
