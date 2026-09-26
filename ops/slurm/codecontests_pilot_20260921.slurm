#!/usr/bin/env bash
#SBATCH --job-name=ccplus-pilot7b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --no-requeue

set -euo pipefail

ROOT_DIR="${ROOT_DIR:?}"
PILOT_ROOT="${PILOT_ROOT:?}"
PILOT_BUNDLE="${PILOT_BUNDLE:?}"
PILOT_SOURCE_HASH="${PILOT_SOURCE_HASH:?}"
PILOT_EXECUTION_HASH="${PILOT_EXECUTION_HASH:?}"
export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
source "$PILOT_BUNDLE/repo_env.sh"
export OAT_ZERO_SOURCE_ROOT="$PILOT_BUNDLE/src"
export OAT_ZERO_TESTLIB_ROOT="$PILOT_BUNDLE/testlib"
export OAT_ZERO_SANDBOX_SOURCE="$PILOT_BUNDLE/ops/constructive_code_sandbox.c"
export OAT_ZERO_CONSTRUCTIVE_V5_PROTOCOL="$PILOT_BUNDLE/inputs/constructive_code_executable_slate_v5_20260730.md"
export OAT_ZERO_CONSTRUCTIVE_V5_INDEX="$PILOT_BUNDLE/inputs/candidate_source_index.json"
export OAT_ZERO_CONSTRUCTIVE_V5_V1_ROOT="$ROOT_DIR/var/data/constructive_code_review_slate_v1"
export PYTHONPATH="$PILOT_BUNDLE/ops:$PILOT_BUNDLE/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V1=0
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4

"$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" - "$PILOT_ROOT/identity.json" "$PILOT_SOURCE_HASH" "$PILOT_EXECUTION_HASH" "$SLURM_JOB_ID" <<'PYPIN'
import hashlib, json, pathlib, sys
identity = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert identity['source_hash'] == sys.argv[2]
assert identity['execution_hash'] == sys.argv[3]
assert identity['job_id'] == int(sys.argv[4])
assert identity['model_revision'] == 'c03e6d358207e414f1eca0bb1891e29f1db0e242'
for row in identity['files']:
    assert hashlib.sha256(pathlib.Path(row['snapshot']).read_bytes()).hexdigest() == row['sha256'], row['snapshot']
print('Frozen pilot bundle verified.', flush=True)
PYPIN

PILOT_WORK="${TMPDIR}/ccplus-pilot7b-${SLURM_JOB_ID:?}"
mkdir -p "$PILOT_WORK/build" "$PILOT_WORK/scratch"

exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$PILOT_BUNDLE/ops/evaluate_constructive_code_pilot_20260921.py" \
  --slate-root "$ROOT_DIR/var/data/constructive_code_v5" \
  --v1-root "$ROOT_DIR/var/data/constructive_code_review_slate_v1" \
  --image "$ROOT_DIR/var/images/python-3.10-slim-c1e4e6c01eb4.sqsh" \
  --runtime-root "$PILOT_WORK/runtime" --launcher "$PILOT_WORK/constructive_code_sandbox" \
  --build-root "$PILOT_WORK/build" --scratch-root "$PILOT_WORK/scratch" \
  --model "$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/c03e6d358207e414f1eca0bb1891e29f1db0e242" \
  --gate-audit "$PILOT_BUNDLE/inputs/constructive_code_v6_gate_audit.json" \
  --gate-identity "$PILOT_BUNDLE/inputs/constructive_code_v6_gate_identity.json" \
  --gate-replays "$PILOT_BUNDLE/inputs/constructive_code_v5_replays.jsonl" \
  --protocol "$PILOT_BUNDLE/docs/codecontests_pilot_protocol_20260921.md" \
  --prompt-overlay "$PILOT_BUNDLE/docs/codecontests_pilot_prompt_overlay_20260921.json" \
  --identity "$PILOT_ROOT/identity.json" \
  --output "$PILOT_ROOT/capability.json" \
  --sample-count 64 --prefix-count 16 --seed 77101 \
  --temperature 1.0 --top-p 1.0 --max-tokens 1024 \
  --max-model-len 8192 --generation-batch-size 16 --execution-workers 8 \
  --source-hash "$PILOT_SOURCE_HASH" --execution-hash "$PILOT_EXECUTION_HASH" \
  --job-id "$SLURM_JOB_ID"
