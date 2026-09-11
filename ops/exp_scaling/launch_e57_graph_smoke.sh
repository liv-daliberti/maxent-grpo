#!/usr/bin/env bash
# Launch E57's direct-MaxEnt-off Graph multi-mode mechanism smoke.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

phase="${1:-}"
case "$phase" in
  config|graph) ;;
  *)
    echo "Usage: $0 {config|graph}" >&2
    exit 1
    ;;
esac

PYTHON_BIN="${OAT_ZERO_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
PROTOCOL="$ROOT_DIR/paper/preregistration/e57_verified_first_split_canonical_05b.md"
AUDITOR="$ROOT_DIR/ops/exp_scaling/audit_e57_graph_smoke.py"
PYTHON_IDENTITY="$ROOT_DIR/var/artifacts/e57_verified_first_split_python_smoke_identity.json"
PYTHON_AUDIT="$ROOT_DIR/var/artifacts/e57_python_smoke_audit_latest.json"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
DATA_ROOT="$ROOT_DIR/var/data/exact_answer_mode_probe"
PREFIX=e57_verified_first_split_smoke_graph_seed9057
MANIFEST="$ROOT_DIR/var/artifacts/${PREFIX}_comparative_jobs.tsv"
IDENTITY="$ROOT_DIR/var/artifacts/e57_verified_first_split_graph_smoke_identity.json"

for required in \
  "$PROTOCOL" "$AUDITOR" "$PYTHON_IDENTITY" "$PYTHON_AUDIT" \
  "$MODEL_ROOT/config.json" \
  "$DATA_ROOT/train/dataset_dict.json" \
  "$DATA_ROOT/eval/dataset_dict.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing E57 graph prerequisite: $required" >&2
    exit 1
  fi
done

"$PYTHON_BIN" - "$PYTHON_IDENTITY" "$PYTHON_AUDIT" <<'PY'
import hashlib
import json
import pathlib
import sys

identity_path = pathlib.Path(sys.argv[1])
audit_path = pathlib.Path(sys.argv[2])
identity = json.loads(identity_path.read_text(encoding="utf-8"))
audit = json.loads(audit_path.read_text(encoding="utf-8"))
if audit.get("status") != "pass" or audit.get("violations"):
    raise SystemExit("E57 graph smoke requires a clean terminal Python smoke")
expected = identity.get("auditor_sha256")
actual = hashlib.sha256(
    pathlib.Path("ops/exp_scaling/audit_e57_smoke.py").read_bytes()
).hexdigest()
if expected != actual:
    raise SystemExit("E57 Python smoke auditor binding changed")
PY

if [[ "$phase" == graph ]]; then
  for fresh in "$MANIFEST" "$IDENTITY"; do
    if [[ -e "$fresh" ]]; then
      echo "Fresh E57 graph artifact required; already exists: $fresh" >&2
      exit 1
    fi
  done
fi

SOURCE_HASH="$(jq -r '.source_hash' "$PYTHON_IDENTITY")"
EXECUTION_HASH="$(jq -r '.execution_surface_hash' "$PYTHON_IDENTITY")"
SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e57_verified_first_${SOURCE_HASH}/src"
OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e57_verified_first_ops_${EXECUTION_HASH}/ops"
if [[ "$phase" == config ]]; then
  SOURCE_ROOT="$ROOT_DIR/src"
  OPS_ROOT="$ROOT_DIR/ops"
fi
if [[ ! -f "$SOURCE_ROOT/oat_drgrpo/__init__.py" ]] \
  || [[ ! -f "$OPS_ROOT/submit_countdown_comparative.sh" ]]; then
  echo "E57 frozen source or execution snapshot is missing" >&2
  exit 1
fi

write_identity() {
  local job_id="$1"
  local protocol_hash launcher_hash auditor_hash manifest_hash python_audit_hash
  protocol_hash="$(sha256sum "$PROTOCOL" | cut -d' ' -f1)"
  launcher_hash="$(sha256sum "$0" | cut -d' ' -f1)"
  auditor_hash="$(sha256sum "$AUDITOR" | cut -d' ' -f1)"
  manifest_hash="$(sha256sum "$MANIFEST" | cut -d' ' -f1)"
  python_audit_hash="$(sha256sum "$PYTHON_AUDIT" | cut -d' ' -f1)"
  "$PYTHON_BIN" - \
    "$IDENTITY" "$protocol_hash" "$launcher_hash" "$auditor_hash" \
    "$manifest_hash" "$python_audit_hash" "$SOURCE_HASH" "$EXECUTION_HASH" \
    "$job_id" <<'PY'
import json
import os
import pathlib
import sys
import tempfile

path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "e57_verified_first_split_graph_smoke_v1",
    "protocol_sha256": sys.argv[2],
    "launcher_sha256": sys.argv[3],
    "auditor_sha256": sys.argv[4],
    "manifest_sha256": sys.argv[5],
    "python_smoke_audit_sha256": sys.argv[6],
    "source_hash": sys.argv[7],
    "execution_surface_hash": sys.argv[8],
    "job_id": int(sys.argv[9]),
    "attempt_selection": "exact_manifest_job_id",
    "domain": "graph_coloring",
    "seed": 9057,
    "max_updates": 32,
    "num_samples": 16,
    "direct_token_entropy": {"coefficient": 0.0, "controller": None},
    "information_firewall": {
        "gold_support_feedback": False,
        "evaluation_feedback": False,
        "desired_entropy": None,
        "desired_mode_count": None,
    },
}
path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY
}

export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_OPS_SNAPSHOT_ROOT="$OPS_ROOT"
export OAT_ZERO_CAMPAIGN_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_PRETRAIN="$MODEL_ROOT"
export OAT_ZERO_REQUIRE_EXISTING_DATA=1
export OAT_ZERO_COMPARATIVE_REBUILD=0
export OAT_ZERO_APPEND_MANIFEST=0
export RUN_STAMP_PREFIX="$PREFIX"
export OAT_ZERO_COMPARATIVE_TASK=graph_coloring
export OAT_ZERO_COMPARATIVE_DATA_ROOT="$DATA_ROOT"
export OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-0.5b-instruct
export OAT_ZERO_TRAIN_SEEDS=9057
export OAT_ZERO_ONLY_ARMS=verified_first_split_canonical
export OAT_ZERO_XDR_TAUS=""
for flag in \
  TOKEN_ENTROPY SEED XDR_ADAPT XDR_TAU_CONTROL XDR_SAC_DUAL \
  MAXENT MAXENT_CONTROL MAXENT_DUAL MAXENT_LENGTH_DUAL DIAYN \
  OUTCOME_COLLISION OUTCOME_COLLISION_OUTSIDE_CENTERING \
  SEMANTIC_SHANNON SEMANTIC_SHANNON_ADVANTAGE \
  QUALITY_GATED_SEMANTIC_NOVELTY \
  SUCCESS_CONDITIONED_SIGNED_SEMANTIC_SHANNON \
  SIGNAL_FIRST_SEMANTIC_BALANCE ONLINE_CANONICAL_MAXENT \
  ONLINE_CANONICAL_HAARNOJA ONLINE_CANONICAL_POLICY_ENTROPY \
  MAXENT_INVERSE MAXENT_INVERSE_CANONICAL \
  MAXENT_INVERSE_CANONICAL_REPLAY OPEN_SET_SPLIT_CANONICAL; do
  export "OAT_ZERO_INCLUDE_${flag}_ARM=0"
done
export OAT_ZERO_INCLUDE_VERIFIED_FIRST_SPLIT_CANONICAL_ARM=1
export OAT_ZERO_MAXENT_ALPHA=0
export OAT_ZERO_MAXENT_INVERSE_BASE_ALPHA=0
export OAT_ZERO_SEMANTIC_SHANNON_COEF=0.10
export OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP=5
export OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT=1
export OAT_ZERO_SEMANTIC_SHANNON_OPEN_SET_WARMUP_STEPS=64
export OAT_ZERO_SEMANTIC_SHANNON_OPEN_SET_EMA_DECAY=0.90
export OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0
export OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.50
export OAT_ZERO_ONLINE_CANONICAL_BANK_PSEUDOCOUNT=1
export OAT_ZERO_ONLINE_CANONICAL_BANK_SURPRISAL_CLIP=5
export OAT_ZERO_ONLINE_CANONICAL_KEY_MODE=modebench_outcome
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=0.10
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_CAPACITY=16
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_WARMUP_STEPS=64
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_EMA_DECAY=0.90
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_ALPHA=0.10
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_WARMUP_STEPS=64
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_EMA_DECAY=0.90
export OAT_ZERO_NUM_SAMPLES=16
export OAT_ZERO_LEARNING_RATE=0.0000002
export OAT_ZERO_MAX_TRAIN=32
export OAT_ZERO_MAX_QUERIES=100000000
export OAT_ZERO_MAX_PROMPT_EPOCHS=1
export OAT_ZERO_NUM_PROMPT_EPOCH=1
export OAT_ZERO_NUM_PPO_EPOCHS=1
export OAT_ZERO_MAX_NORM=1
export OAT_ZERO_BETA=0
export OAT_ZERO_IGNORE_NO_EOS=0
export OAT_ZERO_TRAIN_BATCH_SIZE=16
export OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=1
export OAT_ZERO_ROLLOUT_BATCH_SIZE=1
export OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE=1
export OAT_ZERO_N_GPU=1
export OAT_ZERO_NUM_GPUS_PER_ACTOR=1
export OAT_ZERO_PROMPT_TEMPLATE=qwen_boxed
export OAT_ZERO_INPUT_KEY=problem
export OAT_ZERO_OUTPUT_KEY=answer
export OAT_ZERO_EVAL_INPUT_KEY=problem
export OAT_ZERO_EVAL_OUTPUT_KEY=answer
export OAT_ZERO_TEST_SPLIT=multi_answer
export OAT_ZERO_VERIFIER_VERSION=fast
export OAT_ZERO_PROMPT_MAX_LENGTH=256
export OAT_ZERO_GENERATE_MAX_LENGTH=192
export OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=192
export OAT_ZERO_MAX_MODEL_LEN=512
export OAT_ZERO_TEMPERATURE=1
export OAT_ZERO_TOP_P=1
export OAT_ZERO_EVAL_MODE_COVERAGE_K=0
export OAT_ZERO_EVAL_PROMPT_INTERVAL=1000000
export OAT_ZERO_SYNC_PARAMS_EVERY=1
export OAT_ZERO_ZERO_STAGE=2
export OAT_ZERO_VLLM_GPU_RATIO=0.25
export OAT_ZERO_ENABLE_FLASH_ATTN=0
export OAT_ZERO_ADAM_OFFLOAD=0
export OAT_ZERO_ACTIVATION_OFFLOADING=0
export OAT_ZERO_COLLOCATE=1
export OAT_ZERO_VLLM_SLEEP=1
export VLLM_USE_V1=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OAT_ZERO_SAVE_CKPT=0
export OAT_ZERO_AUTO_RESUME=0
export OAT_ZERO_WATCHDOG_REQUEUE=0
export OAT_ZERO_TRAIN_CPUS_PER_TASK=8
export OAT_ZERO_TRAIN_MEMORY=64G
export OAT_ZERO_TRAIN_TIME_LIMIT=04:00:00
export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E57_GRAPH_NODELIST:-node103,node104,node208}"
export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E57_GRAPH_GRES:-gpu:a6000:1}"
export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E57_GRAPH_PARTITION:-lowprio}"
export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E57_GRAPH_ACCOUNT:-mltheory}"

if [[ "$phase" == config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  export OAT_ZERO_SBATCH_HOLD=0
  "$OPS_ROOT/submit_countdown_comparative.sh"
  echo "[e57] Graph multi-mode smoke configuration passed"
  exit 0
fi

export OAT_ZERO_PROTOCOL_IDENTITY="$IDENTITY"
export OAT_ZERO_SBATCH_HOLD=1
job_id=""
released=0
cleanup_held() {
  local status="$?"
  trap - EXIT
  if [[ "$released" != "1" && "$job_id" =~ ^[0-9]+$ ]]; then
    scancel "$job_id" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup_held EXIT
"$OPS_ROOT/submit_countdown_comparative.sh"
mapfile -t job_ids < <(
  awk -F '\t' 'NR > 1 && $3 ~ /^[0-9]+$/ {print $3}' "$MANIFEST"
)
if [[ "${#job_ids[@]}" -ne 1 ]]; then
  echo "E57 Graph smoke has ${#job_ids[@]} jobs; expected one" >&2
  exit 1
fi
job_id="${job_ids[0]}"
write_identity "$job_id"
scontrol update "JobId=$job_id" Requeue=0
scontrol release "$job_id"
released=1
trap - EXIT
echo "[e57] released Graph multi-mode smoke job $job_id"
echo "[e57] identity=$IDENTITY"
