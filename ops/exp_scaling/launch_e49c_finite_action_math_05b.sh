#!/usr/bin/env bash
# Launch E49C's matched finite-action MATH toy or full cohort.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

stage="${1:-}"
case "$stage" in
  config|toy|full) ;;
  *)
    echo "Usage: $0 {config|toy|full}" >&2
    exit 1
    ;;
esac

PYTHON_BIN="${OAT_ZERO_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
PROTOCOL="$ROOT_DIR/paper/preregistration/e49c_finite_action_math_haarnoja_05b.md"
ENDPOINT_RECORD="${E49C_QWEN72_ENDPOINT_RECORD:-$ROOT_DIR/var/artifacts/e49c_math_strategy_qwen72_v1/qwen72_endpoint.json}"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
TOY_ROOT="$ROOT_DIR/var/data/e49c_math_strategy_menu_toy"
FULL_ROOT="$ROOT_DIR/var/data/e49c_math_strategy_menu_full"
TOY_EVIDENCE="$ROOT_DIR/var/artifacts/e49c_math_strategy_menu_toy_v1"
FULL_EVIDENCE="$ROOT_DIR/var/artifacts/e49c_math_strategy_menu_full_v1"
MATERIALIZER="$ROOT_DIR/ops/math_strategy_calibration/materialize_e49c_strategy_menu_data.py"

if [[ ! -x "$PYTHON_BIN" ]] \
  || [[ ! -f "$PROTOCOL" ]] \
  || [[ ! -f "$MODEL_ROOT/model.safetensors" ]] \
  || ! grep -q '^\*\*Status: FROZEN BEFORE TRAINING; V3' "$PROTOCOL"; then
  echo "E49C frozen prerequisite is missing" >&2
  exit 1
fi

if [[ "$stage" == toy ]]; then
  DATA_ROOT="$TOY_ROOT"
  EVIDENCE_ROOT="$TOY_EVIDENCE"
  SOURCE_DATA="$ROOT_DIR/var/data/e49b_math_strategy_toy"
  MAX_TRAIN=50
  EVAL_INTERVAL=50
  SAVE_INTERVAL=50
  PREFIX=e49c_finite_action_math_toy_05b_v1
elif [[ "$stage" == full ]]; then
  DATA_ROOT="$FULL_ROOT"
  EVIDENCE_ROOT="$FULL_EVIDENCE"
  SOURCE_DATA="$ROOT_DIR/var/data/math12k_384_math500"
  MAX_TRAIN=384
  EVAL_INTERVAL=384
  SAVE_INTERVAL=384
  PREFIX=e49c_finite_action_math_full_05b_v1
else
  DATA_ROOT="$TOY_ROOT"
  EVIDENCE_ROOT="$TOY_EVIDENCE"
  SOURCE_DATA="$ROOT_DIR/var/data/e49b_math_strategy_toy"
  MAX_TRAIN=50
  EVAL_INTERVAL=50
  SAVE_INTERVAL=50
  PREFIX=e49c_finite_action_math_config_05b_v1
fi

if [[ "$stage" == toy || "$stage" == full ]]; then
  if [[ ! -f "$ENDPOINT_RECORD" ]]; then
    echo "The frozen Qwen72 judge endpoint is not live" >&2
    exit 1
  fi
  "$PYTHON_BIN" "$MATERIALIZER" \
    --source "$SOURCE_DATA" \
    --output "$DATA_ROOT" \
    --evidence "$EVIDENCE_ROOT" \
    --endpoint "$ENDPOINT_RECORD" \
    --audit-only >/dev/null
fi

if [[ "$stage" == full ]]; then
  TOY_DECISION="$ROOT_DIR/var/artifacts/e49c_finite_action_math_toy_advancement.json"
  if [[ ! -f "$TOY_DECISION" ]] \
    || [[ "$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("advance_to_full",False))' "$TOY_DECISION")" != True ]]; then
    echo "E49C hard-toy advancement gate has not passed" >&2
    exit 1
  fi
fi

if [[ "$stage" != config ]] \
  && { [[ ! -f "$DATA_ROOT/train/dataset_dict.json" ]] \
    || [[ ! -f "$DATA_ROOT/eval/dataset_dict.json" ]]; }; then
  echo "E49C menu data artifact is incomplete: $DATA_ROOT" >&2
  exit 1
fi

hash_tree() {
  local tree="$1"
  "$PYTHON_BIN" - "$tree" <<'PY'
import hashlib
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
digest = hashlib.sha256()
for item in sorted(path for path in root.rglob("*") if path.is_file()):
    digest.update(str(item.relative_to(root)).encode("utf-8"))
    digest.update(b"\0")
    digest.update(hashlib.sha256(item.read_bytes()).digest())
print(digest.hexdigest())
PY
}

if [[ "$stage" == config ]]; then
  TRAIN_DATA_HASH=pending
  EVAL_DATA_HASH=pending
  MENU_EVIDENCE_HASH=pending
else
  TRAIN_DATA_HASH="$(hash_tree "$DATA_ROOT/train")"
  EVAL_DATA_HASH="$(hash_tree "$DATA_ROOT/eval")"
  MENU_EVIDENCE_HASH="$(sha256sum "$EVIDENCE_ROOT/menu_records.jsonl" | cut -d' ' -f1)"
  read -r EXPECTED_TRAIN_HASH EXPECTED_EVAL_HASH EXPECTED_MENU_COUNT <<<"$(
    "$PYTHON_BIN" - "$DATA_ROOT/MATERIALIZATION_MANIFEST.json" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
if (
    manifest.get("schema") != "e49c_strategy_menu_materialization_v1"
    or manifest.get("all_menus_two_audit_passes") is not True
):
    raise SystemExit("invalid E49C menu manifest")
print(
    manifest["train_tree_sha256"],
    manifest["eval_tree_sha256"],
    manifest["menu_count"],
)
PY
  )"
  if [[ "$TRAIN_DATA_HASH" != "$EXPECTED_TRAIN_HASH" ]] \
    || [[ "$EVAL_DATA_HASH" != "$EXPECTED_EVAL_HASH" ]]; then
    echo "E49C menu data identity mismatch" >&2
    exit 1
  fi
  EXPECTED_ROWS=100
  [[ "$stage" == full ]] && EXPECTED_ROWS=884
  if [[ "$EXPECTED_MENU_COUNT" != "$EXPECTED_ROWS" ]]; then
    echo "E49C menu evidence row-count mismatch" >&2
    exit 1
  fi
fi

SOURCE_HASH="$(hash_tree "$ROOT_DIR/src")"
SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e49c_finite_action_math_${SOURCE_HASH}/src"
if [[ ! -f "$SOURCE_ROOT/oat_drgrpo/__init__.py" ]]; then
  mkdir -p "$(dirname "$SOURCE_ROOT")"
  staging="$(mktemp -d "$(dirname "$SOURCE_ROOT")/.source.XXXXXX")"
  mkdir -p "$staging/src"
  cp -a "$ROOT_DIR/src/." "$staging/src/"
  mv "$staging/src" "$SOURCE_ROOT"
  rmdir "$staging"
fi
if [[ "$(hash_tree "$SOURCE_ROOT")" != "$SOURCE_HASH" ]]; then
  echo "E49C source snapshot hash mismatch" >&2
  exit 1
fi

OPS_INPUT="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.e49c-ops.XXXXXX")"
mkdir -p "$OPS_INPUT/slurm"
for file in repo_env.sh run_experiment.sh train.sh resolve_eval_cadence.py \
  submit_countdown_comparative.sh; do
  cp "$ROOT_DIR/ops/$file" "$OPS_INPUT/$file"
done
cp "$ROOT_DIR/ops/slurm/train_node302.slurm" "$OPS_INPUT/slurm/train_node302.slurm"
OPS_HASH="$(hash_tree "$OPS_INPUT")"
OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e49c_finite_action_math_ops_${OPS_HASH}/ops"
if [[ ! -f "$OPS_ROOT/submit_countdown_comparative.sh" ]]; then
  mkdir -p "$(dirname "$OPS_ROOT")"
  staging="$(mktemp -d "$(dirname "$OPS_ROOT")/.ops.XXXXXX")"
  mv "$OPS_INPUT" "$staging/ops"
  mv "$staging/ops" "$OPS_ROOT"
  rmdir "$staging"
else
  find "$OPS_INPUT" -type f -delete
  rmdir "$OPS_INPUT/slurm" "$OPS_INPUT"
fi

MANIFEST="$ROOT_DIR/var/artifacts/${PREFIX}_comparative_jobs.tsv"
IDENTITY="$ROOT_DIR/var/artifacts/${PREFIX}_identity.json"
if [[ "$stage" != config && -e "$MANIFEST" ]]; then
  echo "Fresh E49C stage required; manifest already exists: $MANIFEST" >&2
  exit 1
fi

if [[ "$stage" == config ]]; then
  ENDPOINT=http://node105:8769/v1
  ENDPOINT_RECORD_HASH=pending
else
  ENDPOINT="$("$PYTHON_BIN" - "$ENDPOINT_RECORD" <<'PY'
import json
import sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {
    "model": "qwen2.5-72b",
    "node": "node105",
    "tensor_parallel_size": 4,
    "max_model_len": 32768,
    "max_num_seqs": 8,
    "enforce_eager": True,
    "checkpoint_revision": "698703eae6604af048a3d2f509995dc302088217",
}
if any(record.get(key) != value for key, value in expected.items()):
    raise SystemExit("unexpected E49C judge endpoint")
print(f"http://{record['node']}:{int(record['port'])}/v1")
PY
  )"
  ENDPOINT_RECORD_HASH="$(sha256sum "$ENDPOINT_RECORD" | cut -d' ' -f1)"
fi

export RUN_STAMP_PREFIX="$PREFIX"
export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_OPS_SNAPSHOT_ROOT="$OPS_ROOT"
export OAT_ZERO_CAMPAIGN_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_PRETRAIN="$MODEL_ROOT"
export OAT_ZERO_PROTOCOL_IDENTITY="$IDENTITY"
export OAT_ZERO_REQUIRE_EXISTING_DATA=1
export OAT_ZERO_COMPARATIVE_REBUILD=0
export OAT_ZERO_APPEND_MANIFEST=0
export OAT_ZERO_COMPARATIVE_TASK=math
export OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-0.5b-instruct
export OAT_ZERO_COMPARATIVE_DATA_ROOT="$DATA_ROOT"
export OAT_ZERO_TRAIN_SEEDS=45
export OAT_ZERO_ONLY_ARMS=grpo,online_canonical_haarnoja
export OAT_ZERO_XDR_TAUS=""
export OAT_ZERO_INCLUDE_TOKEN_ENTROPY_ARM=0
export OAT_ZERO_INCLUDE_SEED_ARM=0
export OAT_ZERO_INCLUDE_XDR_ADAPT_ARM=0
export OAT_ZERO_INCLUDE_XDR_TAU_CONTROL_ARM=0
export OAT_ZERO_INCLUDE_XDR_SAC_DUAL_ARM=0
export OAT_ZERO_INCLUDE_MAXENT_ARM=0
export OAT_ZERO_INCLUDE_MAXENT_CONTROL_ARM=0
export OAT_ZERO_INCLUDE_MAXENT_DUAL_ARM=0
export OAT_ZERO_INCLUDE_MAXENT_LENGTH_DUAL_ARM=0
export OAT_ZERO_INCLUDE_DIAYN_ARM=0
export OAT_ZERO_INCLUDE_OUTCOME_COLLISION_ARM=0
export OAT_ZERO_INCLUDE_SEMANTIC_SHANNON_ARM=0
export OAT_ZERO_INCLUDE_ONLINE_CANONICAL_MAXENT_ARM=0
export OAT_ZERO_INCLUDE_ONLINE_CANONICAL_HAARNOJA_ARM=1

export OAT_ZERO_VERIFIED_DISCOVERY_TRACKING=1
export OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.10
export OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.50
export OAT_ZERO_ONLINE_CANONICAL_BANK_PSEUDOCOUNT=1.0
export OAT_ZERO_ONLINE_CANONICAL_BANK_SURPRISAL_CLIP=5.0
export OAT_ZERO_ONLINE_CANONICAL_KEY_MODE=math_strategy_qwen72
export OAT_ZERO_ONLINE_CANONICAL_DUAL_TARGET_RATIO=0.80
export OAT_ZERO_ONLINE_CANONICAL_DUAL_MIN_ALPHA=0.10
export OAT_ZERO_ONLINE_CANONICAL_DUAL_MAX_ALPHA=0.50
export OAT_ZERO_ONLINE_CANONICAL_DUAL_ALPHA_LR=0.003
export OAT_ZERO_ONLINE_CANONICAL_DUAL_EMA_DECAY=0.90
export OAT_ZERO_MATH_STRATEGY_ENDPOINT="$ENDPOINT"
export OAT_ZERO_MATH_STRATEGY_MODEL=qwen2.5-72b
export OAT_ZERO_MATH_STRATEGY_TIMEOUT_SECONDS=600
export OAT_ZERO_MATH_STRATEGY_WORKERS=4
export OAT_ZERO_MATH_STRATEGY_MAX_ITEM_CHARS=4000
export OAT_ZERO_MATH_STRATEGY_GATE_TASK_REWARD=1

export OAT_ZERO_NUM_SAMPLES=16
export OAT_ZERO_LEARNING_RATE=0.0000002
export OAT_ZERO_MAX_TRAIN="$MAX_TRAIN"
export OAT_ZERO_MAX_QUERIES=100000000
export OAT_ZERO_MAX_PROMPT_EPOCHS=3
export OAT_ZERO_NUM_PROMPT_EPOCH=3
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

export OAT_ZERO_PROMPT_TEMPLATE=qwen_math
export OAT_ZERO_INPUT_KEY=problem
export OAT_ZERO_OUTPUT_KEY=answer
export OAT_ZERO_EVAL_INPUT_KEY=problem
export OAT_ZERO_EVAL_OUTPUT_KEY=answer
export OAT_ZERO_TEST_SPLIT=math
export OAT_ZERO_VERIFIER_VERSION=math_verify
export OAT_ZERO_PROMPT_MAX_LENGTH=2048
export OAT_ZERO_GENERATE_MAX_LENGTH=1024
export OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=1024
export OAT_ZERO_MAX_MODEL_LEN=3072
export OAT_ZERO_TEMPERATURE=1
export OAT_ZERO_TOP_P=1
export OAT_ZERO_EVAL_TEMPERATURE=0
export OAT_ZERO_EVAL_MODE_COVERAGE_K=8
export OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE=1
export OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=1
export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=491900
export OAT_ZERO_EVAL_BATCH_SIZE=64
export OAT_ZERO_ALLOW_SPARSE_EVAL=1
export OAT_ZERO_EVAL_PROMPT_INTERVAL="$EVAL_INTERVAL"
export OAT_ZERO_EVAL_STEPS="$EVAL_INTERVAL"
export OAT_ZERO_SYNC_PARAMS_EVERY=1

export OAT_ZERO_CANONICAL_ACTION_TASK=none
export OAT_ZERO_CANONICAL_GRAPH_ACTIONS=0
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

export OAT_ZERO_SAVE_CKPT=1
export OAT_ZERO_SAVE_STEPS="$SAVE_INTERVAL"
export OAT_ZERO_SAVE_FROM="$SAVE_INTERVAL"
export OAT_ZERO_MAX_SAVE_NUM=2
export OAT_ZERO_RESUME_STEPS="$SAVE_INTERVAL"
export OAT_ZERO_RESUME_FROM="$SAVE_INTERVAL"
export OAT_ZERO_MAX_RESUME_NUM=2
export OAT_ZERO_EXPORT_STEPS=0
export OAT_ZERO_EXPORT_FROM=0
export OAT_ZERO_MAX_EXPORT_NUM=1
export OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=0
export OAT_ZERO_AUTO_RESUME=1
export OAT_ZERO_WATCHDOG_REQUEUE=1
export OAT_ZERO_WATCHDOG_MAX_RESTARTS=6

export OAT_ZERO_TRAIN_NODELIST=node302
export OAT_ZERO_TRAIN_GRES=gpu:a100:1
export OAT_ZERO_TRAIN_CPUS_PER_TASK=16
export OAT_ZERO_TRAIN_PARTITION=mltheory
export OAT_ZERO_TRAIN_ACCOUNT=mltheory
export OAT_ZERO_TRAIN_MEMORY=96G
export OAT_ZERO_TRAIN_TIME_LIMIT=24:00:00

PROTOCOL_HASH="$(sha256sum "$PROTOCOL" | cut -d' ' -f1)"
CORE_HASH="$(sha256sum \
  "$ROOT_DIR/src/oat_drgrpo/math_strategy_menu.py" \
  "$ROOT_DIR/src/oat_drgrpo/math_strategy_canonicalizer.py" \
  "$ROOT_DIR/src/oat_drgrpo/online_canonical_bank.py" \
  "$ROOT_DIR/src/oat_drgrpo/online_canonical_controller.py" \
  "$ROOT_DIR/src/oat_drgrpo/learner/grpo.py" \
  "$ROOT_DIR/src/oat_drgrpo/args.py" \
  "$MATERIALIZER" | sha256sum | cut -d' ' -f1)"

"$PYTHON_BIN" - "$IDENTITY" "$stage" "$SOURCE_HASH" "$OPS_HASH" \
  "$ENDPOINT" "$ENDPOINT_RECORD_HASH" "$TRAIN_DATA_HASH" "$EVAL_DATA_HASH" \
  "$MENU_EVIDENCE_HASH" "$PROTOCOL_HASH" "$CORE_HASH" <<'PY'
import json
import os
import pathlib
import sys
import tempfile
path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "e49c_finite_action_math_haarnoja_05b_v1",
    "stage": sys.argv[2],
    "source_hash": sys.argv[3],
    "ops_hash": sys.argv[4],
    "judge_endpoint": sys.argv[5],
    "judge_endpoint_record_sha256": sys.argv[6],
    "train_data_sha256": sys.argv[7],
    "eval_data_sha256": sys.argv[8],
    "menu_evidence_sha256": sys.argv[9],
    "protocol_sha256": sys.argv[10],
    "core_contract_sha256": sys.argv[11],
    "arms": ["grpo", "online_canonical_haarnoja"],
    "seed": 45,
    "prompt_epochs": 3,
    "num_samples": 16,
    "task_reward_gate": "exact_declaration_and_two_execution_audits",
    "method": {
        "canonical_support": "prompt_local_audited_action_combo",
        "integrity_and_execution_judge_passes": 2,
        "permutation_seeds": [470721, 470722],
        "canonicalizer_state_schema": (
            "math_strategy_canonicalizer_menu_bound_v17"
        ),
        "target_ratio": 0.80,
        "alpha": [0.10, 0.50],
        "alpha_lr": 0.003,
        "ema_decay": 0.90,
        "novelty_beta": 0.50,
    },
    "data_root": os.environ["OAT_ZERO_COMPARATIVE_DATA_ROOT"],
}
path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY

if [[ "$stage" == config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  "$OPS_ROOT/submit_countdown_comparative.sh"
  echo "[e49c] configuration passed; no jobs submitted"
  exit 0
fi

unset OAT_ZERO_COMPARATIVE_CONFIG_ONLY
"$OPS_ROOT/submit_countdown_comparative.sh"
rows="$(awk -F '\t' 'NR > 1 {count++} END {print count+0}' "$MANIFEST")"
if [[ "$rows" != 2 ]]; then
  echo "E49C expected two matched jobs, observed $rows" >&2
  exit 1
fi
echo "[e49c] submitted matched ${stage} cohort"
