#!/usr/bin/env bash
# Launch E49V's matched exact-OAT 384-train/MATH-500 three-epoch cohort.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

mode="${1:-run}"
if [[ "$mode" != config && "$mode" != run ]]; then
  echo "Usage: $0 [config|run]" >&2
  exit 1
fi

PYTHON_BIN="${OAT_ZERO_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
PROTOCOL="${E49V_PROTOCOL:-$ROOT_DIR/paper/preregistration/e49v_exact_oat_natural_menu_full_20260726.md}"
DATA_ROOT="${E49V_DATA_ROOT:-$ROOT_DIR/var/data/e49v_exact_oat_natural_menu_full}"
DATA_MANIFEST="$DATA_ROOT/MATERIALIZATION_MANIFEST.json"
DATA_EVIDENCE="${E49V_DATA_EVIDENCE:-$ROOT_DIR/var/artifacts/e49v_exact_oat_natural_menu_full_v1}"
MATERIALIZER="${E49V_MATERIALIZER:-$ROOT_DIR/ops/math_strategy_calibration/materialize_e49v_exact_oat_natural_menu.py}"
TOY_ADVANCEMENT="${E49V_TOY_ADVANCEMENT:-$ROOT_DIR/var/artifacts/e49t_natural_menu_math_toy_advancement_v1.json}"
TOY_ADVANCEMENT_SCHEMA="${E49V_TOY_ADVANCEMENT_SCHEMA:-e49t_natural_menu_math_toy_advancement_v1}"
DATA_SCHEMA="${E49V_DATA_SCHEMA:-e49v_exact_oat_natural_menu_materialization_v1}"
EXPECTED_SINGLETON_TOTAL="${E49V_SINGLETON_TOTAL:-864}"
EXPECTED_MULTI_TOTAL="${E49V_MULTI_TOTAL:-20}"
EXPECTED_TRAIN_MULTI="${E49V_TRAIN_MULTI:-10}"
EXPECTED_EVAL_MULTI="${E49V_EVAL_MULTI:-10}"
CALIBRATION="${E49V_ROUTE_CALIBRATION_RESULT:-$ROOT_DIR/var/artifacts/e49t_route_confusion_calibration_v1/result.json}"
COHORT_IDENTITY="${E49V_ROUTE_CALIBRATION_IDENTITY:-$ROOT_DIR/var/artifacts/e49t_route_confusion_calibration_v1/frozen_identity.json}"
DECLARATION_CALIBRATION="${E49V_DECLARATION_CALIBRATION_RESULT:-$ROOT_DIR/var/artifacts/e49t_declaration_mismatch_calibration_v1/result.json}"
DECLARATION_IDENTITY="${E49V_DECLARATION_CALIBRATION_IDENTITY:-$ROOT_DIR/var/artifacts/e49t_declaration_mismatch_calibration_v1/frozen_identity.json}"
ENDPOINT_RECORD="${E49V_QWEN72_ENDPOINT_RECORD:-$ROOT_DIR/var/artifacts/e49t_qwen72_node302_v1/qwen72_endpoint.json}"
EXPECTED_ENDPOINT_NODE="${E49V_QWEN72_NODE:-node302}"
EXPECTED_ENDPOINT_PORT="${E49V_QWEN72_PORT:-8770}"
CONTINUITY_CERTIFICATE="${E49V_CONTINUITY_CERTIFICATE:-}"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
SOURCE_ROOT_OVERRIDE="${E49V_SOURCE_ROOT_OVERRIDE:-}"
CANONICALIZER_SOURCE_OVERRIDE="${E49V_CANONICALIZER_SOURCE_OVERRIDE:-}"
OPS_ROOT_OVERRIDE="${E49V_OPS_ROOT_OVERRIDE:-}"
if [[ -n "$CANONICALIZER_SOURCE_OVERRIDE" ]]; then
  CANONICALIZER_SOURCE="$CANONICALIZER_SOURCE_OVERRIDE"
elif [[ -n "$SOURCE_ROOT_OVERRIDE" ]]; then
  CANONICALIZER_SOURCE="$SOURCE_ROOT_OVERRIDE/oat_drgrpo/math_strategy_canonicalizer.py"
else
  CANONICALIZER_SOURCE="$ROOT_DIR/src/oat_drgrpo/math_strategy_canonicalizer.py"
fi
PREFIX="${E49V_PREFIX:-e49v_exact_oat_natural_menu_math_05b_3ep_v1}"
MANIFEST="$ROOT_DIR/var/artifacts/${PREFIX}_comparative_jobs.tsv"
IDENTITY="$ROOT_DIR/var/artifacts/${PREFIX}_identity.json"

for required in \
  "$PYTHON_BIN" "$PROTOCOL" "$DATA_MANIFEST" "$MATERIALIZER" \
  "$TOY_ADVANCEMENT" "$CALIBRATION" "$COHORT_IDENTITY" \
  "$DECLARATION_CALIBRATION" "$DECLARATION_IDENTITY" \
  "$ENDPOINT_RECORD" "$MODEL_ROOT/model.safetensors" \
  "$CANONICALIZER_SOURCE" \
  "$DATA_ROOT/train/dataset_dict.json" \
  "$DATA_ROOT/eval/dataset_dict.json"; do
  if [[ ! -e "$required" ]]; then
    echo "E49V frozen prerequisite is missing: $required" >&2
    exit 1
  fi
done
if [[ -n "$CONTINUITY_CERTIFICATE" && ! -f "$CONTINUITY_CERTIFICATE" ]]; then
  echo "E49V continuity certificate is missing: $CONTINUITY_CERTIFICATE" >&2
  exit 1
fi

"$PYTHON_BIN" "$MATERIALIZER" \
  --output-root "$DATA_ROOT" \
  --evidence-root "$DATA_EVIDENCE" \
  --endpoint-record "$ENDPOINT_RECORD" \
  --audit-only >/dev/null

"$PYTHON_BIN" - "$TOY_ADVANCEMENT" "$TOY_ADVANCEMENT_SCHEMA" \
  "$CALIBRATION" "$COHORT_IDENTITY" \
  "$DECLARATION_CALIBRATION" "$DECLARATION_IDENTITY" "$ENDPOINT_RECORD" \
  "$CANONICALIZER_SOURCE" <<'PY'
import hashlib
import json
import pathlib
import sys
(
    toy_path,
    toy_schema,
    calibration_path,
    cohort_path,
    declaration_path,
    declaration_identity_path,
    endpoint_path,
    source_path,
) = [
    pathlib.Path(sys.argv[1]),
    sys.argv[2],
    *map(pathlib.Path, sys.argv[3:]),
]
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
toy = json.loads(toy_path.read_text(encoding="utf-8"))
if (
    toy.get("schema") != toy_schema
    or toy.get("complete_evidence") is not True
    or toy.get("advance_to_exact_oat_full") is not True
    or not all(toy.get("checks", {}).values())
):
    raise SystemExit("E49V toy advancement gate has not passed")
calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
if (
    calibration.get("schema")
    != "e49t_route_confusion_calibration_result_v1"
    or calibration.get("pass") is not True
    or not all(calibration.get("checks", {}).values())
):
    raise SystemExit("E49V route-confusion calibration has not passed")
identities = calibration.get("identities", {})
if (
    identities.get("cohort_identity_sha256") != sha(cohort_path)
    or identities.get("endpoint_record_sha256") != sha(endpoint_path)
    or identities.get("canonicalizer_sha256") != sha(source_path)
):
    raise SystemExit("E49V route-confusion identity mismatch")
declaration = json.loads(declaration_path.read_text(encoding="utf-8"))
if (
    declaration.get("schema") != "e49t_declaration_mismatch_result_v1"
    or declaration.get("pass") is not True
    or not all(declaration.get("checks", {}).values())
):
    raise SystemExit("E49V declaration-mismatch calibration has not passed")
declaration_ids = declaration.get("identities", {})
if (
    declaration_ids.get("cohort_identity_sha256")
    != sha(declaration_identity_path)
    or declaration_ids.get("endpoint_record_sha256") != sha(endpoint_path)
    or declaration_ids.get("canonicalizer_sha256") != sha(source_path)
):
    raise SystemExit("E49V declaration-mismatch identity mismatch")
PY

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

read -r EXPECTED_TRAIN_HASH EXPECTED_EVAL_HASH <<<"$(
  "$PYTHON_BIN" - "$DATA_MANIFEST" "$DATA_SCHEMA" \
    "$EXPECTED_SINGLETON_TOTAL" "$EXPECTED_MULTI_TOTAL" \
    "$EXPECTED_TRAIN_MULTI" "$EXPECTED_EVAL_MULTI" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {
    "schema": sys.argv[2],
    "train_rows": 384,
    "eval_rows": 500,
    "menu_count": 884,
    "overlay_menu_count": 100,
    "generated_singleton_menu_count": 784,
    "singleton_menu_count": int(sys.argv[3]),
    "multi_strategy_menu_count": int(sys.argv[4]),
    "multi_support_counts": {
        "train": int(sys.argv[5]),
        "eval": int(sys.argv[6]),
    },
    "zero_support_rows": 0,
    "all_generated_singletons_double_audited": True,
    "source_exact_order_preserved": True,
}
if any(manifest.get(key) != value for key, value in expected.items()):
    raise SystemExit("invalid exact OAT full-data manifest")
print(manifest["train_tree_sha256"], manifest["eval_tree_sha256"])
PY
)"
TRAIN_DATA_HASH="$(hash_tree "$DATA_ROOT/train")"
EVAL_DATA_HASH="$(hash_tree "$DATA_ROOT/eval")"
if [[ "$TRAIN_DATA_HASH" != "$EXPECTED_TRAIN_HASH" ]] \
  || [[ "$EVAL_DATA_HASH" != "$EXPECTED_EVAL_HASH" ]]; then
  echo "E49V full data identity mismatch" >&2
  exit 1
fi

ENDPOINT="$("$PYTHON_BIN" - "$ENDPOINT_RECORD" \
  "$EXPECTED_ENDPOINT_NODE" "$EXPECTED_ENDPOINT_PORT" <<'PY'
import json
import sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {
    "model": "qwen2.5-72b",
    "node": sys.argv[2],
    "port": int(sys.argv[3]),
    "tensor_parallel_size": 4,
    "max_model_len": 32768,
    "max_num_seqs": 8,
    "enforce_eager": True,
    "structured_output_backend": "guidance",
    "checkpoint_revision": "698703eae6604af048a3d2f509995dc302088217",
}
if any(record.get(key) != value for key, value in expected.items()):
    raise SystemExit("unexpected E49V Qwen72 endpoint")
print(f"http://{record['node']}:{int(record['port'])}/v1")
PY
)"

if [[ -n "$SOURCE_ROOT_OVERRIDE" ]]; then
  SOURCE_ROOT="$SOURCE_ROOT_OVERRIDE"
  SOURCE_HASH="$(hash_tree "$SOURCE_ROOT")"
else
  SOURCE_HASH="$(hash_tree "$ROOT_DIR/src")"
  SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e49v_exact_oat_${SOURCE_HASH}/src"
  if [[ ! -f "$SOURCE_ROOT/oat_drgrpo/__init__.py" ]]; then
    mkdir -p "$(dirname "$SOURCE_ROOT")"
    staging="$(mktemp -d "$(dirname "$SOURCE_ROOT")/.source.XXXXXX")"
    mkdir -p "$staging/src"
    cp -a "$ROOT_DIR/src/." "$staging/src/"
    mv "$staging/src" "$SOURCE_ROOT"
    rmdir "$staging"
  fi
fi
if [[ "$(hash_tree "$SOURCE_ROOT")" != "$SOURCE_HASH" ]]; then
  echo "E49V source snapshot mismatch" >&2
  exit 1
fi

if [[ -n "$OPS_ROOT_OVERRIDE" ]]; then
  OPS_ROOT="$OPS_ROOT_OVERRIDE"
  OPS_HASH="$(hash_tree "$OPS_ROOT")"
else
  OPS_INPUT="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.e49v-ops.XXXXXX")"
  mkdir -p "$OPS_INPUT/slurm"
  for file in repo_env.sh run_experiment.sh train.sh resolve_eval_cadence.py \
    submit_countdown_comparative.sh; do
    cp "$ROOT_DIR/ops/$file" "$OPS_INPUT/$file"
  done
  cp "$ROOT_DIR/ops/slurm/train_node302.slurm" "$OPS_INPUT/slurm/train_node302.slurm"
  OPS_HASH="$(hash_tree "$OPS_INPUT")"
  OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e49v_exact_oat_ops_${OPS_HASH}/ops"
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
fi

if [[ "$mode" == run && -e "$MANIFEST" ]]; then
  echo "Fresh E49V launch required; manifest already exists: $MANIFEST" >&2
  exit 1
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
export OAT_ZERO_INCLUDE_ONLINE_CANONICAL_POLICY_ENTROPY_ARM=0

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
export OAT_ZERO_ONLINE_CANONICAL_POLICY_ENTROPY_ADAPTATION=0
export OAT_ZERO_MATH_STRATEGY_ENDPOINT="$ENDPOINT"
export OAT_ZERO_MATH_STRATEGY_MODEL=qwen2.5-72b
export OAT_ZERO_MATH_STRATEGY_TIMEOUT_SECONDS=600
export OAT_ZERO_MATH_STRATEGY_WORKERS=4
export OAT_ZERO_MATH_STRATEGY_MAX_ITEM_CHARS=4000
export OAT_ZERO_MATH_STRATEGY_ALLOW_UNSTRUCTURED_INFERENCE=1
export OAT_ZERO_MATH_STRATEGY_GATE_TASK_REWARD=1

export OAT_ZERO_NUM_SAMPLES=16
export OAT_ZERO_LEARNING_RATE=0.0000002
export OAT_ZERO_MAX_TRAIN=384
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
export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=492900
export OAT_ZERO_EVAL_BATCH_SIZE=64
export OAT_ZERO_ALLOW_SPARSE_EVAL=1
export OAT_ZERO_EVAL_PROMPT_INTERVAL=384
export OAT_ZERO_EVAL_STEPS=384
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
export OAT_ZERO_SAVE_STEPS=384
export OAT_ZERO_SAVE_FROM=384
export OAT_ZERO_MAX_SAVE_NUM=3
export OAT_ZERO_RESUME_STEPS=384
export OAT_ZERO_RESUME_FROM=384
export OAT_ZERO_MAX_RESUME_NUM=3
export OAT_ZERO_EXPORT_STEPS=0
export OAT_ZERO_EXPORT_FROM=0
export OAT_ZERO_MAX_EXPORT_NUM=1
export OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=0
export OAT_ZERO_AUTO_RESUME=1
export OAT_ZERO_WATCHDOG_REQUEUE=1
export OAT_ZERO_WATCHDOG_MAX_RESTARTS=6

export OAT_ZERO_TRAIN_NODELIST=node302
export OAT_ZERO_TRAIN_GRES=gpu:a100:1
export OAT_ZERO_TRAIN_CPUS_PER_TASK=8
export OAT_ZERO_TRAIN_PARTITION=mltheory
export OAT_ZERO_TRAIN_ACCOUNT=mltheory
export OAT_ZERO_TRAIN_MEMORY=48G
export OAT_ZERO_TRAIN_TIME_LIMIT=24:00:00

PROTOCOL_HASH="$(sha256sum "$PROTOCOL" | cut -d' ' -f1)"
LAUNCHER_HASH="$(sha256sum "${BASH_SOURCE[0]}" | cut -d' ' -f1)"
CORE_HASH="$(sha256sum \
  "$SOURCE_ROOT/oat_drgrpo/math_strategy_menu.py" \
  "$SOURCE_ROOT/oat_drgrpo/math_strategy_canonicalizer.py" \
  "$SOURCE_ROOT/oat_drgrpo/online_canonical_bank.py" \
  "$SOURCE_ROOT/oat_drgrpo/online_canonical_controller.py" \
  "$SOURCE_ROOT/oat_drgrpo/learner/grpo.py" \
  "$SOURCE_ROOT/oat_drgrpo/args.py" | sha256sum | cut -d' ' -f1)"

"$PYTHON_BIN" - "$IDENTITY" "$SOURCE_HASH" "$OPS_HASH" "$ENDPOINT" \
  "$TRAIN_DATA_HASH" "$EVAL_DATA_HASH" "$PROTOCOL_HASH" "$CORE_HASH" \
  "$DATA_MANIFEST" "$TOY_ADVANCEMENT" "$CALIBRATION" \
  "$COHORT_IDENTITY" "$DECLARATION_CALIBRATION" \
  "$DECLARATION_IDENTITY" "$ENDPOINT_RECORD" "$LAUNCHER_HASH" \
  "$PREFIX" "$EXPECTED_TRAIN_MULTI" "$EXPECTED_EVAL_MULTI" \
  "${E49V_WRAPPER_HASH:-}" "$CONTINUITY_CERTIFICATE" <<'PY'
import hashlib
import json
import os
import pathlib
import sys
import tempfile
path = pathlib.Path(sys.argv[1])
def sha(name):
    return hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()
payload = {
    "schema": sys.argv[17],
    "source_hash": sys.argv[2],
    "ops_hash": sys.argv[3],
    "judge_endpoint": sys.argv[4],
    "train_data_sha256": sys.argv[5],
    "eval_data_sha256": sys.argv[6],
    "protocol_sha256": sys.argv[7],
    "core_contract_sha256": sys.argv[8],
    "data_manifest_sha256": sha(sys.argv[9]),
    "toy_advancement_sha256": sha(sys.argv[10]),
    "route_calibration_result_sha256": sha(sys.argv[11]),
    "route_calibration_identity_sha256": sha(sys.argv[12]),
    "declaration_calibration_result_sha256": sha(sys.argv[13]),
    "declaration_calibration_identity_sha256": sha(sys.argv[14]),
    "judge_endpoint_record_sha256": sha(sys.argv[15]),
    "launcher_sha256": sys.argv[16],
    "wrapper_sha256": sys.argv[20] or None,
    "continuity_certificate_sha256": (
        sha(sys.argv[21]) if sys.argv[21] else None
    ),
    "arms": ["grpo", "online_canonical_haarnoja"],
    "seed": 45,
    "train_rows": 384,
    "eval_rows": 500,
    "optimizer_updates": 1152,
    "prompt_epochs": 3,
    "num_samples": 16,
    "support_counts": {
        "train_multi": int(sys.argv[18]),
        "eval_multi": int(sys.argv[19]),
    },
    "task_reward_gate": (
        "answer_validator_plus_unanimous_finite_menu_route_inference"
    ),
    "method": {
        "canonical_support": "prompt_local_audited_action_combo",
        "unstructured_inference": True,
        "controller": "E46 normalized canonical-bank Haarnoja",
        "target_ratio": 0.80,
        "alpha": [0.10, 0.50],
        "alpha_lr": 0.003,
        "ema_decay": 0.90,
        "novelty_beta": 0.50,
        "policy_entropy_adaptation": False,
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

if [[ "$mode" == config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  "$OPS_ROOT/submit_countdown_comparative.sh"
  echo "[$PREFIX] configuration passed; no jobs submitted"
  exit 0
fi

unset OAT_ZERO_COMPARATIVE_CONFIG_ONLY
"$OPS_ROOT/submit_countdown_comparative.sh"
rows="$(awk -F '\t' 'NR > 1 {count++} END {print count+0}' "$MANIFEST")"
if [[ "$rows" != 2 ]]; then
  echo "$PREFIX expected two matched jobs, observed $rows" >&2
  exit 1
fi
echo "[$PREFIX] submitted matched exact-OAT three-epoch cohort"
