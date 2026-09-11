#!/usr/bin/env bash
# Launch E49B's matched hard-toy or exact OAT MATH cohort.
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
PROTOCOL="$ROOT_DIR/paper/preregistration/e49b_reasoned_math_strategy_haarnoja_05b.md"
CALIBRATION="$ROOT_DIR/var/artifacts/e47w_pairwise_math_strategy_calibration_v1/analysis.json"
ENDPOINT_RECORD="${E49B_QWEN72_ENDPOINT_RECORD:-$ROOT_DIR/var/artifacts/e49_math_strategy_qwen72_v1/qwen72_endpoint.json}"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
TOY_ROOT="$ROOT_DIR/var/data/e49b_math_strategy_toy"
FULL_ROOT="$ROOT_DIR/var/data/math12k_384_math500"
CANONICALIZER="$ROOT_DIR/src/oat_drgrpo/math_strategy_canonicalizer.py"
CALIBRATED_CANONICALIZER_SHA256="91a1a8fdc3b29fa49b1089154f5955ec2d9e759cc85e94b35bc7d101d81bf988"

if [[ ! -x "$PYTHON_BIN" ]] \
  || [[ ! -f "$PROTOCOL" ]] \
  || [[ ! -f "$MODEL_ROOT/model.safetensors" ]] \
  || ! grep -q '^\*\*Status: FROZEN BEFORE LAUNCH' "$PROTOCOL"; then
  echo "E49B frozen prerequisite is missing" >&2
  exit 1
fi
CANONICALIZER_SHA256="$(sha256sum "$CANONICALIZER" | cut -d' ' -f1)"
if [[ "$CANONICALIZER_SHA256" != "$CALIBRATED_CANONICALIZER_SHA256" ]]; then
  echo "E49B canonicalizer differs from the executable calibrated by E47W" >&2
  exit 1
fi

if [[ "$stage" == toy || "$stage" == full ]]; then
  CALIBRATION_STATUS="$(
    if [[ -f "$CALIBRATION" ]]; then
      "$PYTHON_BIN" - "$CALIBRATION" <<'PY'
import json
import sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
expected_checks = {
    "valid_anchor_admitted",
    "invalid_derivations_rejected",
    "routine_equivalence_merged",
    "contextual_clock_equivalence_merged",
    "contextual_median_false_new_prevented",
    "different_proofs_separated",
}
semantic = record.get("semantic_regressions") or {}
checks = semantic.get("checks") or {}
valid = (
    record.get("gate_status") == "pass"
    and record.get("protocol_id") == "E47W-CAL"
    and record.get("schema")
    == "e47w_pairwise_veto_calibration_report_v1"
    and semantic.get("schema") == "e47s_reduced_veto_regressions_v1"
    and semantic.get("pass") is True
    and expected_checks.issubset(checks)
    and all(checks[name] is True for name in expected_checks)
    and (record.get("manual_audit") or {}).get("status") == "complete"
)
print("pass" if valid else "fail")
PY
    else
      printf missing
    fi
  )"
  if [[ "$CALIBRATION_STATUS" != pass ]]; then
    echo "E47W-CAL has not cleared its frozen proof/equivalence gate" >&2
    exit 1
  fi
  if [[ ! -f "$ENDPOINT_RECORD" ]]; then
    echo "The frozen Qwen72 judge endpoint is not live" >&2
    exit 1
  fi
fi

if [[ "$stage" == full ]]; then
  TOY_DECISION="$ROOT_DIR/var/artifacts/e49b_math_strategy_toy_advancement.json"
  if [[ ! -f "$TOY_DECISION" ]] \
    || [[ "$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("advance_to_full",False))' "$TOY_DECISION")" != True ]]; then
    echo "E49B hard-toy advancement gate has not passed" >&2
    exit 1
  fi
fi

if [[ "$stage" == toy ]]; then
  "$PYTHON_BIN" \
    "$ROOT_DIR/ops/math_strategy_calibration/materialize_e49_math_strategy_data.py" \
    --output "$TOY_ROOT" --audit-only >/dev/null
  DATA_ROOT="$TOY_ROOT"
  MAX_TRAIN=50
  EVAL_INTERVAL=50
  SAVE_INTERVAL=50
  PREFIX=e49b_math_strategy_toy_05b_v1
else
  DATA_ROOT="$FULL_ROOT"
  MAX_TRAIN=384
  EVAL_INTERVAL=384
  SAVE_INTERVAL=384
  PREFIX=e49b_math_strategy_full_05b_v1
fi

if [[ ! -f "$DATA_ROOT/train/dataset_dict.json" ]] \
  || [[ ! -f "$DATA_ROOT/eval/dataset_dict.json" ]]; then
  echo "E49B data artifact is incomplete: $DATA_ROOT" >&2
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

if [[ "$stage" == toy ]]; then
  TRAIN_DATA_HASH="$(hash_tree "$DATA_ROOT/train")"
  EVAL_DATA_HASH="$(hash_tree "$DATA_ROOT/eval")"
  if [[ "$TRAIN_DATA_HASH" != "7583599fcfd71494001974035d30459fe11445bd56478a9089b7d8612b3929ec" ]] \
    || [[ "$EVAL_DATA_HASH" != "9cdd064a024df1036c2024d7f9f6f375800be888b1399a950897026210067ff3" ]]; then
    echo "E49B hard-toy data identity mismatch" >&2
    exit 1
  fi
else
  TRAIN_DATA_HASH="$(sha256sum "$DATA_ROOT/train/train/data-00000-of-00001.arrow" | cut -d' ' -f1)"
  EVAL_DATA_HASH="$(sha256sum "$DATA_ROOT/eval/math/data-00000-of-00001.arrow" | cut -d' ' -f1)"
  if [[ "$TRAIN_DATA_HASH" != "359defbf82b6e05a1fdddb3479ed689f8a607dc727814e73ebfe69b2ffdff8b8" ]] \
    || [[ "$EVAL_DATA_HASH" != "2104f8f8eef09ce0bfc929e255f0f04c59311f1f3395bd293f1d03051c482cf7" ]]; then
    echo "E49B exact OAT MATH data identity mismatch" >&2
    exit 1
  fi
fi

SOURCE_HASH="$(hash_tree "$ROOT_DIR/src")"
SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e49b_math_strategy_${SOURCE_HASH}/src"
if [[ ! -f "$SOURCE_ROOT/oat_drgrpo/__init__.py" ]]; then
  mkdir -p "$(dirname "$SOURCE_ROOT")"
  staging="$(mktemp -d "$(dirname "$SOURCE_ROOT")/.source.XXXXXX")"
  mkdir -p "$staging/src"
  cp -a "$ROOT_DIR/src/." "$staging/src/"
  mv "$staging/src" "$SOURCE_ROOT"
  rmdir "$staging"
fi
if [[ "$(hash_tree "$SOURCE_ROOT")" != "$SOURCE_HASH" ]]; then
  echo "E49B source snapshot hash mismatch" >&2
  exit 1
fi

OPS_INPUT="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.e49b-ops.XXXXXX")"
mkdir -p "$OPS_INPUT/slurm"
for file in repo_env.sh run_experiment.sh train.sh resolve_eval_cadence.py \
  submit_countdown_comparative.sh; do
  cp "$ROOT_DIR/ops/$file" "$OPS_INPUT/$file"
done
cp "$ROOT_DIR/ops/slurm/train_node302.slurm" "$OPS_INPUT/slurm/train_node302.slurm"
OPS_HASH="$(hash_tree "$OPS_INPUT")"
OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e49b_math_strategy_ops_${OPS_HASH}/ops"
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
  echo "Fresh E49B stage required; manifest already exists: $MANIFEST" >&2
  exit 1
fi

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
    raise SystemExit("unexpected E47W judge endpoint")
print(f"http://{record['node']}:{int(record['port'])}/v1")
PY
)"

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
export OAT_ZERO_PROMPT_MAX_LENGTH=1024
export OAT_ZERO_GENERATE_MAX_LENGTH=1024
export OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=1024
export OAT_ZERO_MAX_MODEL_LEN=2048
export OAT_ZERO_TEMPERATURE=1
export OAT_ZERO_TOP_P=1
export OAT_ZERO_EVAL_TEMPERATURE=0
export OAT_ZERO_EVAL_MODE_COVERAGE_K=8
export OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE=1
export OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=1
export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=490100
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

CALIBRATION_HASH="$(
  if [[ -f "$CALIBRATION" ]]; then sha256sum "$CALIBRATION" | cut -d' ' -f1; else printf pending; fi
)"
PROTOCOL_HASH="$(sha256sum "$PROTOCOL" | cut -d' ' -f1)"
ENDPOINT_RECORD_HASH="$(
  if [[ -f "$ENDPOINT_RECORD" ]]; then sha256sum "$ENDPOINT_RECORD" | cut -d' ' -f1; else printf pending; fi
)"

"$PYTHON_BIN" - "$IDENTITY" "$stage" "$SOURCE_HASH" "$OPS_HASH" "$ENDPOINT" \
  "$CALIBRATION_HASH" "$PROTOCOL_HASH" "$ENDPOINT_RECORD_HASH" \
  "$TRAIN_DATA_HASH" "$EVAL_DATA_HASH" "$CANONICALIZER_SHA256" <<'PY'
import json
import os
import pathlib
import sys
import tempfile

path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "e49b_reasoned_math_strategy_haarnoja_05b_v1",
    "stage": sys.argv[2],
    "source_hash": sys.argv[3],
    "ops_hash": sys.argv[4],
    "judge_endpoint": sys.argv[5],
    "calibration_analysis_sha256": sys.argv[6],
    "protocol_sha256": sys.argv[7],
    "judge_endpoint_record_sha256": sys.argv[8],
    "train_data_sha256": sys.argv[9],
    "eval_data_sha256": sys.argv[10],
    "calibrated_canonicalizer_sha256": sys.argv[11],
    "arms": ["grpo", "online_canonical_haarnoja"],
    "seed": 45,
    "prompt_epochs": 3,
    "num_samples": 16,
    "data_root": os.environ["OAT_ZERO_COMPARATIVE_DATA_ROOT"],
    "method": {
        "key_mode": "math_strategy_qwen72",
        "validator": "math_verify",
        "integrity_judge_passes": 2,
        "boundary_partition_passes": 2,
        "pairwise_veto_passes": 2,
        "pairwise_veto_max_pairs": 128,
        "permutation_seeds": [470721, 470722],
        "canonicalizer_state_schema": (
            "math_strategy_canonicalizer_pair_veto_v15"
        ),
        "target_ratio": 0.80,
        "alpha": [0.10, 0.50],
        "alpha_lr": 0.003,
        "ema_decay": 0.90,
        "novelty_beta": 0.50,
    },
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
  echo "[e49b] configuration passed; no jobs submitted"
  exit 0
fi

unset OAT_ZERO_COMPARATIVE_CONFIG_ONLY
"$OPS_ROOT/submit_countdown_comparative.sh"

rows="$(awk -F '\t' 'NR > 1 {count++} END {print count+0}' "$MANIFEST")"
if [[ "$rows" != 2 ]]; then
  echo "E49B expected two matched jobs, observed $rows" >&2
  exit 1
fi
echo "[e49b] submitted matched ${stage} cohort"
