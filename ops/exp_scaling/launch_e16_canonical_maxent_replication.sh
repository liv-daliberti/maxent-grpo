#!/usr/bin/env bash
# Configure or launch only E16's E15-derived 0.5B canonical MaxEnt grid.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"
export VLLM_USE_V1=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OAT_ZERO_AUTO_RESUME=0
export OAT_ZERO_WATCHDOG_REQUEUE=0

PYTHON_BIN="${OAT_ZERO_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
PHASE="${1:-}"
case "$PHASE" in
  smoke-config)
    STAGE=smoke
    CONFIG_ONLY=1
    ;;
  smoke)
    STAGE=smoke
    CONFIG_ONLY=0
    ;;
  full-config)
    STAGE=full
    CONFIG_ONLY=1
    ;;
  full)
    STAGE=full
    CONFIG_ONLY=0
    ;;
  *)
    echo "Usage: $0 {smoke-config|smoke|full-config|full}" >&2
    exit 1
    ;;
esac

PROTOCOL="$ROOT_DIR/paper/preregistration/e16_canonical_maxent_replication.md"
E15_OUTCOME="${OAT_ZERO_E16_E15_OUTCOME:-$ROOT_DIR/var/artifacts/e15_canonical_dose_decision_20260718_2229.json}"
GRAPH_DATA_ROOT="${OAT_ZERO_E16_GRAPH_DATA_ROOT:-$ROOT_DIR/var/data/exact_answer_mode_probe}"
COUNTDOWN_DATA_ROOT="${OAT_ZERO_E16_COUNTDOWN_DATA_ROOT:-$ROOT_DIR/var/data/exact_countdown_easy3_probe}"

for required in "$PROTOCOL" "$E15_OUTCOME"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing E16 prerequisite: $required" >&2
    exit 1
  fi
done
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing E16 Python: $PYTHON_BIN" >&2
  exit 1
fi

plan_payload="$($PYTHON_BIN "$ROOT_DIR/ops/exp_scaling/e16_canonical_plan.py" --stage "$STAGE")"
config_digest="$(jq -er '.config_digest' <<<"$plan_payload")"
smoke_plan_payload="$($PYTHON_BIN "$ROOT_DIR/ops/exp_scaling/e16_canonical_plan.py" --stage smoke)"
smoke_config_digest="$(jq -er '.config_digest' <<<"$smoke_plan_payload")"
full_plan_payload="$($PYTHON_BIN "$ROOT_DIR/ops/exp_scaling/e16_canonical_plan.py" --stage full)"
full_config_digest="$(jq -er '.config_digest' <<<"$full_plan_payload")"
antecedent_identity="$($PYTHON_BIN "$ROOT_DIR/ops/exp_scaling/verify_e16_antecedent.py" --e15-outcome "$E15_OUTCOME")"
dataset_identity="$($PYTHON_BIN "$ROOT_DIR/ops/exp_scaling/verify_e16_canonical_datasets.py" \
  --graph-data-root "$GRAPH_DATA_ROOT" \
  --countdown-data-root "$COUNTDOWN_DATA_ROOT")"
runtime_identity="$($PYTHON_BIN "$ROOT_DIR/ops/exp_scaling/verify_e16_canonical_runtime.py" | tail -n 1)"
source_hash="$(PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" -c \
  'import sys; from pathlib import Path; from ops.exp_scaling.check_e14_preflight import source_tree_hash; print(source_tree_hash(Path(sys.argv[1]), logical_repo_root=Path(sys.argv[2])))' \
  "$ROOT_DIR/src" "$ROOT_DIR")"
protocol_sha256="$(sha256sum "$PROTOCOL" | cut -d' ' -f1)"
execution_identity="$($PYTHON_BIN "$ROOT_DIR/ops/exp_scaling/verify_e16_execution_surface.py" --repo-root "$ROOT_DIR")"
execution_surface_hash="$(jq -er '.sha256' <<<"$execution_identity")"
expected_execution_surface_hash="$(sed -n 's/^| Shell\/tooling execution surface | `\([^`]*\)` |$/\1/p' "$PROTOCOL")"
e15_outcome_sha256="$(jq -er '.e15_outcome_sha256' <<<"$antecedent_identity")"
dataset_identity_sha256="$($PYTHON_BIN -c \
  'import hashlib,json,sys; print(hashlib.sha256(json.dumps(json.loads(sys.argv[1]),allow_nan=False,separators=(",",":"),sort_keys=True).encode()).hexdigest())' \
  "$dataset_identity")"
runtime_identity_sha256="$($PYTHON_BIN -c \
  'import hashlib,json,sys; print(hashlib.sha256(json.dumps(json.loads(sys.argv[1]),allow_nan=False,separators=(",",":"),sort_keys=True).encode()).hexdigest())' \
  "$runtime_identity")"
expected_source_hash="$(jq -er '.frozen_identity.source_hash' <<<"$plan_payload")"
expected_dataset_identity_sha256="$(jq -er '.frozen_identity.dataset_identity_sha256' <<<"$plan_payload")"
expected_runtime_identity_sha256="$(jq -er '.frozen_identity.runtime_identity_sha256' <<<"$plan_payload")"
expected_config_digest="$(jq -er '.frozen_identity.config_digest' <<<"$plan_payload")"
if [[ "$dataset_identity_sha256" != "$expected_dataset_identity_sha256" ]]; then
  echo "E16 dataset identity differs from its prospective constant." >&2
  exit 1
fi
if [[ "$runtime_identity_sha256" != "$expected_runtime_identity_sha256" ]]; then
  echo "E16 runtime identity differs from its prospective constant." >&2
  exit 1
fi
if [[ "$config_digest" != "$expected_config_digest" ]]; then
  echo "E16 config differs from its prospective stage digest." >&2
  exit 1
fi

echo "[e16] phase=$PHASE stage=$STAGE config_only=$CONFIG_ONLY"
echo "[e16] formulation=e15_derived_direct_on_policy_canonical_maxent"
echo "[e16] arms=maxent,maxent_control,maxent_dual"
echo "[e16] source_hash=$source_hash"
echo "[e16] expected_source_hash=$expected_source_hash"
echo "[e16] protocol_sha256=$protocol_sha256"
echo "[e16] execution_surface_hash=$execution_surface_hash"
echo "[e16] expected_execution_surface_hash=$expected_execution_surface_hash"
echo "[e16] e15_outcome_sha256=$e15_outcome_sha256"
echo "[e16] dataset_identity_sha256=$dataset_identity_sha256"
echo "[e16] runtime_identity_sha256=$runtime_identity_sha256"
echo "[e16] config_digest=$config_digest smoke_config_digest=$smoke_config_digest"
echo "[e16] graph_prefix=$(jq -er '.plan.tasks.graph_coloring.prefix' <<<"$plan_payload")"
echo "[e16] countdown_prefix=$(jq -er '.plan.tasks.countdown.prefix' <<<"$plan_payload")"
echo "[e16] recovery=disabled auto_resume=0 watchdog_requeue=0"

approval_summary='null'
if [[ "$CONFIG_ONLY" == "0" ]]; then
  if ! grep -q '^\*\*Status: FROZEN' "$PROTOCOL"; then
    echo "E16 protocol is not FROZEN; refusing to create artifacts or submit jobs." >&2
    exit 1
  fi
  if [[ ! "$expected_source_hash" =~ ^[0-9a-f]{64}$ ]]; then
    echo "E16 expected source hash was not recorded at protocol freeze." >&2
    exit 1
  fi
  if [[ "$source_hash" != "$expected_source_hash" ]]; then
    echo "E16 Python source differs from the prospectively frozen hash." >&2
    exit 1
  fi
  if [[ ! "$expected_execution_surface_hash" =~ ^[0-9a-f]{64}$ ]]; then
    echo "E16 execution-surface hash was not recorded at protocol freeze." >&2
    exit 1
  fi
  if [[ "$execution_surface_hash" != "$expected_execution_surface_hash" ]]; then
    echo "E16 shell/tooling execution surface differs from the frozen hash." >&2
    exit 1
  fi
fi
if [[ "$CONFIG_ONLY" == "0" && "$STAGE" == "full" ]]; then
  approval_path="${OAT_ZERO_E16_SMOKE_APPROVAL:?full requires OAT_ZERO_E16_SMOKE_APPROVAL}"
  approval_sha256="${OAT_ZERO_E16_SMOKE_APPROVAL_SHA256:?full requires OAT_ZERO_E16_SMOKE_APPROVAL_SHA256}"
  approval_verifier="$($PYTHON_BIN -c '
import json, pathlib, sys
approval = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
paths = {
    cell["campaign_identity"]["approval_verifier"]["path"]
    for cell in approval.get("cells", [])
}
if len(paths) != 1:
    raise SystemExit("E16 approval does not bind one frozen verifier")
print(paths.pop())
' "$approval_path")"
  if [[ ! -f "$approval_verifier" ]]; then
    echo "E16 approval-bound frozen verifier is missing: $approval_verifier" >&2
    exit 1
  fi
  approval_summary="$($PYTHON_BIN "$approval_verifier" \
    --approval "$approval_path" \
    --approval-sha256 "$approval_sha256" \
    --source-hash "$source_hash" \
    --protocol-sha256 "$protocol_sha256" \
    --e15-outcome-sha256 "$e15_outcome_sha256" \
    --dataset-identity-sha256 "$dataset_identity_sha256" \
    --runtime-identity-sha256 "$runtime_identity_sha256" \
    --smoke-config-digest "$smoke_config_digest" \
    --full-config-digest "$full_config_digest" \
    --execution-surface-hash "$execution_surface_hash")"
  echo "[e16] full_authorization=$approval_summary"
fi

write_identity() {
  local task="$1" prefix="$2" identity_path="$3" snapshot_root="$4"
  "$PYTHON_BIN" -c '
import json, os, pathlib, sys, tempfile
path = pathlib.Path(sys.argv[1])
config = json.loads(sys.argv[13])
plan = config["plan"]
payload = {
    "schema": "e16_canonical_campaign_identity_v1",
    "protocol": "E16",
    "stage": sys.argv[2],
    "task": sys.argv[3],
    "prefix": sys.argv[4],
    "source_hash": sys.argv[5],
    "protocol_path": str(pathlib.Path(sys.argv[6]).resolve()),
    "protocol_sha256": sys.argv[7],
    "e15_antecedent": json.loads(sys.argv[8]),
    "dataset_identity": json.loads(sys.argv[9]),
    "dataset_identity_sha256": sys.argv[10],
    "runtime_identity": json.loads(sys.argv[11]),
    "runtime_identity_sha256": sys.argv[12],
    "config": config,
    "config_digest": sys.argv[14],
    "smoke_config_digest": sys.argv[21],
    "full_config_digest": sys.argv[22],
    "task_config": {
        "arms": plan["arms"],
        "common": plan["common"],
        "seeds": plan["seeds"],
        "stage": plan["stage"],
        "task": plan["tasks"][sys.argv[3]],
    },
    "smoke_approval": json.loads(sys.argv[15]),
    "source_snapshot": str(pathlib.Path(sys.argv[16]).resolve()),
    "source_snapshot_hash": sys.argv[17],
    "execution_identity": json.loads(sys.argv[18]),
    "execution_surface_hash": sys.argv[19],
    "execution_snapshot_root": str(pathlib.Path(sys.argv[20]).resolve()),
    "auto_resume": False,
    "watchdog_requeue": False,
}
path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)
except BaseException:
    pathlib.Path(temporary).unlink(missing_ok=True)
    raise
' "$identity_path" "$STAGE" "$task" "$prefix" "$source_hash" \
    "$PROTOCOL" "$protocol_sha256" "$antecedent_identity" "$dataset_identity" \
    "$dataset_identity_sha256" "$runtime_identity" "$runtime_identity_sha256" \
    "$plan_payload" "$config_digest" "$approval_summary" "$snapshot_root" \
    "$source_hash" "$execution_identity" "$execution_surface_hash" \
    "$(dirname "$snapshot_root")" "$smoke_config_digest" \
    "$full_config_digest"
}

submit_task() {
  local task="$1"
  local prefix data_root target updates budget eval_interval prompt_template legacy_graph seeds epochs
  prefix="$(jq -er --arg task "$task" '.plan.tasks[$task].prefix' <<<"$plan_payload")"
  target="$(jq -er --arg task "$task" '.plan.tasks[$task].target_entropy' <<<"$plan_payload")"
  updates="$(jq -er --arg task "$task" '.plan.tasks[$task].target_optimizer_updates' <<<"$plan_payload")"
  budget="$(jq -er --arg task "$task" '.plan.tasks[$task].trajectory_query_budget' <<<"$plan_payload")"
  eval_interval="$(jq -er --arg task "$task" '.plan.tasks[$task].eval_prompt_interval' <<<"$plan_payload")"
  prompt_template="$(jq -er --arg task "$task" '.plan.tasks[$task].prompt_template' <<<"$plan_payload")"
  seeds="$(jq -er '.plan.seeds | map(tostring) | join(",")' <<<"$plan_payload")"
  epochs="$(jq -er '.plan.common.prompt_epochs' <<<"$plan_payload")"
  if [[ "$task" == "graph_coloring" ]]; then
    data_root="$GRAPH_DATA_ROOT"
    legacy_graph=1
  else
    data_root="$COUNTDOWN_DATA_ROOT"
    legacy_graph=0
  fi
  identity_path="$ROOT_DIR/var/artifacts/${prefix}_e16_identity.json"
  if [[ "${E16_CONFIG_PREFLIGHT:-0}" == "0" && ! -f "$identity_path" ]]; then
    echo "E16 identity was not frozen before submission: $identity_path" >&2
    exit 1
  fi

  export RUN_STAMP_PREFIX="$prefix"
  export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
  export OAT_ZERO_COMPARATIVE_TASK="$task"
  export OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-0.5b-instruct
  export OAT_ZERO_COMPARATIVE_DATA_ROOT="$data_root"
  export OAT_ZERO_TRAIN_SEEDS="$seeds"
  export OAT_ZERO_ONLY_ARMS=maxent,maxent_control,maxent_dual
  export OAT_ZERO_INCLUDE_TOKEN_ENTROPY_ARM=0
  export OAT_ZERO_INCLUDE_SEED_ARM=0
  export OAT_ZERO_INCLUDE_XDR_ADAPT_ARM=0
  export OAT_ZERO_INCLUDE_XDR_TAU_CONTROL_ARM=0
  export OAT_ZERO_INCLUDE_XDR_SAC_DUAL_ARM=0
  export OAT_ZERO_INCLUDE_MAXENT_ARM=1
  export OAT_ZERO_INCLUDE_MAXENT_CONTROL_ARM=1
  export OAT_ZERO_INCLUDE_MAXENT_DUAL_ARM=1
  export OAT_ZERO_INCLUDE_MAXENT_LENGTH_DUAL_ARM=0
  export OAT_ZERO_COMPARATIVE_REBUILD=0
  export OAT_ZERO_REQUIRE_EXISTING_DATA=1
  export OAT_ZERO_APPEND_MANIFEST=0
  export OAT_ZERO_XDR_TAUS=""
  export OAT_ZERO_NUM_SAMPLES=16
  export OAT_ZERO_MAX_TRAIN="$budget"
  export OAT_ZERO_MAX_QUERIES="$budget"
  export OAT_ZERO_NUM_PROMPT_EPOCH="$epochs"
  export OAT_ZERO_EVAL_PROMPT_INTERVAL="$eval_interval"
  export OAT_ZERO_SAVE_STEPS="$eval_interval"
  export OAT_ZERO_SAVE_FROM="$eval_interval"
  export OAT_ZERO_SAVE_CKPT=1
  export OAT_ZERO_MAX_SAVE_NUM=5
  export OAT_ZERO_MAX_SAVE_MEM=2000
  export OAT_ZERO_LEARNING_RATE=0.0000002
  export OAT_ZERO_MAXENT_ALPHA=0.10
  export OAT_ZERO_MAXENT_FIXED_ALPHA=0.10
  export OAT_ZERO_MAXENT_CONTROL_BASE_ALPHA=0.075
  export OAT_ZERO_MAXENT_DUAL_BASE_ALPHA=0.075
  export OAT_ZERO_MAXENT_CONTROL_RATIO=1.0
  export OAT_ZERO_MAXENT_CONTROL_TARGET_ENTROPY="$target"
  export OAT_ZERO_MAXENT_CONTROL_WARMUP_STEPS=1
  export OAT_ZERO_MAXENT_CONTROL_MAX_ALPHA=0.10
  export OAT_ZERO_MAXENT_CONTROL_EMA_DECAY=0.9
  export OAT_ZERO_MAXENT_CONTROL_GAIN=4.0
  export OAT_ZERO_MAXENT_DUAL_RATIO=1.0
  export OAT_ZERO_MAXENT_DUAL_TARGET_ENTROPY="$target"
  export OAT_ZERO_MAXENT_DUAL_WARMUP_STEPS=1
  export OAT_ZERO_MAXENT_DUAL_MIN_ALPHA=0.05
  export OAT_ZERO_MAXENT_DUAL_MAX_ALPHA=0.10
  export OAT_ZERO_MAXENT_DUAL_ALPHA_LR=0.005
  export OAT_ZERO_MAXENT_LENGTH_TARGET=0
  export OAT_ZERO_POLICY_ENTROPY_COEF=0
  export OAT_ZERO_NUM_PPO_EPOCHS=1
  export OAT_ZERO_MAX_NORM=1.0
  export OAT_ZERO_BETA=0
  export OAT_ZERO_IGNORE_NO_EOS=0
  export OAT_ZERO_TRAIN_BATCH_SIZE=16
  export OAT_ZERO_TRAIN_BATCH_SIZE_PER_DEVICE=4
  export OAT_ZERO_ROLLOUT_BATCH_SIZE=1
  export OAT_ZERO_ROLLOUT_BATCH_SIZE_PER_DEVICE=1
  export OAT_ZERO_N_GPU=1
  export OAT_ZERO_NUM_GPUS_PER_ACTOR=1
  export OAT_ZERO_PROMPT_TEMPLATE="$prompt_template"
  export OAT_ZERO_TEST_SPLIT=multi_answer
  export OAT_ZERO_CANONICAL_ACTION_TASK="$task"
  export OAT_ZERO_CANONICAL_GRAPH_ACTIONS="$legacy_graph"
  export OAT_ZERO_CANONICAL_GRAPH_ACTION_COUNT=3
  export OAT_ZERO_CANONICAL_GRAPH_LEARNER_SAMPLING=1
  export OAT_ZERO_CANONICAL_GRAPH_FIXED_SHAPE_SAMPLING=1
  export OAT_ZERO_GENERATE_MAX_LENGTH=192
  export OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=192
  export OAT_ZERO_TEMPERATURE=1
  export OAT_ZERO_TOP_P=1
  export OAT_ZERO_EVAL_MODE_COVERAGE_K=8
  export OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE=1
  export OAT_ZERO_EVAL_TEMPERATURE=0
  export OAT_ZERO_EVAL_BATCH_SIZE=64
  export OAT_ZERO_PROMPT_MAX_LENGTH=256
  export OAT_ZERO_ZERO_STAGE=2
  export OAT_ZERO_VLLM_GPU_RATIO=0.25
  export OAT_ZERO_ENABLE_FLASH_ATTN=0
  export OAT_ZERO_ADAM_OFFLOAD=0
  export OAT_ZERO_ACTIVATION_OFFLOADING=0
  export OAT_ZERO_COLLOCATE=1
  export OAT_ZERO_VLLM_SLEEP=1
  export OAT_ZERO_PYTHON="$PYTHON_BIN"
  export OAT_ZERO_PYTHON_LIB_DIR="$ROOT_DIR/var/seed_paper_eval/paper310/lib"
  export OAT_ZERO_PRETRAIN="$TRANSFORMERS_CACHE/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY="${E16_CONFIG_PREFLIGHT:-0}"
  export OAT_ZERO_SBATCH_HOLD=1
  export OAT_ZERO_TRAIN_TIME_LIMIT="${OAT_ZERO_E16_TRAIN_TIME_LIMIT:-08:00:00}"
  export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E16_TRAIN_NODELIST:-node302}"
  export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E16_TRAIN_GRES:-gpu:a100:1}"
  export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E16_TRAIN_PARTITION:-mltheory}"
  export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E16_TRAIN_ACCOUNT:-mltheory}"
  if [[ "${E16_CONFIG_PREFLIGHT:-0}" == "0" ]]; then
    export OAT_ZERO_PROTOCOL_IDENTITY="$identity_path"
  else
    unset OAT_ZERO_PROTOCOL_IDENTITY
  fi
  export OAT_ZERO_E16_TARGET_OPTIMIZER_UPDATES="$updates"

  unset OAT_ZERO_PROMPT_DATA
  unset OAT_ZERO_EVAL_DATA
  unset OAT_ZERO_RESUME_DIR
  unset OAT_ZERO_RESUME_TAG
  unset SAVE_PATH

  if [[ "${E16_CONFIG_PREFLIGHT:-0}" == "0" ]]; then
    bash "$OAT_ZERO_OPS_SNAPSHOT_ROOT/submit_countdown_comparative.sh"
  else
    bash "$ROOT_DIR/ops/submit_countdown_comparative.sh"
  fi
}

unset OAT_ZERO_CAMPAIGN_SOURCE_ROOT
unset OAT_ZERO_OPS_SNAPSHOT_ROOT
export E16_CONFIG_PREFLIGHT=1
submit_task graph_coloring
submit_task countdown

if [[ "$CONFIG_ONLY" == "1" ]]; then
  if [[ "$STAGE" == "full" && -z "${OAT_ZERO_E16_SMOKE_APPROVAL:-}" ]]; then
    echo "[e16] full_authorization=HELD (immutable all-six smoke approval not supplied)"
  fi
  if grep -q '^\*\*Status: DESIGN DRAFT' "$PROTOCOL"; then
    echo "[e16] protocol_status=DESIGN (configuration only; submission remains disabled)"
  else
    echo "[e16] protocol_status=$(sed -n 's/^\*\*Status: \([^*]*\).*/\1/p' "$PROTOCOL" | head -n 1)"
  fi
  echo "[e16] both shared-submitter task configurations passed"
  echo "[e16] configuration only; no identity, source snapshot, manifest, or job created"
  exit 0
fi

# Only after both task/arm matrices have passed the shared submitter's own
# configuration path do we create one source snapshot and any identities.
graph_prefix="$(jq -er '.plan.tasks.graph_coloring.prefix' <<<"$plan_payload")"
countdown_prefix="$(jq -er '.plan.tasks.countdown.prefix' <<<"$plan_payload")"
graph_identity="$ROOT_DIR/var/artifacts/${graph_prefix}_e16_identity.json"
countdown_identity="$ROOT_DIR/var/artifacts/${countdown_prefix}_e16_identity.json"
graph_manifest="$ROOT_DIR/var/artifacts/${graph_prefix}_comparative_jobs.tsv"
countdown_manifest="$ROOT_DIR/var/artifacts/${countdown_prefix}_comparative_jobs.tsv"
snapshot_parent="$ROOT_DIR/var/artifacts/source_snapshots/e16_canonical_${STAGE}_${source_hash}"
snapshot_root="$snapshot_parent/src"
for candidate in \
  "$snapshot_parent" \
  "$graph_identity" \
  "$countdown_identity" \
  "$graph_manifest" \
  "$countdown_manifest"; do
  if [[ -e "$candidate" ]]; then
    echo "E16 launch artifact already exists; refusing an ambiguous resubmission: $candidate" >&2
    exit 1
  fi
done

# The source-snapshot directory is the atomic ownership claim for this fixed
# E16 stage/prefix.  The EXIT rollback removes only paths that were absent
# above and created by this invocation.  A transient failure therefore cannot
# strand fixed-prefix artifacts and permanently prevent a clean retry.
snapshot_created=0
cohort_unreleased=1
rollback_unreleased_cohort() {
  local status=$?
  if [[ "${cohort_unreleased:-0}" == "1" ]]; then
    local -a created_job_ids=()
    local manifest job_id cleanup_safe=1
    for manifest in "$graph_manifest" "$countdown_manifest"; do
      if [[ -f "$manifest" ]]; then
        while IFS=$'\t' read -r _ _ job_id _; do
          if [[ "$job_id" =~ ^[1-9][0-9]*$ ]]; then
            created_job_ids+=("$job_id")
          fi
        done < <(tail -n +2 "$manifest")
      fi
    done
    if (( ${#created_job_ids[@]} > 0 )); then
      echo "[e16] rollback: cancelling unreleased held jobs ${created_job_ids[*]}" >&2
      if ! scancel "${created_job_ids[@]}"; then
        cleanup_safe=0
        echo "[e16] rollback: scancel failed; retaining frozen artifacts for safe manual recovery" >&2
      fi
    fi
    if [[ "$cleanup_safe" == "1" ]]; then
      rm -f -- \
        "$graph_manifest" "$countdown_manifest" \
        "$graph_identity" "$countdown_identity"
      if [[ "${snapshot_created:-0}" == "1" ]]; then
        rm -rf -- "$snapshot_parent"
      fi
    fi
  fi
  return "$status"
}
trap rollback_unreleased_cohort EXIT

# Plain mkdir (not -p) makes concurrent ownership fail closed.
mkdir "$snapshot_parent"
snapshot_created=1
snapshot_tmp="$(mktemp -d "$snapshot_parent/.source.XXXXXX")"
mkdir -p "$snapshot_tmp/src"
cp -a "$ROOT_DIR/src/." "$snapshot_tmp/src/"
while IFS= read -r relative; do
  mkdir -p "$snapshot_tmp/$(dirname "$relative")"
  cp "$ROOT_DIR/$relative" "$snapshot_tmp/$relative"
done < <(jq -r '.files[].path' <<<"$execution_identity")
mv "$snapshot_tmp/src" "$snapshot_root"
mv "$snapshot_tmp/ops" "$snapshot_parent/ops"
rmdir "$snapshot_tmp"
snapshot_hash="$(PYTHONPATH="$snapshot_parent" "$PYTHON_BIN" -c \
  'import sys; from pathlib import Path; from ops.exp_scaling.check_e14_preflight import source_tree_hash; print(source_tree_hash(Path(sys.argv[1]), logical_repo_root=Path(sys.argv[2])))' \
  "$snapshot_root" "$ROOT_DIR")"
if [[ "$snapshot_hash" != "$source_hash" ]]; then
  echo "E16 shared source snapshot hash mismatch" >&2
  exit 1
fi
snapshot_execution_identity="$($PYTHON_BIN "$snapshot_parent/ops/exp_scaling/verify_e16_execution_surface.py" --repo-root "$snapshot_parent")"
snapshot_execution_hash="$(jq -er '.sha256' <<<"$snapshot_execution_identity")"
if [[ "$snapshot_execution_hash" != "$execution_surface_hash" ]]; then
  echo "E16 shared execution-surface snapshot hash mismatch" >&2
  exit 1
fi
export OAT_ZERO_CAMPAIGN_SOURCE_ROOT="$snapshot_root"
export OAT_ZERO_OPS_SNAPSHOT_ROOT="$snapshot_parent/ops"

for task in graph_coloring countdown; do
  prefix="$(jq -er --arg task "$task" '.plan.tasks[$task].prefix' <<<"$plan_payload")"
  identity_path="$ROOT_DIR/var/artifacts/${prefix}_e16_identity.json"
  write_identity "$task" "$prefix" "$identity_path" "$snapshot_root"
done

export E16_CONFIG_PREFLIGHT=0
submit_task graph_coloring
submit_task countdown
cohort_identity="$($PYTHON_BIN "$snapshot_parent/ops/exp_scaling/verify_e16_held_cohort.py" \
  --stage "$STAGE" \
  --graph-manifest "$graph_manifest" \
  --countdown-manifest "$countdown_manifest")"
release_argument="$(jq -er '.release_argument' <<<"$cohort_identity")"
scontrol release "$release_argument"
cohort_unreleased=0
trap - EXIT
echo "[e16] released held cohort=$cohort_identity"
echo "[e16] submitted stage=$STAGE; full grid remains contingent on all-six Stage-S approval"
