#!/usr/bin/env bash
# Configure or atomically submit E64's smoke-gated MATH realism cohort.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

phase="${1:-}"
case "$phase" in
  config|full) ;;
  *)
    echo "Usage: $0 {config|full}" >&2
    exit 1
    ;;
esac

PYTHON_BIN="${OAT_ZERO_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
PROTOCOL="$ROOT_DIR/paper/preregistration/e64_math500_realism_transfer_05b.md"
SMOKE_IDENTITY="$ROOT_DIR/var/artifacts/e64_math500_realism_smoke_identity.json"
SMOKE_AUDIT="$ROOT_DIR/var/artifacts/e64_math500_realism_smoke_audit_latest.json"
SMOKE_CHECKPOINT_AUDIT="$ROOT_DIR/var/artifacts/e64_math500_realism_smoke_checkpoint_audit_latest.json"
DATA_ROOT="$ROOT_DIR/var/data/math12k_384_math500"
DATA_IDENTITY="$DATA_ROOT/MATERIALIZATION_MANIFEST.json"
AUDITOR="$ROOT_DIR/ops/exp_scaling/audit_e64_math500_realism_matched.py"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
PREFIX=mte64_math500_realism_05b_12ep
MANIFEST="$ROOT_DIR/var/artifacts/${PREFIX}_comparative_jobs.tsv"
IDENTITY="$ROOT_DIR/var/artifacts/e64_math500_realism_matched_identity.json"

for required in \
  "$PROTOCOL" "$DATA_IDENTITY" \
  "$AUDITOR" \
  "$DATA_ROOT/train/dataset_dict.json" \
  "$DATA_ROOT/eval/dataset_dict.json" \
  "$MODEL_ROOT/config.json" \
  "$MODEL_ROOT/tokenizer.json" \
  "$MODEL_ROOT/model.safetensors"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing E64 matched prerequisite: $required" >&2
    exit 1
  fi
done

"$PYTHON_BIN" - "$DATA_IDENTITY" <<'PY'
import hashlib
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
output = payload.get("output", {})
audit = payload.get("audit", {})
if payload.get("schema") != "e39_math12k_384_math500_materialization_v1":
    raise SystemExit("E64 requires the frozen MATH12K/MATH-500 materialization")
if (
    output.get("train_rows") != 384
    or output.get("eval_rows") != 500
    or output.get("train_ordered_row_sha256")
    != "051baa5571a1865518ef200c414178f1e50decea261bcd46ec0328e5837c0f36"
    or output.get("eval_ordered_row_sha256")
    != "1576fd11df21dc705a7c85000f232031212225cd9c00520faa26f6bdfc751166"
    or audit.get("normalized_problem_overlap") != 0
):
    raise SystemExit("E64 frozen train/eval boundary mismatch")
expected = {
    "train/train/data-00000-of-00001.arrow": (
        "359defbf82b6e05a1fdddb3479ed689f8a607dc727814e73ebfe69b2ffdff8b8"
    ),
    "eval/math/data-00000-of-00001.arrow": (
        "2104f8f8eef09ce0bfc929e255f0f04c59311f1f3395bd293f1d03051c482cf7"
    ),
}
for relative, digest in expected.items():
    actual = hashlib.sha256((path.parent / relative).read_bytes()).hexdigest()
    if actual != digest:
        raise SystemExit(f"E64 frozen data hash mismatch: {relative}")
PY

hash_tree() {
  local tree="$1"
  (
    cd "$tree"
    find . -type f -print0 \
      | sort -z \
      | xargs -0 sha256sum \
      | sha256sum \
      | cut -d' ' -f1
  )
}

if [[ "$phase" == config ]]; then
  SOURCE_HASH="$(hash_tree "$ROOT_DIR/src")"
  EXECUTION_HASH=config-only-current-tree
  SOURCE_ROOT="$ROOT_DIR/src"
  OPS_ROOT="$ROOT_DIR/ops"
else
  for required in \
    "$SMOKE_IDENTITY" "$SMOKE_AUDIT" "$SMOKE_CHECKPOINT_AUDIT"; do
    if [[ ! -f "$required" ]]; then
      echo "E64 full cohort requires terminal smoke artifact: $required" >&2
      exit 1
    fi
  done
  for fresh in "$MANIFEST" "$IDENTITY"; do
    if [[ -e "$fresh" ]]; then
      echo "Fresh E64 matched artifact required; already exists: $fresh" >&2
      exit 1
    fi
  done
  mapfile -t smoke_binding < <(
    "$PYTHON_BIN" - \
      "$SMOKE_IDENTITY" "$SMOKE_AUDIT" "$SMOKE_CHECKPOINT_AUDIT" <<'PY'
import json
import pathlib
import sys

identity_path = pathlib.Path(sys.argv[1]).resolve()
identity = json.loads(identity_path.read_text(encoding="utf-8"))
audit = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
checkpoint_audit = json.loads(
    pathlib.Path(sys.argv[3]).read_text(encoding="utf-8")
)
if identity.get("schema") != "e64_math500_realism_smoke_v1":
    raise SystemExit("E64 matched cohort requires the exact smoke identity")
if (
    audit.get("schema") != "e64_math500_realism_smoke_audit_v1"
    or audit.get("status") != "pass"
    or audit.get("violations")
    or pathlib.Path(audit.get("identity", "")).resolve() != identity_path
    or audit.get("latest_step", -1) < 96
    or audit.get("first_discovery_step") is None
    or audit.get("controller_observations", {}).get("balance") != 0
    or audit.get("checkpoint_gate", {}).get("status") != "pass"
    or checkpoint_audit.get("schema")
    != "e64_math500_realism_smoke_checkpoint_audit_v1"
    or checkpoint_audit.get("status") != "pass"
    or checkpoint_audit.get("violations")
    or pathlib.Path(checkpoint_audit.get("identity", "")).resolve()
    != identity_path
    or checkpoint_audit.get("model_tensor_count", 0) <= 0
    or checkpoint_audit.get("model_parameter_count", 0) <= 0
    or checkpoint_audit.get("nonfinite_model_tensors")
):
    raise SystemExit("E64 matched cohort requires a clean terminal smoke")
print(identity["source_hash"])
print(identity["execution_surface_hash"])
PY
  )
  if [[ "${#smoke_binding[@]}" -ne 2 ]]; then
    echo "Could not resolve E64 smoke-bound source" >&2
    exit 1
  fi
  SOURCE_HASH="${smoke_binding[0]}"
  EXECUTION_HASH="${smoke_binding[1]}"
  SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e64_math500_realism_${SOURCE_HASH}/src"
  OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e64_math500_realism_ops_${EXECUTION_HASH}/ops"
  if [[ "$(hash_tree "$SOURCE_ROOT")" != "$SOURCE_HASH" ]]; then
    echo "E64 smoke-bound source mismatch" >&2
    exit 1
  fi
  if [[ "$(hash_tree "$OPS_ROOT")" != "$EXECUTION_HASH" ]]; then
    echo "E64 smoke-bound execution surface mismatch" >&2
    exit 1
  fi
fi

write_identity() {
  "$PYTHON_BIN" - \
    "$IDENTITY" "$PROTOCOL" "$0" "$AUDITOR" "$MANIFEST" "$DATA_IDENTITY" \
    "$SMOKE_IDENTITY" "$SMOKE_AUDIT" "$SMOKE_CHECKPOINT_AUDIT" \
    "$SOURCE_HASH" "$EXECUTION_HASH" "$@" <<'PY'
import hashlib
import json
import os
import pathlib
import sys
import tempfile

def digest(raw: str) -> str:
    return hashlib.sha256(pathlib.Path(raw).read_bytes()).hexdigest()

path = pathlib.Path(sys.argv[1])
job_ids = [int(value) for value in sys.argv[12:]]
payload = {
    "schema": "e64_math500_realism_matched_v1",
    "protocol_sha256": digest(sys.argv[2]),
    "launcher_sha256": digest(sys.argv[3]),
    "auditor_sha256": digest(sys.argv[4]),
    "manifest_sha256": digest(sys.argv[5]),
    "data_manifest_sha256": digest(sys.argv[6]),
    "smoke_identity_sha256": digest(sys.argv[7]),
    "smoke_audit_sha256": digest(sys.argv[8]),
    "smoke_checkpoint_audit_sha256": digest(sys.argv[9]),
    "source_hash": sys.argv[10],
    "execution_surface_hash": sys.argv[11],
    "job_ids": job_ids,
    "attempt_selection": "exact_manifest_job_ids",
    "model": (
        "Qwen2.5-0.5B-Instruct@"
        "7ae557604adf67be50417f59c2c2f167def9a775"
    ),
    "domain": "MATH12K-384 train / held-out MATH-500 evaluation",
    "track": "external_validity_generalization_not_modebench_mode_coverage",
    "arms": ["grpo", "verified_first_global_replay_canonical"],
    "seeds": [43, 44, 45],
    "prompt_epochs": 12,
    "optimizer_updates_per_run": 4608,
    "num_samples": 16,
    "learning_rate": 2e-7,
    "canonical_key_mode": "math_verified_answer",
    "canonical_contract": {
        "reward_positive_key": "math_verified_answer:correct",
        "reward_zero_key": None,
        "maximum_support_per_prompt": 1,
        "reasoning_strategy_claim": False,
    },
    "evaluation": {
        "split": "held-out MATH-500",
        "rows": 500,
        "passes": [0, 2, 4, 6, 8, 10, 12],
        "sampled_k": 8,
        "draws": 1,
        "seed": 640100,
        "distinct_is_reasoning_endpoint": False,
    },
    "coefficient_projection": None,
    "information_firewall": {
        "gold_support_feedback": False,
        "evaluation_feedback": False,
        "desired_entropy": None,
        "desired_mode_count": None,
        "math500_is_training_input": False,
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
export OAT_ZERO_COMPARATIVE_TASK=math
export OAT_ZERO_COMPARATIVE_DATA_ROOT="$DATA_ROOT"
export OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-0.5b-instruct
export OAT_ZERO_TRAIN_SEEDS=43,44,45
export OAT_ZERO_ONLY_ARMS=grpo,verified_first_global_replay_canonical
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
  MAXENT_INVERSE_CANONICAL_REPLAY OPEN_SET_SPLIT_CANONICAL \
  VERIFIED_FIRST_SPLIT_CANONICAL VERIFIED_FIRST_BOOTSTRAP_LOCAL_CANONICAL \
  VERIFIED_COUNTERFACTUAL_CANONICAL \
  VERIFIED_FIRST_GLOBAL_REPLAY_CANONICAL; do
  export "OAT_ZERO_INCLUDE_${flag}_ARM=0"
done
export OAT_ZERO_INCLUDE_VERIFIED_FIRST_GLOBAL_REPLAY_CANONICAL_ARM=1

export OAT_ZERO_VERIFIED_DISCOVERY_TRACKING=1
export OAT_ZERO_MAXENT_ALPHA=0
export OAT_ZERO_MAXENT_INVERSE_BASE_ALPHA=0
export OAT_ZERO_SEMANTIC_SHANNON_COEF=0.10
export OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP=5.0
export OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT=1.0
export OAT_ZERO_SEMANTIC_SHANNON_OPEN_SET_WARMUP_STEPS=64
export OAT_ZERO_SEMANTIC_SHANNON_OPEN_SET_EMA_DECAY=0.90
export OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0
export OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.50
export OAT_ZERO_ONLINE_CANONICAL_BANK_PSEUDOCOUNT=1.0
export OAT_ZERO_ONLINE_CANONICAL_BANK_SURPRISAL_CLIP=5.0
export OAT_ZERO_ONLINE_CANONICAL_KEY_MODE=math_verified_answer
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=0.10
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_CAPACITY=16
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=1
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_BOOTSTRAP_STEPS=0
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_WARMUP_STEPS=64
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_EMA_DECAY=0.90
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_ALPHA=0.10
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_WARMUP_STEPS=64
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_EMA_DECAY=0.90

export OAT_ZERO_NUM_SAMPLES=16
export OAT_ZERO_LEARNING_RATE=0.0000002
export OAT_ZERO_MAX_TRAIN=384
export OAT_ZERO_MAX_QUERIES=100000000
export OAT_ZERO_MAX_PROMPT_EPOCHS=12
export OAT_ZERO_NUM_PROMPT_EPOCH=12
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
export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=640100
export OAT_ZERO_EVAL_PROMPT_INTERVAL=768
export OAT_ZERO_EVAL_STEPS=768
export OAT_ZERO_EVAL_BATCH_SIZE=64
export OAT_ZERO_ALLOW_SPARSE_EVAL=0
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
export OAT_ZERO_SAVE_CKPT=1
export OAT_ZERO_SAVE_STEPS=384
export OAT_ZERO_SAVE_FROM=384
export OAT_ZERO_MAX_SAVE_NUM=2
export OAT_ZERO_RESUME_STEPS=384
export OAT_ZERO_RESUME_FROM=384
export OAT_ZERO_MAX_RESUME_NUM=2
export OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=0
export OAT_ZERO_AUTO_RESUME=1
export OAT_ZERO_WATCHDOG_REQUEUE=1
export OAT_ZERO_WATCHDOG_MAX_RESTARTS=6
export OAT_ZERO_TRAIN_CPUS_PER_TASK=8
export OAT_ZERO_TRAIN_MEMORY=64G
export OAT_ZERO_TRAIN_TIME_LIMIT=7-00:00:00
export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E64_MATH_NODELIST:-node020,node021,node022,node023,node024,node026}"
export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E64_MATH_GRES:-gpu:rtx_3090:1}"
export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E64_MATH_PARTITION:-lowprio}"
export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E64_MATH_ACCOUNT:-allcs}"

if [[ "$phase" == config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  export OAT_ZERO_SBATCH_HOLD=0
  "$OPS_ROOT/submit_countdown_comparative.sh"
  echo "[e64] matched MATH realism configuration passed"
  exit 0
fi

export OAT_ZERO_PROTOCOL_IDENTITY="$IDENTITY"
export OAT_ZERO_SBATCH_HOLD=1
released=0
job_ids=()
cleanup_held() {
  local status="$?"
  trap - EXIT
  if [[ "$released" != "1" && "${#job_ids[@]}" -gt 0 ]]; then
    scancel "${job_ids[@]}" 2>/dev/null || true
    echo "[e64] cancelled incomplete held cohort: ${job_ids[*]}" >&2
  fi
  exit "$status"
}
trap cleanup_held EXIT

"$OPS_ROOT/submit_countdown_comparative.sh"
mapfile -t job_ids < <(
  awk -F $'\t' 'NR > 1 && $3 ~ /^[0-9]+$/ {print $3}' "$MANIFEST"
)
if [[ "${#job_ids[@]}" -ne 6 ]]; then
  echo "E64 matched cohort has ${#job_ids[@]} jobs; expected six" >&2
  exit 1
fi
write_identity "${job_ids[@]}"

for job_id in "${job_ids[@]}"; do
  job_record="$(scontrol show job "$job_id" -o)"
  for required in \
    'JobState=PENDING' \
    'Reason=JobHeldUser' \
    'OAT_ZERO_COMPARATIVE_TASK=math' \
    'OAT_ZERO_PROMPT_TEMPLATE=qwen_math' \
    'OAT_ZERO_VERIFIER_VERSION=math_verify' \
    'OAT_ZERO_TEST_SPLIT=math' \
    'OAT_ZERO_ONLINE_CANONICAL_KEY_MODE=math_verified_answer' \
    'OAT_ZERO_MAX_TRAIN=384' \
    'OAT_ZERO_MAX_PROMPT_EPOCHS=12' \
    'OAT_ZERO_NUM_PROMPT_EPOCH=12' \
    'OAT_ZERO_NUM_SAMPLES=16' \
    'OAT_ZERO_EVAL_PROMPT_INTERVAL=768' \
    'OAT_ZERO_EVAL_MODE_COVERAGE_K=8' \
    'OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=1' \
    'OAT_ZERO_EVAL_MODE_COVERAGE_SEED=640100' \
    "OAT_ZERO_SOURCE_ROOT=${SOURCE_ROOT}" \
    "OAT_ZERO_OPS_SNAPSHOT_ROOT=${OPS_ROOT}" \
    "OAT_ZERO_PROTOCOL_IDENTITY=${IDENTITY}"; do
    if [[ "$job_record" != *"$required"* ]]; then
      echo "E64 held-job audit failed for $job_id: missing $required" >&2
      exit 1
    fi
  done
  if [[ "$job_record" == *'OAT_ZERO_VARIANT=verified_first_global_replay_canonical'* ]]; then
    for required in \
      'OAT_ZERO_MAXENT_ALPHA=0.0' \
      'OAT_ZERO_MAXENT_INVERSE_ADAPTATION=0' \
      'OAT_ZERO_SEMANTIC_SHANNON_OPEN_SET_INVERSE_ADAPTATION=1' \
      'OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=1' \
      'OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=split_mass_balance_per_rollout'; do
      if [[ "$job_record" != *"$required"* ]]; then
        echo "E64 treatment audit failed for $job_id: missing $required" >&2
        exit 1
      fi
    done
  elif [[ "$job_record" != *'OAT_ZERO_VARIANT=grpo'* ]]; then
    echo "E64 job $job_id is neither frozen arm" >&2
    exit 1
  fi
done

scontrol release "${job_ids[@]}"
released=1
trap - EXIT
echo "[e64] released six matched MATH realism jobs: ${job_ids[*]}"
echo "[e64] identity=$IDENTITY"
