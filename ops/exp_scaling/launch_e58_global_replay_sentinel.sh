#!/usr/bin/env bash
# Configure or atomically submit E58's smoke-bound three-domain sentinel.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

phase="${1:-}"
case "$phase" in
  config|sentinel) ;;
  *)
    echo "Usage: $0 {config|sentinel}" >&2
    exit 1
    ;;
esac

PYTHON_BIN="${OAT_ZERO_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
PROTOCOL="$ROOT_DIR/paper/preregistration/e58_global_verified_replay_canonical_05b.md"
AUDITOR="$ROOT_DIR/ops/exp_scaling/audit_e58_sentinel.py"
HELPER_AUDITOR="$ROOT_DIR/ops/exp_scaling/audit_e57_sentinel.py"
SMOKE_AUDITOR="$ROOT_DIR/ops/exp_scaling/audit_e58_global_replay_smoke.py"
SMOKE_IDENTITY="$ROOT_DIR/var/artifacts/e58_global_verified_replay_python_smoke_attempt3_identity.json"
SMOKE_AUDIT="$ROOT_DIR/var/artifacts/e58_global_replay_smoke_audit_latest.json"
E53_IDENTITY="$ROOT_DIR/var/artifacts/e53_verified_replay_05b_sentinel_identity.json"
IDENTITY="$ROOT_DIR/var/artifacts/e58_global_verified_replay_canonical_05b_sentinel_identity.json"
AUDIT_OUT="$ROOT_DIR/var/artifacts/e58_sentinel_audit_latest.json"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
GRAPH_DATA_ROOT="$ROOT_DIR/var/data/exact_answer_mode_probe"
COUNTDOWN_DATA_ROOT="$ROOT_DIR/var/data/exact_countdown_easy3_probe"
PYTHON_DATA_ROOT="$ROOT_DIR/var/data/python_factor_modebench_v1"
GRAPH_PREFIX=gce58_global_verified_replay_canonical_05b_50ep_sentinel
COUNTDOWN_PREFIX=cde58_global_verified_replay_canonical_05b_50ep_sentinel_allcs
PYTHON_PREFIX=pye58_global_verified_replay_canonical_05b_50ep_sentinel_allcs
GRAPH_MANIFEST="$ROOT_DIR/var/artifacts/${GRAPH_PREFIX}_comparative_jobs.tsv"
COUNTDOWN_MANIFEST="$ROOT_DIR/var/artifacts/${COUNTDOWN_PREFIX}_comparative_jobs.tsv"
PYTHON_MANIFEST="$ROOT_DIR/var/artifacts/${PYTHON_PREFIX}_comparative_jobs.tsv"
E53_GRAPH_MANIFEST="$ROOT_DIR/var/artifacts/gce53_verified_replay_05b_50ep_sentinel_comparative_jobs.tsv"
E53_COUNTDOWN_MANIFEST="$ROOT_DIR/var/artifacts/cde53_verified_replay_05b_50ep_sentinel_allcs_comparative_jobs.tsv"
E53_PYTHON_MANIFEST="$ROOT_DIR/var/artifacts/pye53_verified_replay_05b_50ep_sentinel_allcs_comparative_jobs.tsv"

for required in \
  "$PROTOCOL" "$AUDITOR" "$HELPER_AUDITOR" "$SMOKE_AUDITOR" \
  "$SMOKE_IDENTITY" "$SMOKE_AUDIT" "$E53_IDENTITY" \
  "$E53_GRAPH_MANIFEST" "$E53_COUNTDOWN_MANIFEST" \
  "$E53_PYTHON_MANIFEST" "$MODEL_ROOT/config.json" \
  "$GRAPH_DATA_ROOT/train/dataset_dict.json" \
  "$GRAPH_DATA_ROOT/eval/dataset_dict.json" \
  "$COUNTDOWN_DATA_ROOT/train/dataset_dict.json" \
  "$COUNTDOWN_DATA_ROOT/eval/dataset_dict.json" \
  "$PYTHON_DATA_ROOT/identity.json" \
  "$PYTHON_DATA_ROOT/train/dataset_dict.json" \
  "$PYTHON_DATA_ROOT/eval/dataset_dict.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing E58 sentinel prerequisite: $required" >&2
    exit 1
  fi
done

if [[ "$phase" == sentinel ]]; then
  for fresh in \
    "$IDENTITY" "$GRAPH_MANIFEST" "$COUNTDOWN_MANIFEST" "$PYTHON_MANIFEST"; do
    if [[ -e "$fresh" ]]; then
      echo "Fresh E58 sentinel artifact required; already exists: $fresh" >&2
      exit 1
    fi
  done
  "$PYTHON_BIN" "$SMOKE_AUDITOR" --out "$SMOKE_AUDIT"
fi

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

mapfile -t frozen < <(
  "$PYTHON_BIN" - "$SMOKE_IDENTITY" "$SMOKE_AUDIT" "$phase" <<'PY'
import json
import pathlib
import re
import sys

identity = json.loads(pathlib.Path(sys.argv[1]).read_text())
audit = json.loads(pathlib.Path(sys.argv[2]).read_text())
phase = sys.argv[3]
if identity.get("schema") != "e58_global_verified_replay_python_smoke_v1":
    raise SystemExit("E58 sentinel requires the attempt-3 smoke identity")
if identity.get("attempt") != 3:
    raise SystemExit("E58 sentinel requires smoke attempt 3")
if phase == "sentinel" and (
    audit.get("status") != "pass" or audit.get("violations")
):
    raise SystemExit("E58 sentinel requires a clean terminal Python smoke")
for key in ("source_hash", "execution_surface_hash"):
    if not identity.get(key):
        raise SystemExit(f"E58 smoke identity lacks {key}")
if phase == "sentinel":
    job_id = int(identity["job_id"])
    log_root = pathlib.Path(sys.argv[1]).parent / "logs"
    logs = [
        log_root / f"xdr_train-{job_id}.out",
        log_root / f"xdr_train-{job_id}.err",
    ]
    readable = [path for path in logs if path.is_file()]
    if not readable:
        raise SystemExit("E58 terminal smoke has no exact Slurm logs")
    crash_pattern = re.compile(
        r"Traceback \(most recent call last\)|CUDA out of memory|"
        r"torch\.OutOfMemoryError|ChildFailedError|RayActorError|"
        r"worker unexpectedly died|RuntimeError:[^\n]*non-finite|"
        r"segmentation fault",
        re.IGNORECASE,
    )
    for path in readable:
        text = path.read_text(encoding="utf-8", errors="replace")
        if crash_pattern.search(text):
            raise SystemExit(
                f"E58 smoke log contains a crash signature: {path}"
            )
print(identity["source_hash"])
print(identity["execution_surface_hash"])
PY
)
if [[ "${#frozen[@]}" -ne 2 ]]; then
  echo "Could not resolve E58 smoke-bound source and execution hashes" >&2
  exit 1
fi
SOURCE_HASH="${frozen[0]}"
EXECUTION_HASH="${frozen[1]}"
SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e58_global_verified_replay_${SOURCE_HASH}/src"
OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e58_global_verified_replay_ops_${EXECUTION_HASH}/ops"
if [[ "$(hash_tree "$SOURCE_ROOT")" != "$SOURCE_HASH" ]]; then
  echo "E58 smoke-bound source snapshot hash mismatch" >&2
  exit 1
fi
if [[ "$(hash_tree "$OPS_ROOT")" != "$EXECUTION_HASH" ]]; then
  echo "E58 smoke-bound execution snapshot hash mismatch" >&2
  exit 1
fi

write_identity() {
  local graph_job="$1" countdown_job="$2" python_job="$3"
  "$PYTHON_BIN" - \
    "$IDENTITY" "$PROTOCOL" "$0" "$AUDITOR" "$HELPER_AUDITOR" \
    "$E53_IDENTITY" "$SMOKE_IDENTITY" "$SMOKE_AUDIT" \
    "$SOURCE_HASH" "$EXECUTION_HASH" \
    "$graph_job" "$countdown_job" "$python_job" \
    "$GRAPH_MANIFEST" "$COUNTDOWN_MANIFEST" "$PYTHON_MANIFEST" \
    "$E53_GRAPH_MANIFEST" "$E53_COUNTDOWN_MANIFEST" \
    "$E53_PYTHON_MANIFEST" <<'PY'
import hashlib
import json
import os
import pathlib
import sys
import tempfile

def digest(raw):
    return hashlib.sha256(pathlib.Path(raw).read_bytes()).hexdigest()

path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "e58_global_verified_replay_canonical_05b_sentinel_v1",
    "protocol_sha256": digest(sys.argv[2]),
    "launcher_sha256": digest(sys.argv[3]),
    "auditor_sha256": digest(sys.argv[4]),
    "helper_auditor_sha256": digest(sys.argv[5]),
    "e53_control_identity_sha256": digest(sys.argv[6]),
    "smoke_identity_sha256": digest(sys.argv[7]),
    "smoke_audit_sha256": digest(sys.argv[8]),
    "source_hash": sys.argv[9],
    "execution_surface_hash": sys.argv[10],
    "jobs": {
        "graph_coloring": int(sys.argv[11]),
        "countdown": int(sys.argv[12]),
        "python_factor": int(sys.argv[13]),
    },
    "job_manifest_sha256": {
        "graph_coloring": digest(sys.argv[14]),
        "countdown": digest(sys.argv[15]),
        "python_factor": digest(sys.argv[16]),
    },
    "e53_control_manifest_sha256": {
        "graph_coloring": digest(sys.argv[17]),
        "countdown": digest(sys.argv[18]),
        "python_factor": digest(sys.argv[19]),
    },
    "attempt_selection": "exact_manifest_job_id",
    "model": (
        "Qwen2.5-0.5B-Instruct@"
        "7ae557604adf67be50417f59c2c2f167def9a775"
    ),
    "domains": {
        "graph_coloring": {"prompt_pool": 192, "eval_pool": 96},
        "countdown": {"prompt_pool": 384, "eval_pool": 128},
        "python_factor": {"prompt_pool": 384, "eval_pool": 128},
    },
    "arm": "verified_first_global_replay_canonical",
    "seed": 9010,
    "num_samples": 16,
    "global_replay": {
        "groups_per_step": 1,
        "selection": "persistent_prompt_hash_round_robin",
        "capacity": 16,
    },
    "direct_token_entropy": {"coefficient": 0.0, "controller": None},
    "controllers": {
        "open_set_semantic": {
            "base": 0.10,
            "rule": "inverse_self_warmup",
            "projection": None,
        },
        "verified_mass": {
            "base": 0.10,
            "rule": "surprisal_ratio_self_warmup",
            "projection": None,
        },
        "known_mode_balance": {
            "base": 0.10,
            "rule": "inverse_self_warmup",
            "projection": None,
        },
    },
    "cold_start": "zero_policy_gradient_until_model_verified_discovery",
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
export OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-0.5b-instruct
export OAT_ZERO_TRAIN_SEEDS=9010
export OAT_ZERO_ONLY_ARMS=verified_first_global_replay_canonical
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
  VERIFIED_FIRST_SPLIT_CANONICAL; do
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
export OAT_ZERO_ONLINE_CANONICAL_KEY_MODE=modebench_outcome
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=0.10
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_CAPACITY=16
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=1
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_WARMUP_STEPS=64
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_EMA_DECAY=0.90
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_ALPHA=0.10
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_WARMUP_STEPS=64
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_EMA_DECAY=0.90

export OAT_ZERO_NUM_SAMPLES=16
export OAT_ZERO_LEARNING_RATE=0.0000002
export OAT_ZERO_MAX_QUERIES=100000000
export OAT_ZERO_MAX_PROMPT_EPOCHS=50
export OAT_ZERO_NUM_PROMPT_EPOCH=50
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
export OAT_ZERO_EVAL_TEMPERATURE=0
export OAT_ZERO_EVAL_MODE_COVERAGE_K=8
export OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE=1
export OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=4
export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=530100
export OAT_ZERO_EVAL_BATCH_SIZE=64
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
export OAT_ZERO_MAX_SAVE_NUM=2
export OAT_ZERO_EXPORT_STEPS=0
export OAT_ZERO_MAX_EXPORT_NUM=1
export OAT_ZERO_MAX_RESUME_NUM=2
export OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=0
export OAT_ZERO_AUTO_RESUME=1
export OAT_ZERO_WATCHDOG_REQUEUE=1
export OAT_ZERO_WATCHDOG_MAX_RESTARTS=6
export OAT_ZERO_TRAIN_CPUS_PER_TASK=8
export OAT_ZERO_TRAIN_MEMORY=64G
export OAT_ZERO_TRAIN_TIME_LIMIT=7-00:00:00

submit_domain() {
  local domain="$1"
  case "$domain" in
    graph_coloring)
      export RUN_STAMP_PREFIX="$GRAPH_PREFIX"
      export OAT_ZERO_COMPARATIVE_TASK=graph_coloring
      export OAT_ZERO_COMPARATIVE_DATA_ROOT="$GRAPH_DATA_ROOT"
      export OAT_ZERO_MAX_TRAIN=192
      export OAT_ZERO_EVAL_PROMPT_INTERVAL=48
      export OAT_ZERO_SAVE_STEPS=192
      export OAT_ZERO_SAVE_FROM=192
      export OAT_ZERO_RESUME_STEPS=192
      export OAT_ZERO_RESUME_FROM=192
      export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E58_SENTINEL_GRAPH_NODELIST:-node103,node104,node208}"
      export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E58_SENTINEL_GRAPH_GRES:-gpu:a6000:1}"
      export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E58_SENTINEL_GRAPH_PARTITION:-lowprio}"
      export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E58_SENTINEL_GRAPH_ACCOUNT:-mltheory}"
      ;;
    countdown)
      export RUN_STAMP_PREFIX="$COUNTDOWN_PREFIX"
      export OAT_ZERO_COMPARATIVE_TASK=countdown
      export OAT_ZERO_COMPARATIVE_DATA_PRESET=easy3
      export OAT_ZERO_COMPARATIVE_DATA_ROOT="$COUNTDOWN_DATA_ROOT"
      export OAT_ZERO_MAX_TRAIN=384
      export OAT_ZERO_EVAL_PROMPT_INTERVAL=96
      export OAT_ZERO_SAVE_STEPS=384
      export OAT_ZERO_SAVE_FROM=384
      export OAT_ZERO_RESUME_STEPS=384
      export OAT_ZERO_RESUME_FROM=384
      export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E58_SENTINEL_COUNTDOWN_NODELIST:-node020,node022,node023,node024,node026}"
      export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E58_SENTINEL_COUNTDOWN_GRES:-gpu:rtx_3090:1}"
      export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E58_SENTINEL_COUNTDOWN_PARTITION:-lowprio}"
      export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E58_SENTINEL_COUNTDOWN_ACCOUNT:-allcs}"
      ;;
    python_factor)
      export RUN_STAMP_PREFIX="$PYTHON_PREFIX"
      export OAT_ZERO_COMPARATIVE_TASK=python_factor
      export OAT_ZERO_COMPARATIVE_DATA_ROOT="$PYTHON_DATA_ROOT"
      export OAT_ZERO_MAX_TRAIN=384
      export OAT_ZERO_EVAL_PROMPT_INTERVAL=96
      export OAT_ZERO_SAVE_STEPS=384
      export OAT_ZERO_SAVE_FROM=384
      export OAT_ZERO_RESUME_STEPS=384
      export OAT_ZERO_RESUME_FROM=384
      export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E58_SENTINEL_PYTHON_NODELIST:-node020,node022,node023,node024,node026}"
      export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E58_SENTINEL_PYTHON_GRES:-gpu:rtx_3090:1}"
      export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E58_SENTINEL_PYTHON_PARTITION:-lowprio}"
      export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E58_SENTINEL_PYTHON_ACCOUNT:-allcs}"
      ;;
    *)
      echo "Unknown E58 sentinel domain: $domain" >&2
      exit 1
      ;;
  esac
  "$OPS_ROOT/submit_countdown_comparative.sh"
}

if [[ "$phase" == config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  export OAT_ZERO_SBATCH_HOLD=0
  submit_domain graph_coloring
  submit_domain countdown
  submit_domain python_factor
  echo "[e58] all three smoke-bound sentinel configurations passed"
  exit 0
fi

export OAT_ZERO_PROTOCOL_IDENTITY="$IDENTITY"
unset OAT_ZERO_COMPARATIVE_CONFIG_ONLY
export OAT_ZERO_SBATCH_HOLD=1
released=0
job_ids=()
cleanup_held() {
  local status="$?"
  trap - EXIT
  if [[ "$released" != "1" && "${#job_ids[@]}" -gt 0 ]]; then
    scancel "${job_ids[@]}" 2>/dev/null || true
    echo "[e58] cancelled incomplete held sentinel: ${job_ids[*]}" >&2
  fi
  exit "$status"
}
trap cleanup_held EXIT

submit_domain graph_coloring
submit_domain countdown
submit_domain python_factor
for manifest in "$GRAPH_MANIFEST" "$COUNTDOWN_MANIFEST" "$PYTHON_MANIFEST"; do
  mapfile -t manifest_jobs < <(
    awk -F '\t' 'NR > 1 && $3 ~ /^[0-9]+$/ {print $3}' "$manifest"
  )
  if [[ "${#manifest_jobs[@]}" -ne 1 ]]; then
    echo "E58 manifest $manifest has ${#manifest_jobs[@]} jobs; expected one" >&2
    exit 1
  fi
  job_ids+=("${manifest_jobs[0]}")
done
write_identity "${job_ids[@]}"

for job_id in "${job_ids[@]}"; do
  job_record="$(scontrol show job "$job_id" -o)"
  for required in \
    'JobState=PENDING' \
    'Reason=JobHeldUser' \
    'OAT_ZERO_MAX_PROMPT_EPOCHS=50' \
    'OAT_ZERO_NUM_PROMPT_EPOCH=50' \
    'OAT_ZERO_SEED=9010' \
    'OAT_ZERO_VARIANT=verified_first_global_replay_canonical' \
    'OAT_ZERO_NUM_SAMPLES=16' \
    'OAT_ZERO_MAXENT_ALPHA=0' \
    'OAT_ZERO_MAXENT_INVERSE_ADAPTATION=0' \
    'OAT_ZERO_SEMANTIC_SHANNON_OPEN_SET_INVERSE_ADAPTATION=1' \
    'OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=split_mass_balance_per_rollout' \
    'OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=1' \
    'OAT_ZERO_EVAL_MODE_COVERAGE_K=8' \
    'OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=4' \
    "OAT_ZERO_SOURCE_ROOT=${SOURCE_ROOT}" \
    "OAT_ZERO_OPS_SNAPSHOT_ROOT=${OPS_ROOT}" \
    "OAT_ZERO_PROTOCOL_IDENTITY=${IDENTITY}" \
    'NumCPUs=8' \
    'MinMemoryNode=64G' \
    'TimeLimit=7-00:00:00'; do
    if [[ "$job_record" != *"$required"* ]]; then
      echo "E58 held-job audit failed for $job_id: missing $required" >&2
      exit 1
    fi
  done
done

"$PYTHON_BIN" "$AUDITOR" --out "$AUDIT_OUT"
"$PYTHON_BIN" - "$AUDIT_OUT" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
if payload.get("status") != "in_progress" or payload.get("violations"):
    raise SystemExit("E58 held sentinel failed its exact-job pre-release audit")
PY

scontrol release "${job_ids[@]}"
released=1
trap - EXIT
echo "[e58] released held sentinel cohort: ${job_ids[*]}"
echo "[e58] identity=$IDENTITY"
