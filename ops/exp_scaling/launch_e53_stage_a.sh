#!/usr/bin/env bash
# Launch E53's conditionally authorized 3-domain x 3-arm x 3-seed cohort.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
phase="${1:-}"
case "$phase" in
  config|stage_a) ;;
  *)
    echo "Usage: $0 {config|stage_a}" >&2
    exit 1
    ;;
esac

PYTHON_BIN="${OAT_ZERO_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
STAGE_PROTOCOL="$ROOT_DIR/paper/preregistration/e53_stage_a_execution_20260726.md"
RUNTIME_REPAIR_PROTOCOL="$ROOT_DIR/paper/preregistration/e53_runtime_audit_scaling_repair_20260726.md"
SENTINEL_AUDITOR_V2="$ROOT_DIR/ops/exp_scaling/audit_e53_sentinel_v2.py"
VERIFIER="$ROOT_DIR/ops/exp_scaling/verify_e53_sentinel_approval.py"
AUDITOR="$ROOT_DIR/ops/exp_scaling/audit_e53_stage_a.py"
WATCHER="$ROOT_DIR/ops/exp_scaling/watch_e53_stage_a.sh"
WATCHER_SLURM="$ROOT_DIR/ops/slurm/watch_e53_stage_a.slurm"
APPROVAL="$ROOT_DIR/var/artifacts/e53_sentinel_stage_a_approval.json"
SENTINEL_IDENTITY="$ROOT_DIR/var/artifacts/e53_verified_replay_05b_sentinel_identity.json"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
GRAPH_DATA_ROOT="$ROOT_DIR/var/data/exact_answer_mode_probe"
COUNTDOWN_DATA_ROOT="$ROOT_DIR/var/data/exact_countdown_easy3_probe"
PYTHON_DATA_ROOT="$ROOT_DIR/var/data/python_factor_modebench_v1"
GRAPH_PREFIX=gce53_verified_replay_05b_50ep_stage_a
COUNTDOWN_PREFIX=cde53_verified_replay_05b_50ep_stage_a_allcs
PYTHON_PREFIX=pye53_verified_replay_05b_50ep_stage_a_allcs
IDENTITY="$ROOT_DIR/var/artifacts/e53_verified_replay_05b_stage_a_identity.json"
EXPECTED_JOBS_PER_DOMAIN=9
EXPECTED_JOBS=27
PROMPT_EPOCHS=50

for required in \
  "$PYTHON_BIN" "$STAGE_PROTOCOL" "$RUNTIME_REPAIR_PROTOCOL" \
  "$SENTINEL_AUDITOR_V2" "$VERIFIER" "$AUDITOR" "$WATCHER" \
  "$WATCHER_SLURM" "$SENTINEL_IDENTITY" "$MODEL_ROOT/config.json" \
  "$MODEL_ROOT/tokenizer.json" "$MODEL_ROOT/model.safetensors" \
  "$GRAPH_DATA_ROOT/train/dataset_dict.json" \
  "$GRAPH_DATA_ROOT/eval/dataset_dict.json" \
  "$COUNTDOWN_DATA_ROOT/train/dataset_dict.json" \
  "$COUNTDOWN_DATA_ROOT/eval/dataset_dict.json" \
  "$PYTHON_DATA_ROOT/identity.json" \
  "$PYTHON_DATA_ROOT/train/dataset_dict.json" \
  "$PYTHON_DATA_ROOT/eval/dataset_dict.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing frozen E53 Stage-A prerequisite: $required" >&2
    exit 1
  fi
done
if ! grep -q 'FROZEN DURING SENTINEL PASS 0' "$STAGE_PROTOCOL"; then
  echo "E53 Stage-A protocol is not frozen" >&2
  exit 1
fi

mapfile -t sentinel_fields < <(
  "$PYTHON_BIN" - "$SENTINEL_IDENTITY" <<'PY'
import json
import pathlib
import sys

identity = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if identity.get("schema") != "e53_verified_exemplar_replay_05b_sentinel_v1":
    raise SystemExit("incompatible E53 sentinel identity")
print(identity["source_hash"])
print(identity["execution_surface_hash"])
PY
)
if [[ "${#sentinel_fields[@]}" -ne 2 ]]; then
  echo "E53 sentinel identity did not resolve both execution hashes" >&2
  exit 1
fi
SOURCE_HASH="${sentinel_fields[0]}"
EXECUTION_HASH="${sentinel_fields[1]}"
SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e53_verified_replay_${SOURCE_HASH}/src"
OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e53_verified_replay_ops_${EXECUTION_HASH}/ops"
if [[ ! -f "$SOURCE_ROOT/oat_drgrpo/__init__.py" \
  || ! -f "$OPS_ROOT/submit_countdown_comparative.sh" \
  || ! -f "$OPS_ROOT/repo_env.sh" ]]; then
  echo "E53 immutable sentinel execution snapshots are unavailable" >&2
  exit 1
fi
export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
source "$OPS_ROOT/repo_env.sh"

for prefix in "$GRAPH_PREFIX" "$COUNTDOWN_PREFIX" "$PYTHON_PREFIX"; do
  manifest="$ROOT_DIR/var/artifacts/${prefix}_comparative_jobs.tsv"
  if [[ "$phase" == stage_a && -e "$manifest" ]]; then
    echo "Fresh E53 Stage-A prefix required; manifest exists: $manifest" >&2
    exit 1
  fi
done
if [[ "$phase" == stage_a && -e "$IDENTITY" ]]; then
  echo "Fresh E53 Stage-A identity required; file exists: $IDENTITY" >&2
  exit 1
fi

APPROVAL_SHA256=""
if [[ "$phase" == stage_a ]]; then
  if [[ ! -f "$APPROVAL" ]]; then
    echo "E53 Stage A requires terminal sentinel approval: $APPROVAL" >&2
    exit 1
  fi
  APPROVAL_SHA256="$(sha256sum "$APPROVAL" | cut -d' ' -f1)"
  "$PYTHON_BIN" "$VERIFIER" \
    --approval "$APPROVAL" \
    --approval-sha256 "$APPROVAL_SHA256"
fi

export OAT_ZERO_OPS_SNAPSHOT_ROOT="$OPS_ROOT"
export OAT_ZERO_CAMPAIGN_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_PRETRAIN="$MODEL_ROOT"
export OAT_ZERO_REQUIRE_EXISTING_DATA=1
export OAT_ZERO_COMPARATIVE_REBUILD=0
export OAT_ZERO_APPEND_MANIFEST=0
export OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-0.5b-instruct
export OAT_ZERO_TRAIN_SEEDS=43,44,45
export OAT_ZERO_ONLY_ARMS=grpo,maxent_inverse,maxent_inverse_canonical_replay
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
  MAXENT_INVERSE_CANONICAL; do
  export "OAT_ZERO_INCLUDE_${flag}_ARM=0"
done
export OAT_ZERO_INCLUDE_MAXENT_INVERSE_ARM=1
export OAT_ZERO_INCLUDE_MAXENT_INVERSE_CANONICAL_REPLAY_ARM=1

export OAT_ZERO_VERIFIED_DISCOVERY_TRACKING=1
export OAT_ZERO_MAXENT_OBJECTIVE=conditional_token_mean
export OAT_ZERO_MAXENT_ALPHA=0.000075
export OAT_ZERO_MAXENT_INVERSE_BASE_ALPHA=0.000075
export OAT_ZERO_MAXENT_INVERSE_ADAPTATION=1
export OAT_ZERO_MAXENT_INVERSE_WARMUP_STEPS=64
export OAT_ZERO_MAXENT_INVERSE_EMA_DECAY=0.90
export OAT_ZERO_MAXENT_CONTROL_TARGET_RATIO=0.0
export OAT_ZERO_MAXENT_DUAL_TARGET_RATIO=0.0
export OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.0
export OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.50
export OAT_ZERO_ONLINE_CANONICAL_BANK_PSEUDOCOUNT=1.0
export OAT_ZERO_ONLINE_CANONICAL_BANK_SURPRISAL_CLIP=5.0
export OAT_ZERO_ONLINE_CANONICAL_KEY_MODE=modebench_outcome
export OAT_ZERO_ONLINE_CANONICAL_DUAL_TARGET_RATIO=0.0
export OAT_ZERO_ONLINE_CANONICAL_POLICY_ENTROPY_ADAPTATION=0
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=0.10
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_CAPACITY=16
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_WARMUP_STEPS=64
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_EMA_DECAY=0.90
export OAT_ZERO_OUTCOME_COLLISION_COEF=0
export OAT_ZERO_SEMANTIC_SHANNON_COEF=0
export OAT_ZERO_POLICY_ENTROPY_COEF=0
export OAT_ZERO_SEED_ENTROPY_ALPHA=0
export OAT_ZERO_DIAYN_NUM_OPTIONS=0
export OAT_ZERO_DIAYN_MI_BETA=0

export OAT_ZERO_NUM_SAMPLES=16
export OAT_ZERO_LEARNING_RATE=0.0000002
export OAT_ZERO_MAX_QUERIES=100000000
export OAT_ZERO_MAX_PROMPT_EPOCHS="$PROMPT_EPOCHS"
export OAT_ZERO_NUM_PROMPT_EPOCH="$PROMPT_EPOCHS"
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

write_identity() {
  local protocol_hash repair_protocol_hash sentinel_auditor_hash
  local launcher_hash verifier_hash auditor_hash
  local watcher_hash watcher_slurm_hash sentinel_identity_hash
  protocol_hash="$(sha256sum "$STAGE_PROTOCOL" | cut -d' ' -f1)"
  repair_protocol_hash="$(
    sha256sum "$RUNTIME_REPAIR_PROTOCOL" | cut -d' ' -f1
  )"
  sentinel_auditor_hash="$(
    sha256sum "$SENTINEL_AUDITOR_V2" | cut -d' ' -f1
  )"
  launcher_hash="$(sha256sum "$0" | cut -d' ' -f1)"
  verifier_hash="$(sha256sum "$VERIFIER" | cut -d' ' -f1)"
  auditor_hash="$(sha256sum "$AUDITOR" | cut -d' ' -f1)"
  watcher_hash="$(sha256sum "$WATCHER" | cut -d' ' -f1)"
  watcher_slurm_hash="$(sha256sum "$WATCHER_SLURM" | cut -d' ' -f1)"
  sentinel_identity_hash="$(sha256sum "$SENTINEL_IDENTITY" | cut -d' ' -f1)"
  "$PYTHON_BIN" - "$IDENTITY" "$protocol_hash" "$repair_protocol_hash" \
    "$sentinel_auditor_hash" "$launcher_hash" "$verifier_hash" \
    "$auditor_hash" "$watcher_hash" "$watcher_slurm_hash" \
    "$sentinel_identity_hash" "$APPROVAL" "$APPROVAL_SHA256" \
    "$SOURCE_HASH" "$EXECUTION_HASH" <<'PY'
import json
import os
import pathlib
import sys
import tempfile

path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "e53_verified_replay_05b_stage_a_v1",
    "stage_protocol_sha256": sys.argv[2],
    "runtime_repair_protocol_sha256": sys.argv[3],
    "sentinel_runtime_auditor_sha256": sys.argv[4],
    "launcher_sha256": sys.argv[5],
    "approval_verifier_sha256": sys.argv[6],
    "stage_a_auditor_sha256": sys.argv[7],
    "stage_a_watcher_sha256": sys.argv[8],
    "stage_a_watcher_slurm_sha256": sys.argv[9],
    "sentinel_identity_sha256": sys.argv[10],
    "sentinel_approval": str(pathlib.Path(sys.argv[11]).resolve()),
    "sentinel_approval_sha256": sys.argv[12],
    "source_hash": sys.argv[13],
    "execution_surface_hash": sys.argv[14],
    "model": "Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775",
    "domains": {
        "graph_coloring": {"prompt_pool": 192, "eval_pool": 96},
        "countdown": {"prompt_pool": 384, "eval_pool": 128},
        "python_factor": {"prompt_pool": 384, "eval_pool": 128},
    },
    "arms": [
        "grpo", "maxent_inverse", "maxent_inverse_canonical_replay"
    ],
    "seeds": [43, 44, 45],
    "prompt_epochs": 50,
    "num_samples": 16,
    "resume_from_sentinel": False,
    "direct_controller": {
        "base_alpha": 0.000075, "warmup_steps": 64,
        "ema_decay": 0.90, "projection": None,
    },
    "replay": {
        "base_alpha": 0.10, "capacity": 16,
        "warmup_eligible_steps": 64, "ema_decay": 0.90,
        "projection": None, "gold_support_feedback": False,
    },
    "bank": {"entropy_alpha": 0.0, "novelty_beta": 0.50},
}
path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY
}

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
      export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E53_GRAPH_NODELIST:-node302}"
      export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E53_GRAPH_GRES:-gpu:a100:1}"
      export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E53_GRAPH_PARTITION:-all}"
      export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E53_GRAPH_ACCOUNT:-mltheory}"
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
      export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E53_COUNTDOWN_NODELIST:-node020,node022,node023,node024,node026}"
      export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E53_COUNTDOWN_GRES:-gpu:rtx_3090:1}"
      export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E53_COUNTDOWN_PARTITION:-lowprio}"
      export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E53_COUNTDOWN_ACCOUNT:-allcs}"
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
      export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E53_PYTHON_NODELIST:-node020,node022,node023,node024,node026}"
      export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E53_PYTHON_GRES:-gpu:rtx_3090:1}"
      export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E53_PYTHON_PARTITION:-lowprio}"
      export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E53_PYTHON_ACCOUNT:-allcs}"
      ;;
    *)
      echo "Unsupported E53 Stage-A domain: $domain" >&2
      exit 1
      ;;
  esac
  export OAT_ZERO_TRAIN_CPUS_PER_TASK=8
  export OAT_ZERO_TRAIN_MEMORY=64G
  export OAT_ZERO_TRAIN_TIME_LIMIT=7-00:00:00
  "$OPS_ROOT/submit_countdown_comparative.sh"
}

if [[ "$phase" == config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  export OAT_ZERO_SBATCH_HOLD=0
  submit_domain graph_coloring
  submit_domain countdown
  submit_domain python_factor
  echo "[e53-stage-a] all three configurations passed; no jobs submitted"
  exit 0
fi

unset OAT_ZERO_COMPARATIVE_CONFIG_ONLY
export OAT_ZERO_SBATCH_HOLD=1
COHORT_RELEASED=0
job_ids=()
monitor_job_id=""
cleanup_partial_cohort() {
  local status="$?"
  local prefix manifest
  local -a discovered=()
  local -a partial=("${job_ids[@]}")
  trap - EXIT
  if [[ "$COHORT_RELEASED" != "1" ]]; then
    for prefix in "$GRAPH_PREFIX" "$COUNTDOWN_PREFIX" "$PYTHON_PREFIX"; do
      manifest="$ROOT_DIR/var/artifacts/${prefix}_comparative_jobs.tsv"
      if [[ -f "$manifest" ]]; then
        mapfile -t discovered < <(
          awk -F '\t' 'NR > 1 && $3 ~ /^[0-9]+$/ {print $3}' "$manifest"
        )
        partial+=("${discovered[@]}")
      fi
    done
    if [[ "${#partial[@]}" -gt 0 ]]; then
      scancel "${partial[@]}" 2>/dev/null || true
    fi
    if [[ -n "$monitor_job_id" ]]; then
      scancel "$monitor_job_id" 2>/dev/null || true
    fi
  fi
  exit "$status"
}
trap cleanup_partial_cohort EXIT

write_identity
export OAT_ZERO_PROTOCOL_IDENTITY="$IDENTITY"
submit_domain graph_coloring
submit_domain countdown
submit_domain python_factor
monitor_job_id="$(
  sbatch --parsable --hold \
    --partition=mltheory --account=mltheory --nodelist=node915 \
    --cpus-per-task=2 --mem=8G --time=4-00:00:00 \
    "$WATCHER_SLURM"
)"

for prefix in "$GRAPH_PREFIX" "$COUNTDOWN_PREFIX" "$PYTHON_PREFIX"; do
  manifest="$ROOT_DIR/var/artifacts/${prefix}_comparative_jobs.tsv"
  mapfile -t task_jobs < <(
    awk -F '\t' 'NR > 1 && $3 ~ /^[0-9]+$/ {print $3}' "$manifest"
  )
  if [[ "${#task_jobs[@]}" -ne "$EXPECTED_JOBS_PER_DOMAIN" ]]; then
    echo "E53 Stage-A ${prefix} has ${#task_jobs[@]} jobs; expected 9" >&2
    exit 1
  fi
  job_ids+=("${task_jobs[@]}")
  "$PYTHON_BIN" - "$manifest" <<'PY'
import csv
import pathlib
import sys

with pathlib.Path(sys.argv[1]).open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
expected = {
    (arm, str(seed))
    for arm in ("grpo", "maxent_inverse", "maxent_inverse_canonical_replay")
    for seed in (43, 44, 45)
}
observed = {(row["arm"], row["seed"]) for row in rows}
if len(rows) != 9 or observed != expected:
    raise SystemExit(f"E53 Stage-A held manifest mismatch: {observed!r}")
PY
done
if [[ "${#job_ids[@]}" -ne "$EXPECTED_JOBS" ]]; then
  echo "E53 Stage A has ${#job_ids[@]} jobs; expected $EXPECTED_JOBS" >&2
  exit 1
fi

for job_id in "${job_ids[@]}"; do
  job_record="$(scontrol show job "$job_id" -o)"
  for required in \
    'JobState=PENDING' 'Reason=JobHeldUser' \
    'OAT_ZERO_MAX_PROMPT_EPOCHS=50' \
    'OAT_ZERO_NUM_PROMPT_EPOCH=50' \
    'OAT_ZERO_VERIFIED_DISCOVERY_TRACKING=1' \
    'OAT_ZERO_NUM_SAMPLES=16' \
    'OAT_ZERO_MAXENT_OBJECTIVE=conditional_token_mean' \
    'OAT_ZERO_ONLINE_CANONICAL_REPLAY_CAPACITY=16' \
    'OAT_ZERO_EVAL_MODE_COVERAGE_K=8'; do
    if [[ "$job_record" != *"$required"* ]]; then
      echo "E53 Stage-A job $job_id lacks held config: $required" >&2
      exit 1
    fi
  done
done
scontrol release "${job_ids[@]}" "$monitor_job_id"
COHORT_RELEASED=1
trap - EXIT
echo "[e53-stage-a] released 27 jobs and monitor $monitor_job_id"
