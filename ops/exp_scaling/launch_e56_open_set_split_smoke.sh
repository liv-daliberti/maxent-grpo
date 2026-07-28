#!/usr/bin/env bash
# Launch target-free E56 engineering smokes or the frozen three-domain
# sentinel. Sentinel submission is fail-closed on the completed smoke audit.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

phase="${1:-}"
case "$phase" in
  config|graph|python|sentinel_config|sentinel) ;;
  *)
    echo "Usage: $0 {config|graph|python|sentinel_config|sentinel}" >&2
    exit 1
    ;;
esac

PYTHON_BIN="${OAT_ZERO_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
PROTOCOL="$ROOT_DIR/paper/preregistration/e56_open_set_split_controller_05b.md"
AUDITOR="$ROOT_DIR/ops/exp_scaling/audit_e56_sentinel.py"
SMOKE_AUDIT="$ROOT_DIR/var/artifacts/e56_smoke_audit_latest.json"
E53_IDENTITY="$ROOT_DIR/var/artifacts/e53_verified_replay_05b_sentinel_identity.json"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
GRAPH_DATA_ROOT="$ROOT_DIR/var/data/exact_answer_mode_probe"
COUNTDOWN_DATA_ROOT="$ROOT_DIR/var/data/exact_countdown_easy3_probe"
PYTHON_DATA_ROOT="$ROOT_DIR/var/data/python_factor_modebench_v1"
GRAPH_PREFIX=e56_open_set_split_smoke_graph_telemetry_v2
PYTHON_PREFIX=e56_open_set_split_smoke_python_telemetry_v2
GRAPH_IDENTITY="$ROOT_DIR/var/artifacts/e56_open_set_split_graph_smoke_telemetry_v2_identity.json"
PYTHON_IDENTITY="$ROOT_DIR/var/artifacts/e56_open_set_split_python_smoke_telemetry_v2_identity.json"
GRAPH_SENTINEL_PREFIX=gce56_open_set_split_canonical_05b_50ep_sentinel
COUNTDOWN_SENTINEL_PREFIX=cde56_open_set_split_canonical_05b_50ep_sentinel_allcs
PYTHON_SENTINEL_PREFIX=pye56_open_set_split_canonical_05b_50ep_sentinel_allcs
SENTINEL_IDENTITY="$ROOT_DIR/var/artifacts/e56_open_set_split_canonical_05b_sentinel_identity.json"
E53_GRAPH_MANIFEST="$ROOT_DIR/var/artifacts/gce53_verified_replay_05b_50ep_sentinel_comparative_jobs.tsv"
E53_COUNTDOWN_MANIFEST="$ROOT_DIR/var/artifacts/cde53_verified_replay_05b_50ep_sentinel_allcs_comparative_jobs.tsv"
E53_PYTHON_MANIFEST="$ROOT_DIR/var/artifacts/pye53_verified_replay_05b_50ep_sentinel_allcs_comparative_jobs.tsv"
GRAPH_SENTINEL_MANIFEST="$ROOT_DIR/var/artifacts/${GRAPH_SENTINEL_PREFIX}_comparative_jobs.tsv"
COUNTDOWN_SENTINEL_MANIFEST="$ROOT_DIR/var/artifacts/${COUNTDOWN_SENTINEL_PREFIX}_comparative_jobs.tsv"
PYTHON_SENTINEL_MANIFEST="$ROOT_DIR/var/artifacts/${PYTHON_SENTINEL_PREFIX}_comparative_jobs.tsv"

for required in \
  "$PROTOCOL" "$AUDITOR" "$SMOKE_AUDIT" "$E53_IDENTITY" \
  "$E53_GRAPH_MANIFEST" "$E53_COUNTDOWN_MANIFEST" \
  "$E53_PYTHON_MANIFEST" \
  "$MODEL_ROOT/config.json" \
  "$GRAPH_DATA_ROOT/train/dataset_dict.json" \
  "$GRAPH_DATA_ROOT/eval/dataset_dict.json" \
  "$COUNTDOWN_DATA_ROOT/train/dataset_dict.json" \
  "$COUNTDOWN_DATA_ROOT/eval/dataset_dict.json" \
  "$PYTHON_DATA_ROOT/identity.json" \
  "$PYTHON_DATA_ROOT/train/dataset_dict.json" \
  "$PYTHON_DATA_ROOT/eval/dataset_dict.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing E56 smoke prerequisite: $required" >&2
    exit 1
  fi
done

if [[ "$phase" == graph && -e "$GRAPH_IDENTITY" ]]; then
  echo "Fresh E56 graph smoke identity required: $GRAPH_IDENTITY exists" >&2
  exit 1
fi
if [[ "$phase" == python && -e "$PYTHON_IDENTITY" ]]; then
  echo "Fresh E56 Python smoke identity required: $PYTHON_IDENTITY exists" >&2
  exit 1
fi
if [[ "$phase" == sentinel && -e "$SENTINEL_IDENTITY" ]]; then
  echo "Fresh E56 sentinel identity required: $SENTINEL_IDENTITY exists" >&2
  exit 1
fi
if [[ "$phase" == sentinel || "$phase" == sentinel_config ]]; then
  "$PYTHON_BIN" - "$SMOKE_AUDIT" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("status") != "pass" or payload.get("violations"):
    raise SystemExit("E56 sentinel requires a clean terminal smoke audit")
PY
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

freeze_execution() {
  local staging ops_input ops_staging
  SOURCE_HASH="$(hash_tree "$ROOT_DIR/src")"
  SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e56_open_set_split_${SOURCE_HASH}/src"
  if [[ ! -f "$SOURCE_ROOT/oat_drgrpo/__init__.py" ]]; then
    mkdir -p "$(dirname "$SOURCE_ROOT")"
    staging="$(mktemp -d "$(dirname "$SOURCE_ROOT")/.source.XXXXXX")"
    mkdir -p "$staging/src"
    cp -a "$ROOT_DIR/src/." "$staging/src/"
    mv "$staging/src" "$SOURCE_ROOT"
    rmdir "$staging"
  fi
  if [[ "$(hash_tree "$SOURCE_ROOT")" != "$SOURCE_HASH" ]]; then
    echo "E56 source snapshot hash mismatch" >&2
    exit 1
  fi

  ops_input="$(mktemp -d "$ROOT_DIR/var/artifacts/source_snapshots/.e56-ops-input.XXXXXX")"
  mkdir -p "$ops_input/slurm"
  for file in \
    repo_env.sh run_experiment.sh train.sh resolve_eval_cadence.py \
    submit_countdown_comparative.sh; do
    cp "$ROOT_DIR/ops/$file" "$ops_input/$file"
  done
  cp "$ROOT_DIR/ops/slurm/train_node302.slurm" \
    "$ops_input/slurm/train_node302.slurm"
  EXECUTION_HASH="$(hash_tree "$ops_input")"
  OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e56_open_set_split_ops_${EXECUTION_HASH}/ops"
  if [[ ! -f "$OPS_ROOT/submit_countdown_comparative.sh" ]]; then
    mkdir -p "$(dirname "$OPS_ROOT")"
    ops_staging="$(mktemp -d "$(dirname "$OPS_ROOT")/.ops.XXXXXX")"
    mv "$ops_input" "$ops_staging/ops"
    mv "$ops_staging/ops" "$OPS_ROOT"
    rmdir "$ops_staging"
  else
    find "$ops_input" -type f -delete
    rmdir "$ops_input/slurm" "$ops_input"
  fi
  if [[ "$(hash_tree "$OPS_ROOT")" != "$EXECUTION_HASH" ]]; then
    echo "E56 execution snapshot hash mismatch" >&2
    exit 1
  fi
}

write_identity() {
  local path="$1" domain="$2" protocol_hash
  protocol_hash="$(sha256sum "$PROTOCOL" | cut -d' ' -f1)"
  "$PYTHON_BIN" - "$path" "$domain" "$protocol_hash" \
    "$SOURCE_HASH" "$EXECUTION_HASH" <<'PY'
import json
import os
import pathlib
import sys
import tempfile

path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "e56_open_set_split_engineering_smoke_telemetry_v2",
    "domain": sys.argv[2],
    "protocol_sha256": sys.argv[3],
    "source_hash": sys.argv[4],
    "execution_surface_hash": sys.argv[5],
    "seed": 9056,
    "num_samples": 16,
    "information_firewall": {
        "gold_support_feedback": False,
        "evaluation_feedback": False,
        "desired_entropy": None,
        "desired_mode_count": None,
    },
    "controllers": {
        "token_entropy": "unprojected_self_warmup_inverse",
        "open_set_semantic": "unprojected_self_warmup_inverse",
        "verified_mass": "unprojected_self_warmup_surprisal_ratio",
        "known_mode_balance": "unprojected_self_warmup_inverse",
    },
    "replay_objective": "split_mass_balance_per_rollout",
    "replay_measure": (15 / 16) * (1 / 16),
}
path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY
}

write_sentinel_identity() {
  local graph_job="$1" countdown_job="$2" python_job="$3"
  local protocol_hash launcher_hash auditor_hash e53_hash smoke_hash
  local graph_manifest_hash countdown_manifest_hash python_manifest_hash
  local e53_graph_manifest_hash e53_countdown_manifest_hash
  local e53_python_manifest_hash
  protocol_hash="$(sha256sum "$PROTOCOL" | cut -d' ' -f1)"
  launcher_hash="$(sha256sum "$0" | cut -d' ' -f1)"
  auditor_hash="$(sha256sum "$AUDITOR" | cut -d' ' -f1)"
  e53_hash="$(sha256sum "$E53_IDENTITY" | cut -d' ' -f1)"
  smoke_hash="$(sha256sum "$SMOKE_AUDIT" | cut -d' ' -f1)"
  graph_manifest_hash="$(sha256sum "$GRAPH_SENTINEL_MANIFEST" | cut -d' ' -f1)"
  countdown_manifest_hash="$(sha256sum "$COUNTDOWN_SENTINEL_MANIFEST" | cut -d' ' -f1)"
  python_manifest_hash="$(sha256sum "$PYTHON_SENTINEL_MANIFEST" | cut -d' ' -f1)"
  e53_graph_manifest_hash="$(sha256sum "$E53_GRAPH_MANIFEST" | cut -d' ' -f1)"
  e53_countdown_manifest_hash="$(sha256sum "$E53_COUNTDOWN_MANIFEST" | cut -d' ' -f1)"
  e53_python_manifest_hash="$(sha256sum "$E53_PYTHON_MANIFEST" | cut -d' ' -f1)"
  "$PYTHON_BIN" - "$SENTINEL_IDENTITY" "$protocol_hash" \
    "$launcher_hash" "$auditor_hash" "$SOURCE_HASH" "$EXECUTION_HASH" \
    "$e53_hash" "$smoke_hash" \
    "$graph_job" "$countdown_job" "$python_job" \
    "$graph_manifest_hash" "$countdown_manifest_hash" \
    "$python_manifest_hash" "$e53_graph_manifest_hash" \
    "$e53_countdown_manifest_hash" "$e53_python_manifest_hash" <<'PY'
import json
import os
import pathlib
import sys
import tempfile

path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "e56_open_set_split_canonical_05b_sentinel_v1",
    "protocol_sha256": sys.argv[2],
    "launcher_sha256": sys.argv[3],
    "auditor_sha256": sys.argv[4],
    "source_hash": sys.argv[5],
    "execution_surface_hash": sys.argv[6],
    "e53_control_identity_sha256": sys.argv[7],
    "smoke_audit_sha256": sys.argv[8],
    "jobs": {
        "graph_coloring": int(sys.argv[9]),
        "countdown": int(sys.argv[10]),
        "python_factor": int(sys.argv[11]),
    },
    "job_manifest_sha256": {
        "graph_coloring": sys.argv[12],
        "countdown": sys.argv[13],
        "python_factor": sys.argv[14],
    },
    "e53_control_manifest_sha256": {
        "graph_coloring": sys.argv[15],
        "countdown": sys.argv[16],
        "python_factor": sys.argv[17],
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
    "arm": "open_set_split_canonical",
    "seed": 9010,
    "num_samples": 16,
    "information_firewall": {
        "gold_support_feedback": False,
        "evaluation_feedback": False,
        "desired_entropy": None,
        "desired_mode_count": None,
    },
    "controllers": {
        "conditional_token_entropy": {
            "rule": "alpha=alpha0*reference_entropy/entropy_ema",
            "base": 0.000075,
            "warmup_eligible_steps": 64,
            "ema_decay": 0.90,
            "projection": None,
        },
        "open_set_semantic_entropy": {
            "rule": "beta=beta0*reference_entropy/entropy_ema",
            "base": 0.10,
            "warmup_eligible_steps": 64,
            "ema_decay": 0.90,
            "support": "verified_model_modes_plus_one_unseen_bucket",
            "projection": None,
        },
        "verified_mass": {
            "rule": "mu=mu0*surprisal_ema/reference_surprisal",
            "base": 0.10,
            "warmup_eligible_steps": 64,
            "ema_decay": 0.90,
            "projection": None,
        },
        "known_mode_balance": {
            "rule": "alpha=alpha0*reference_entropy/entropy_ema",
            "base": 0.10,
            "warmup_eligible_steps": 64,
            "ema_decay": 0.90,
            "projection": None,
        },
    },
    "replay": {
        "loss": "split_mass_balance_per_rollout",
        "mass_raw_score_gradient_sum": -1.0,
        "balance_raw_score_gradient_sum": 0.0,
        "reward_estimator_scale": 15 / 16,
        "objective_scale": 1 / 16,
        "capacity": 16,
        "gold_support_feedback": False,
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

if [[ "$phase" == config || "$phase" == sentinel_config ]]; then
  SOURCE_HASH="$(hash_tree "$ROOT_DIR/src")"
  EXECUTION_HASH=config-only-current-tree
  SOURCE_ROOT="$ROOT_DIR/src"
  OPS_ROOT="$ROOT_DIR/ops"
else
  freeze_execution
fi

export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_OPS_SNAPSHOT_ROOT="$OPS_ROOT"
export OAT_ZERO_CAMPAIGN_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_PRETRAIN="$MODEL_ROOT"
export OAT_ZERO_REQUIRE_EXISTING_DATA=1
export OAT_ZERO_COMPARATIVE_REBUILD=0
export OAT_ZERO_APPEND_MANIFEST=0
export OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-0.5b-instruct
if [[ "$phase" == sentinel || "$phase" == sentinel_config ]]; then
  export OAT_ZERO_TRAIN_SEEDS=9010
else
  export OAT_ZERO_TRAIN_SEEDS=9056
fi
export OAT_ZERO_ONLY_ARMS=open_set_split_canonical
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
  MAXENT_INVERSE_CANONICAL_REPLAY; do
  export "OAT_ZERO_INCLUDE_${flag}_ARM=0"
done
export OAT_ZERO_INCLUDE_OPEN_SET_SPLIT_CANONICAL_ARM=1

export OAT_ZERO_VERIFIED_DISCOVERY_TRACKING=1
export OAT_ZERO_MAXENT_OBJECTIVE=conditional_token_mean
export OAT_ZERO_MAXENT_ALPHA=0.000075
export OAT_ZERO_MAXENT_INVERSE_BASE_ALPHA=0.000075
export OAT_ZERO_MAXENT_INVERSE_WARMUP_STEPS=64
export OAT_ZERO_MAXENT_INVERSE_EMA_DECAY=0.90
export OAT_ZERO_SEMANTIC_SHANNON_COEF=0.10
export OAT_ZERO_SEMANTIC_SHANNON_SURPRISAL_CLIP=5.0
export OAT_ZERO_SEMANTIC_SHANNON_PSEUDOCOUNT=1.0
export OAT_ZERO_SEMANTIC_SHANNON_OPEN_SET_WARMUP_STEPS=64
export OAT_ZERO_SEMANTIC_SHANNON_OPEN_SET_EMA_DECAY=0.90
export OAT_ZERO_ONLINE_CANONICAL_BANK_ALPHA=0.0
export OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.50
export OAT_ZERO_ONLINE_CANONICAL_BANK_PSEUDOCOUNT=1.0
export OAT_ZERO_ONLINE_CANONICAL_BANK_SURPRISAL_CLIP=5.0
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
export OAT_ZERO_MAX_QUERIES=100000000
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
if [[ "$phase" == sentinel || "$phase" == sentinel_config ]]; then
  export OAT_ZERO_EVAL_MODE_COVERAGE_K=8
  export OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE=1
  export OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=4
  export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=530100
  export OAT_ZERO_EVAL_BATCH_SIZE=64
else
  export OAT_ZERO_EVAL_MODE_COVERAGE_K=0
fi
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
export OAT_ZERO_SAVE_CKPT=0
export OAT_ZERO_AUTO_RESUME=0
export OAT_ZERO_MAX_PROMPT_EPOCHS=1
export OAT_ZERO_NUM_PROMPT_EPOCH=1
export OAT_ZERO_MAX_TRAIN=16
export OAT_ZERO_EVAL_PROMPT_INTERVAL=1000000
export OAT_ZERO_TRAIN_CPUS_PER_TASK=8
export OAT_ZERO_TRAIN_MEMORY=64G
export OAT_ZERO_TRAIN_TIME_LIMIT=04:00:00
export OAT_ZERO_SBATCH_HOLD=0

submit_domain() {
  local domain="$1"
  case "$domain" in
    graph)
      export OAT_ZERO_MAX_TRAIN=16
      export RUN_STAMP_PREFIX="$GRAPH_PREFIX"
      export OAT_ZERO_COMPARATIVE_TASK=graph_coloring
      export OAT_ZERO_COMPARATIVE_DATA_ROOT="$GRAPH_DATA_ROOT"
      export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E56_GRAPH_NODELIST:-node302}"
      export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E56_GRAPH_GRES:-gpu:a100:1}"
      export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E56_GRAPH_PARTITION:-all}"
      export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E56_GRAPH_ACCOUNT:-mltheory}"
      ;;
    python)
      # E55's 16-prompt Python smoke saw no validator-positive rollout. A
      # still-bounded 128-prompt smoke gives this harder domain room for
      # natural on-policy discovery without seeding a gold exemplar.
      export OAT_ZERO_MAX_TRAIN=128
      export RUN_STAMP_PREFIX="$PYTHON_PREFIX"
      export OAT_ZERO_COMPARATIVE_TASK=python_factor
      export OAT_ZERO_COMPARATIVE_DATA_ROOT="$PYTHON_DATA_ROOT"
      export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E56_PYTHON_NODELIST:-node020,node022,node023,node024,node026}"
      export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E56_PYTHON_GRES:-gpu:rtx_3090:1}"
      export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E56_PYTHON_PARTITION:-lowprio}"
      export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E56_PYTHON_ACCOUNT:-allcs}"
      ;;
    *)
      echo "Unknown E56 smoke domain: $domain" >&2
      exit 1
      ;;
  esac
  "$OPS_ROOT/submit_countdown_comparative.sh"
}

submit_sentinel_domain() {
  local domain="$1"
  case "$domain" in
    graph_coloring)
      export RUN_STAMP_PREFIX="$GRAPH_SENTINEL_PREFIX"
      export OAT_ZERO_COMPARATIVE_TASK=graph_coloring
      export OAT_ZERO_COMPARATIVE_DATA_ROOT="$GRAPH_DATA_ROOT"
      export OAT_ZERO_MAX_TRAIN=192
      export OAT_ZERO_EVAL_PROMPT_INTERVAL=48
      export OAT_ZERO_SAVE_STEPS=192
      export OAT_ZERO_SAVE_FROM=192
      export OAT_ZERO_RESUME_STEPS=192
      export OAT_ZERO_RESUME_FROM=192
      export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E56_SENTINEL_GRAPH_NODELIST:-node103,node104,node208}"
      export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E56_SENTINEL_GRAPH_GRES:-gpu:a6000:1}"
      export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E56_SENTINEL_GRAPH_PARTITION:-lowprio}"
      export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E56_SENTINEL_GRAPH_ACCOUNT:-mltheory}"
      ;;
    countdown)
      export RUN_STAMP_PREFIX="$COUNTDOWN_SENTINEL_PREFIX"
      export OAT_ZERO_COMPARATIVE_TASK=countdown
      export OAT_ZERO_COMPARATIVE_DATA_PRESET=easy3
      export OAT_ZERO_COMPARATIVE_DATA_ROOT="$COUNTDOWN_DATA_ROOT"
      export OAT_ZERO_MAX_TRAIN=384
      export OAT_ZERO_EVAL_PROMPT_INTERVAL=96
      export OAT_ZERO_SAVE_STEPS=384
      export OAT_ZERO_SAVE_FROM=384
      export OAT_ZERO_RESUME_STEPS=384
      export OAT_ZERO_RESUME_FROM=384
      export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E56_SENTINEL_COUNTDOWN_NODELIST:-node020,node022,node023,node024,node026}"
      export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E56_SENTINEL_COUNTDOWN_GRES:-gpu:rtx_3090:1}"
      export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E56_SENTINEL_COUNTDOWN_PARTITION:-lowprio}"
      export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E56_SENTINEL_COUNTDOWN_ACCOUNT:-allcs}"
      ;;
    python_factor)
      export RUN_STAMP_PREFIX="$PYTHON_SENTINEL_PREFIX"
      export OAT_ZERO_COMPARATIVE_TASK=python_factor
      export OAT_ZERO_COMPARATIVE_DATA_ROOT="$PYTHON_DATA_ROOT"
      export OAT_ZERO_MAX_TRAIN=384
      export OAT_ZERO_EVAL_PROMPT_INTERVAL=96
      export OAT_ZERO_SAVE_STEPS=384
      export OAT_ZERO_SAVE_FROM=384
      export OAT_ZERO_RESUME_STEPS=384
      export OAT_ZERO_RESUME_FROM=384
      export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E56_SENTINEL_PYTHON_NODELIST:-node020,node022,node023,node024,node026}"
      export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E56_SENTINEL_PYTHON_GRES:-gpu:rtx_3090:1}"
      export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E56_SENTINEL_PYTHON_PARTITION:-lowprio}"
      export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E56_SENTINEL_PYTHON_ACCOUNT:-allcs}"
      ;;
    *)
      echo "Unknown E56 sentinel domain: $domain" >&2
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
  submit_domain graph
  submit_domain python
  echo "[e56] both engineering smoke configurations passed"
  exit 0
fi

if [[ "$phase" == sentinel || "$phase" == sentinel_config ]]; then
  export OAT_ZERO_MAX_PROMPT_EPOCHS=50
  export OAT_ZERO_NUM_PROMPT_EPOCH=50
  export OAT_ZERO_SAVE_CKPT=1
  export OAT_ZERO_MAX_SAVE_NUM=2
  export OAT_ZERO_EXPORT_STEPS=0
  export OAT_ZERO_MAX_EXPORT_NUM=1
  export OAT_ZERO_MAX_RESUME_NUM=2
  export OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=0
  export OAT_ZERO_AUTO_RESUME=1
  export OAT_ZERO_WATCHDOG_REQUEUE=1
  export OAT_ZERO_WATCHDOG_MAX_RESTARTS=6
  if [[ "$phase" == sentinel_config ]]; then
    export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
    export OAT_ZERO_SBATCH_HOLD=0
    submit_sentinel_domain graph_coloring
    submit_sentinel_domain countdown
    submit_sentinel_domain python_factor
    echo "[e56] all three sentinel configurations passed; no jobs submitted"
    exit 0
  fi

  export OAT_ZERO_PROTOCOL_IDENTITY="$SENTINEL_IDENTITY"
  unset OAT_ZERO_COMPARATIVE_CONFIG_ONLY
  export OAT_ZERO_SBATCH_HOLD=1
  COHORT_RELEASED=0
  job_ids=()
  cleanup_partial_cohort() {
    local status="$?"
    trap - EXIT
    if [[ "$COHORT_RELEASED" != "1" && "${#job_ids[@]}" -gt 0 ]]; then
      scancel "${job_ids[@]}" 2>/dev/null || true
      echo "[e56] cancelled incomplete held sentinel: ${job_ids[*]}" >&2
    fi
    exit "$status"
  }
  trap cleanup_partial_cohort EXIT

  submit_sentinel_domain graph_coloring
  submit_sentinel_domain countdown
  submit_sentinel_domain python_factor

  for prefix in \
    "$GRAPH_SENTINEL_PREFIX" \
    "$COUNTDOWN_SENTINEL_PREFIX" \
    "$PYTHON_SENTINEL_PREFIX"; do
    manifest="$ROOT_DIR/var/artifacts/${prefix}_comparative_jobs.tsv"
    mapfile -t task_jobs < <(
      awk -F '\t' 'NR > 1 && $3 ~ /^[0-9]+$/ {print $3}' "$manifest"
    )
    if [[ "${#task_jobs[@]}" -ne 1 ]]; then
      echo "E56 ${prefix} has ${#task_jobs[@]} jobs; expected 1" >&2
      exit 1
    fi
    job_ids+=("${task_jobs[@]}")
  done
  if [[ "${#job_ids[@]}" -ne 3 ]]; then
    echo "E56 held cohort has ${#job_ids[@]} jobs; expected 3" >&2
    exit 1
  fi
  write_sentinel_identity "${job_ids[@]}"
  for job_id in "${job_ids[@]}"; do
    job_record="$(scontrol show job "$job_id" -o)"
    for required in \
      'JobState=PENDING' \
      'Reason=JobHeldUser' \
      'OAT_ZERO_MAX_PROMPT_EPOCHS=50' \
      'OAT_ZERO_NUM_PROMPT_EPOCH=50' \
      'OAT_ZERO_SEED=9010' \
      'OAT_ZERO_VARIANT=open_set_split_canonical' \
      'OAT_ZERO_NUM_SAMPLES=16' \
      'OAT_ZERO_MAXENT_INVERSE_ADAPTATION=1' \
      'OAT_ZERO_SEMANTIC_SHANNON_OPEN_SET_INVERSE_ADAPTATION=1' \
      'OAT_ZERO_ONLINE_CANONICAL_REPLAY_OBJECTIVE=split_mass_balance_per_rollout' \
      'OAT_ZERO_EVAL_MODE_COVERAGE_K=8' \
      'OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=4' \
      "OAT_ZERO_SOURCE_ROOT=${SOURCE_ROOT}" \
      "OAT_ZERO_OPS_SNAPSHOT_ROOT=${OPS_ROOT}" \
      "OAT_ZERO_PROTOCOL_IDENTITY=${SENTINEL_IDENTITY}" \
      'NumCPUs=8' \
      'MinMemoryNode=64G' \
      'TimeLimit=7-00:00:00'; do
      if [[ "$job_record" != *"$required"* ]]; then
        echo "E56 held-job audit failed for $job_id: missing $required" >&2
        exit 1
      fi
    done
  done
  scontrol release "${job_ids[@]}"
  COHORT_RELEASED=1
  trap - EXIT
  echo "[e56] released held sentinel cohort: ${job_ids[*]}"
  echo "[e56] identity=$SENTINEL_IDENTITY"
  exit 0
fi

if [[ "$phase" == graph ]]; then
  write_identity "$GRAPH_IDENTITY" graph_coloring
  export OAT_ZERO_PROTOCOL_IDENTITY="$GRAPH_IDENTITY"
else
  write_identity "$PYTHON_IDENTITY" python_factor
  export OAT_ZERO_PROTOCOL_IDENTITY="$PYTHON_IDENTITY"
fi
submit_domain "$phase"
echo "[e56] submitted $phase engineering smoke"
