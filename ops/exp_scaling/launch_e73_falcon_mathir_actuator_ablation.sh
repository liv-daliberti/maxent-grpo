#!/usr/bin/env bash
# E73 companion: the terminal MathIR separated-support actuator comparison on
# Falcon3-1B-Instruct.
#
# This is the cross-family replication of the manuscript's terminal causal row.
# It submits only the treatment arm, the entropy-gated singleton-escape
# actuator, at seeds 43-45 on MathIR. The paired "actuator off" reference is the
# E73 cohort's own MathIR xGRPO arm at the same three seeds, which runs the
# identical configuration minus the actuator. The contrast therefore changes
# separated replay support and nothing else.
#
# Because the reference is the E73 MathIR arm, this cohort must match it on every
# dimension that is not the actuator, including the two that E73 itself amended:
#   * response budget 128 generate / 128 evaluate, max_model_len 384 (Amendment 2)
#   * A5000 placement (Amendment 3 revision)
# Getting either wrong would confound the causal claim with a decoding or
# hardware difference rather than isolating the actuator.
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

VARIANT=verified_entropy_gated_singleton_escape_canonical
PYTHON_BIN="${OAT_ZERO_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
PROTOCOL="$ROOT_DIR/paper/preregistration/e73_falcon3_1b_cross_family_replication_20260731.md"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--tiiuae--Falcon3-1B-Instruct/snapshots/28ba2251970a01dd1edc7ba7dad2eb71216ccfdf"
MATHIR_DATA="$ROOT_DIR/var/data/mathir_action_menu_v1"
COHORT_IDENTITY="$ROOT_DIR/var/artifacts/e73_falcon3_1b_cross_family_identity.json"
MATHIR_AMENDMENT="$ROOT_DIR/var/artifacts/e73_falcon3_1b_mathir_budget_amendment.json"
IDENTITY="$ROOT_DIR/var/artifacts/e73_falcon_mathir_actuator_identity.json"

PREFIX=mie73_falcon_actuator_12pass
MANIFEST="$ROOT_DIR/var/artifacts/${PREFIX}_comparative_jobs.tsv"

# Inherited from E73 Amendment 2; these must not drift from the reference arm.
GENERATE_MAX_LENGTH=128
MAX_MODEL_LEN=384
GPU_MODEL=a5000
SEEDS=43,44,45

for required in \
  "$PROTOCOL" "$COHORT_IDENTITY" "$MATHIR_AMENDMENT" \
  "$MODEL_ROOT/config.json" \
  "$MATHIR_DATA/train/dataset_dict.json" \
  "$MATHIR_DATA/eval/dataset_dict.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing E73 actuator prerequisite: $required" >&2
    exit 1
  fi
done

# Fail closed unless the reference arm this will be compared against really does
# carry the amended budget. A silent drift here would look like a causal effect.
"$PYTHON_BIN" - "$MATHIR_AMENDMENT" "$GENERATE_MAX_LENGTH" "$MAX_MODEL_LEN" <<'PY'
import json
import pathlib
import sys

amendment = json.loads(pathlib.Path(sys.argv[1]).read_text())
if amendment.get("domain") != "mathir":
    raise SystemExit("expected the MathIR budget amendment record")
if amendment["generate_max_length"] != int(sys.argv[2]):
    raise SystemExit(
        "actuator cohort generate budget disagrees with the reference arm: "
        f"{amendment['generate_max_length']} vs {sys.argv[2]}"
    )
if amendment["max_model_len"] != int(sys.argv[3]):
    raise SystemExit("actuator cohort max_model_len disagrees with the reference")
print("[e73-actuator] reference MathIR budget verified")
PY

if [[ "$phase" == full ]]; then
  for fresh in "$IDENTITY" "$MANIFEST"; do
    if [[ -e "$fresh" ]]; then
      echo "Fresh E73 actuator artifact required; already exists: $fresh" >&2
      exit 1
    fi
  done
fi

# Reuse the cohort's frozen snapshots: the ablation must execute the same code
# as the arm it is compared against.
read -r SOURCE_ROOT OPS_ROOT SOURCE_HASH EXECUTION_HASH < <(
  "$PYTHON_BIN" - "$COHORT_IDENTITY" "$ROOT_DIR" <<'PY'
import json
import pathlib
import sys

identity = json.loads(pathlib.Path(sys.argv[1]).read_text())
root = pathlib.Path(sys.argv[2])
sh = identity["source_hash"]
eh = identity["execution_surface_hash"]
print(
    root / "var/artifacts/source_snapshots" / f"e73_falcon3_1b_{sh}/src",
    root / "var/artifacts/source_snapshots" / f"e73_falcon3_1b_ops_{eh}/ops",
    sh,
    eh,
)
PY
)

for required in "$SOURCE_ROOT/oat_drgrpo/__init__.py" \
  "$OPS_ROOT/submit_countdown_comparative.sh"; do
  if [[ ! -e "$required" ]]; then
    echo "E73 actuator cannot find the cohort's frozen snapshot: $required" >&2
    exit 1
  fi
done

export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_OPS_SNAPSHOT_ROOT="$OPS_ROOT"
export OAT_ZERO_CAMPAIGN_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_PRETRAIN="$MODEL_ROOT"
export OAT_ZERO_REQUIRE_EXISTING_DATA=1
export OAT_ZERO_COMPARATIVE_REBUILD=0
export OAT_ZERO_APPEND_MANIFEST=0
export OAT_ZERO_COMPARATIVE_MODEL=falcon3-1b-instruct
export OAT_ZERO_TRAIN_SEEDS="$SEEDS"
export OAT_ZERO_ONLY_ARMS="$VARIANT"
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
  VERIFIED_FIRST_SPLIT_CANONICAL VERIFIED_FIRST_GLOBAL_REPLAY_CANONICAL \
  VERIFIED_FIRST_BOOTSTRAP_LOCAL_CANONICAL VERIFIED_COUNTERFACTUAL_CANONICAL; do
  export "OAT_ZERO_INCLUDE_${flag}_ARM=0"
done
export OAT_ZERO_INCLUDE_VERIFIED_ENTROPY_GATED_SINGLETON_ESCAPE_CANONICAL_ARM=1

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
export OAT_ZERO_EXPECT_ONLINE_CANONICAL_NOVELTY_BETA=0.50
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
# The actuator itself: a separate support store for counterfactual admissions,
# so proposed off-policy support never enters PPO's on-policy advantage.
export OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SEPARATE_OBJECTIVE_SUPPORT=1
export OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_MAX_ATTEMPTS=3
export OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SAMPLING_TEMPERATURE=1.0

export OAT_ZERO_NUM_SAMPLES=16
export OAT_ZERO_LEARNING_RATE=0.0000002
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
export OAT_ZERO_PROMPT_TEMPLATE=falcon_boxed
export OAT_ZERO_INPUT_KEY=problem
export OAT_ZERO_OUTPUT_KEY=answer
export OAT_ZERO_EVAL_INPUT_KEY=problem
export OAT_ZERO_EVAL_OUTPUT_KEY=answer
export OAT_ZERO_TEST_SPLIT=multi_answer
export OAT_ZERO_VERIFIER_VERSION=fast
export OAT_ZERO_PROMPT_MAX_LENGTH=256
export OAT_ZERO_TEMPERATURE=1
export OAT_ZERO_TOP_P=1
export OAT_ZERO_EVAL_TEMPERATURE=0
export OAT_ZERO_EVAL_MODE_COVERAGE_K=8
export OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE=1
export OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=4
export OAT_ZERO_EVAL_BATCH_SIZE=64
export OAT_ZERO_SYNC_PARAMS_EVERY=1
export OAT_ZERO_CANONICAL_ACTION_TASK=none
export OAT_ZERO_CANONICAL_GRAPH_ACTIONS=0
export OAT_ZERO_ZERO_STAGE=2
export OAT_ZERO_VLLM_GPU_RATIO=0.25
export OAT_ZERO_ENABLE_FLASH_ATTN=0
export OAT_ZERO_ADAM_OFFLOAD=1
export OAT_ZERO_ACTIVATION_OFFLOADING=0
export OAT_ZERO_COLLOCATE=1
export OAT_ZERO_VLLM_SLEEP=1
export VLLM_USE_V1=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OAT_ZERO_SAVE_CKPT=1
export OAT_ZERO_MAX_SAVE_NUM=2
export OAT_ZERO_RESUME_FROM=0
export OAT_ZERO_MAX_RESUME_NUM=2
export OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=0
export OAT_ZERO_AUTO_RESUME=1
export OAT_ZERO_WATCHDOG_REQUEUE=1
export OAT_ZERO_WATCHDOG_MAX_RESTARTS=6
export OAT_ZERO_TRAIN_CPUS_PER_TASK=8
export OAT_ZERO_TRAIN_MEMORY=64G
export OAT_ZERO_TRAIN_TIME_LIMIT=3-00:00:00

export RUN_STAMP_PREFIX="$PREFIX"
export OAT_ZERO_COMPARATIVE_TASK=math
export OAT_ZERO_COMPARATIVE_DATA_ROOT="$MATHIR_DATA"
export OAT_ZERO_MAX_TRAIN=384
export OAT_ZERO_EVAL_PROMPT_INTERVAL=96
export OAT_ZERO_SAVE_STEPS=384
export OAT_ZERO_SAVE_FROM=384
export OAT_ZERO_RESUME_STEPS=384
export OAT_ZERO_GENERATE_MAX_LENGTH="$GENERATE_MAX_LENGTH"
export OAT_ZERO_EVAL_GENERATE_MAX_LENGTH="$GENERATE_MAX_LENGTH"
export OAT_ZERO_MAX_MODEL_LEN="$MAX_MODEL_LEN"
export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=610400
export OAT_ZERO_TRAIN_NODELIST=node202,node203,node204
export OAT_ZERO_TRAIN_GRES="gpu:${GPU_MODEL}:1"
export OAT_ZERO_TRAIN_PARTITION=cs
export OAT_ZERO_TRAIN_ACCOUNT=allcs

if [[ "$phase" == config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  export OAT_ZERO_SBATCH_HOLD=0
  "$OPS_ROOT/submit_countdown_comparative.sh"
  echo "[e73-actuator] MathIR separated-support configuration passed"
  exit 0
fi

export OAT_ZERO_PROTOCOL_IDENTITY="$COHORT_IDENTITY"
export OAT_ZERO_SBATCH_HOLD=1
released=0
job_ids=()
cleanup_held() {
  local status="$?"
  trap - EXIT
  if [[ "$released" != "1" && "${#job_ids[@]}" -gt 0 ]]; then
    scancel "${job_ids[@]}" 2>/dev/null || true
    echo "[e73-actuator] cancelled incomplete held cohort: ${job_ids[*]}" >&2
  fi
  exit "$status"
}
trap cleanup_held EXIT

"$OPS_ROOT/submit_countdown_comparative.sh"

mapfile -t job_ids < <(
  awk -F $'\t' 'NR > 1 && $3 ~ /^[0-9]+$/ {print $3}' "$MANIFEST"
)
if [[ "${#job_ids[@]}" -ne 3 ]]; then
  echo "E73 actuator cohort has ${#job_ids[@]} jobs; expected 3" >&2
  exit 1
fi

for job_id in "${job_ids[@]}"; do
  job_record="$(scontrol show job "$job_id" -o)"
  for required in \
    'JobState=PENDING' \
    'Reason=JobHeldUser' \
    'Partition=cs' \
    'Account=allcs' \
    "OAT_ZERO_VARIANT=${VARIANT}" \
    "OAT_ZERO_GENERATE_MAX_LENGTH=${GENERATE_MAX_LENGTH}" \
    "OAT_ZERO_MAX_MODEL_LEN=${MAX_MODEL_LEN}" \
    'OAT_ZERO_PROMPT_TEMPLATE=falcon_boxed' \
    'OAT_ZERO_MAX_PROMPT_EPOCHS=12' \
    'OAT_ZERO_NUM_SAMPLES=16' \
    'OAT_ZERO_ADAM_OFFLOAD=1' \
    'OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SEPARATE_OBJECTIVE_SUPPORT=1' \
    "OAT_ZERO_SOURCE_ROOT=${SOURCE_ROOT}" \
    "OAT_ZERO_OPS_SNAPSHOT_ROOT=${OPS_ROOT}"; do
    if [[ "$job_record" != *"$required"* ]]; then
      echo "E73 actuator audit failed for $job_id: missing $required" >&2
      exit 1
    fi
  done
  # The reference arm runs on A5000; a different GPU model here would confound
  # the actuator contrast with sampling numerics.
  if [[ "$job_record" != *"gpu:${GPU_MODEL}"* ]]; then
    echo "E73 actuator job $job_id is not pinned to ${GPU_MODEL}" >&2
    exit 1
  fi
done

"$PYTHON_BIN" - "$IDENTITY" "$PROTOCOL" "$0" "$COHORT_IDENTITY" \
  "$MANIFEST" "$SOURCE_HASH" "$EXECUTION_HASH" "$VARIANT" \
  "$GENERATE_MAX_LENGTH" "$MAX_MODEL_LEN" "$GPU_MODEL" <<'PY'
import csv
import hashlib
import json
import os
import pathlib
import sys
import tempfile

def digest(raw):
    return hashlib.sha256(pathlib.Path(raw).read_bytes()).hexdigest()

path = pathlib.Path(sys.argv[1])
manifest = pathlib.Path(sys.argv[5])
rows = list(csv.DictReader(manifest.open(), delimiter="\t"))
cohort = json.loads(pathlib.Path(sys.argv[4]).read_text())
reference = [
    job
    for job in cohort["jobs"]["mathir"]
    if job["arm"] == "verified_first_global_replay_canonical"
    and job["seed"] in (43, 44, 45)
]

payload = {
    "schema": "e73_falcon_mathir_actuator_v1",
    "replicates": "manuscript terminal MathIR separated-support causal rows",
    "model": "Falcon3-1B-Instruct@28ba2251970a01dd1edc7ba7dad2eb71216ccfdf",
    "treatment_arm": sys.argv[8],
    "reference_arm": "verified_first_global_replay_canonical",
    "reference_source": "e73_falcon3_1b_cross_family_v1 mathir jobs",
    "reference_jobs": reference,
    "matched_on": {
        "generate_max_length": int(sys.argv[9]),
        "max_model_len": int(sys.argv[10]),
        "gpu_model": sys.argv[11],
        "prompt_template": "falcon_boxed",
        "seeds": [43, 44, 45],
        "prompt_epochs": 12,
        "num_samples": 16,
    },
    "isolated_component": "separated replay support actuator",
    "protocol_sha256": digest(sys.argv[2]),
    "launcher_sha256": digest(sys.argv[3]),
    "cohort_identity_sha256": digest(sys.argv[4]),
    "manifest_sha256": digest(manifest),
    "source_hash": sys.argv[6],
    "execution_surface_hash": sys.argv[7],
    "code_refrozen": False,
    "jobs": [
        {
            "arm": row["arm"],
            "seed": int(row["seed"]),
            "job_id": int(row["job_id"]),
            "run_stamp": row["run_stamp"],
        }
        for row in rows
    ],
}
path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY

scontrol release "${job_ids[@]}"
released=1
trap - EXIT
echo "[e73-actuator] released 3 separated-support jobs: ${job_ids[*]}"
echo "[e73-actuator] identity=$IDENTITY"
