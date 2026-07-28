#!/usr/bin/env bash
# Configure or atomically submit E69 Gate 3's 20 new seed-44/45 jobs.
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
PARENT_PROTOCOL="$ROOT_DIR/paper/preregistration/e69_verified_route_successor_protocol_20260728.md"
EXEC_PROTOCOL="$ROOT_DIR/paper/preregistration/e69_gate3_confirmatory_execution_20260728.md"
GATE2_IDENTITY="$ROOT_DIR/var/artifacts/e69_gate2_compute_matched_screen_identity.json"
GATE2_AUDIT="$ROOT_DIR/var/artifacts/e69_gate2_compute_matched_screen_audit_latest.json"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
GRAPH_DATA="$ROOT_DIR/var/data/exact_answer_mode_probe"
COUNTDOWN_DATA="$ROOT_DIR/var/data/exact_countdown_easy3_probe"
PYTHON_DATA="$ROOT_DIR/var/data/python_factor_modebench_v1"
MATHIR_DATA="$ROOT_DIR/var/data/mathir_action_menu_v1"
MATH_DATA="$ROOT_DIR/var/data/math12k_384_route_dev128_v1"
IDENTITY="$ROOT_DIR/var/artifacts/e69_gate3_confirmatory_identity.json"

GRAPH_PREFIX=gce69_gate3_confirmatory_05b_6pass
COUNTDOWN_PREFIX=cde69_gate3_confirmatory_05b_6pass
PYTHON_PREFIX=pye69_gate3_confirmatory_05b_6pass
MATHIR_PREFIX=mie69_gate3_confirmatory_05b_6pass
MATH_PREFIX=mde69_gate3_confirmatory_05b_6pass
GRAPH_MANIFEST="$ROOT_DIR/var/artifacts/${GRAPH_PREFIX}_comparative_jobs.tsv"
COUNTDOWN_MANIFEST="$ROOT_DIR/var/artifacts/${COUNTDOWN_PREFIX}_comparative_jobs.tsv"
PYTHON_MANIFEST="$ROOT_DIR/var/artifacts/${PYTHON_PREFIX}_comparative_jobs.tsv"
MATHIR_MANIFEST="$ROOT_DIR/var/artifacts/${MATHIR_PREFIX}_comparative_jobs.tsv"
MATH_MANIFEST="$ROOT_DIR/var/artifacts/${MATH_PREFIX}_comparative_jobs.tsv"
TRANSITION_SLURM="$ROOT_DIR/ops/slurm/e69_gate3_to_gate4.slurm"
TRANSITION_RECORD="$ROOT_DIR/var/artifacts/e69_gate3_to_gate4_transition_job.json"

for required in \
  "$PARENT_PROTOCOL" "$EXEC_PROTOCOL" "$GATE2_IDENTITY" \
  "$TRANSITION_SLURM" \
  "$MODEL_ROOT/config.json" \
  "$GRAPH_DATA/train/dataset_dict.json" \
  "$GRAPH_DATA/eval/dataset_dict.json" \
  "$COUNTDOWN_DATA/train/dataset_dict.json" \
  "$COUNTDOWN_DATA/eval/dataset_dict.json" \
  "$PYTHON_DATA/train/dataset_dict.json" \
  "$PYTHON_DATA/eval/dataset_dict.json" \
  "$MATHIR_DATA/train/dataset_dict.json" \
  "$MATHIR_DATA/eval/dataset_dict.json" \
  "$MATH_DATA/train/dataset_dict.json" \
  "$MATH_DATA/eval/dataset_dict.json" \
  "$MATH_DATA/MATERIALIZATION_MANIFEST.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing E69 Gate 3 prerequisite: $required" >&2
    exit 1
  fi
done

if [[ "$phase" == full ]]; then
  if [[ ! -f "$GATE2_AUDIT" ]]; then
    echo "E69 Gate 2 terminal audit is absent: $GATE2_AUDIT" >&2
    exit 1
  fi
  "$PYTHON_BIN" - "$GATE2_AUDIT" <<'PY'
import json
import pathlib
import sys

audit = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if audit.get("status") != "pass":
    raise SystemExit("E69 Gate 3 requires a passing frozen Gate 2 audit")
if audit.get("summary", {}).get("integrity_violations") != 0:
    raise SystemExit("E69 Gate 3 requires zero Gate 2 integrity violations")
if audit.get("summary", {}).get("terminal_physical_runs") != 18:
    raise SystemExit("E69 Gate 3 requires all 18 Gate 2 runs terminal")
if audit.get("math500_sealed") is not True:
    raise SystemExit("E69 Gate 2 did not preserve the MATH-500 seal")
PY
  for fresh in \
    "$IDENTITY" "$GRAPH_MANIFEST" "$COUNTDOWN_MANIFEST" \
    "$PYTHON_MANIFEST" "$MATHIR_MANIFEST" "$MATH_MANIFEST" \
    "$TRANSITION_RECORD"; do
    if [[ -e "$fresh" ]]; then
      echo "Fresh E69 Gate 3 artifact required; already exists: $fresh" >&2
      exit 1
    fi
  done
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

readarray -t frozen_hashes < <(
  "$PYTHON_BIN" - "$GATE2_IDENTITY" <<'PY'
import json
import pathlib
import sys

identity = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if identity.get("schema") != "e69_gate2_compute_matched_screen_v1":
    raise SystemExit("unexpected E69 Gate 2 identity schema")
print(identity["source_hash"])
print(identity["execution_surface_hash"])
PY
)
SOURCE_HASH="${frozen_hashes[0]}"
EXECUTION_HASH="${frozen_hashes[1]}"
SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e69_gate2_${SOURCE_HASH}/src"
OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e69_gate2_ops_${EXECUTION_HASH}/ops"
if [[ "$(hash_tree "$SOURCE_ROOT")" != "$SOURCE_HASH" ]]; then
  echo "E69 Gate 3 frozen source snapshot hash mismatch" >&2
  exit 1
fi
if [[ "$(hash_tree "$OPS_ROOT")" != "$EXECUTION_HASH" ]]; then
  echo "E69 Gate 3 frozen execution snapshot hash mismatch" >&2
  exit 1
fi

GRAPH_DATA_HASH="$(hash_tree "$GRAPH_DATA")"
COUNTDOWN_DATA_HASH="$(hash_tree "$COUNTDOWN_DATA")"
PYTHON_DATA_HASH="$(hash_tree "$PYTHON_DATA")"
MATHIR_DATA_HASH="$(hash_tree "$MATHIR_DATA")"
MATH_DATA_HASH="$(hash_tree "$MATH_DATA")"

write_identity() {
  export SOURCE_HASH EXECUTION_HASH
  "$PYTHON_BIN" - \
    "$IDENTITY" "$PARENT_PROTOCOL" "$EXEC_PROTOCOL" "$GATE2_IDENTITY" \
    "$GATE2_AUDIT" "$0" \
    "$GRAPH_DATA_HASH" "$COUNTDOWN_DATA_HASH" "$PYTHON_DATA_HASH" \
    "$MATHIR_DATA_HASH" "$MATH_DATA_HASH" \
    "$GRAPH_MANIFEST" "$COUNTDOWN_MANIFEST" "$PYTHON_MANIFEST" \
    "$MATHIR_MANIFEST" "$MATH_MANIFEST" <<'PY'
import csv
import hashlib
import json
import os
import pathlib
import sys
import tempfile

def digest(raw):
    return hashlib.sha256(pathlib.Path(raw).read_bytes()).hexdigest()

identity_path = pathlib.Path(sys.argv[1])
gate2 = json.loads(pathlib.Path(sys.argv[4]).read_text(encoding="utf-8"))
gate2_audit = json.loads(pathlib.Path(sys.argv[5]).read_text(encoding="utf-8"))
if (
    gate2.get("schema") != "e69_gate2_compute_matched_screen_v1"
    or
    gate2_audit.get("status") != "pass"
    or gate2_audit.get("summary", {}).get("integrity_violations") != 0
):
    raise SystemExit("Gate 3 identity requires a clean passing Gate 2 audit")
domains = ("graph_coloring", "countdown", "python_factor", "mathir", "math_dev")
manifests = [pathlib.Path(raw) for raw in sys.argv[12:17]]
jobs = {}
manifest_hashes = {}
for domain, manifest in zip(domains, manifests):
    rows = list(csv.DictReader(manifest.open(), delimiter="\t"))
    if len(rows) != 4:
        raise SystemExit(f"{manifest} has {len(rows)} rows; expected 4")
    successor = (
        "verified_first_global_replay_canonical"
        if domain == "math_dev"
        else "verified_route_successor"
    )
    wanted = {"grpo", successor}
    reused = []
    for row in gate2_audit["physical_runs"]:
        if (
            row["domain"] == domain
            and int(row.get("seed", 43)) == 43
            and row["arm"] in wanted
            and row.get("terminal") is True
        ):
            reused.append(
                {
                    "arm": row["arm"],
                    "seed": 43,
                    "job_id": int(row["job_id"]),
                    "run_stamp": row["run_stamp"],
                }
            )
    if len(reused) != 2:
        raise SystemExit(f"{domain} does not have exactly two reusable seed-43 jobs")
    new = [
        {
            "arm": row["arm"],
            "seed": int(row["seed"]),
            "job_id": int(row["job_id"]),
            "run_stamp": row["run_stamp"],
            "origin": "gate3_new",
        }
        for row in rows
    ]
    combined = [dict(row, origin="gate2_reuse") for row in reused] + new
    observed = {(row["arm"], int(row["seed"])) for row in combined}
    expected = {(arm, seed) for arm in wanted for seed in (43, 44, 45)}
    if observed != expected or len(combined) != 6:
        raise SystemExit(f"{domain} confirmatory cells do not match the frozen grid")
    jobs[domain] = sorted(
        combined,
        key=lambda row: (int(row["seed"]), str(row["arm"])),
    )
    manifest_hashes[domain] = digest(manifest)

payload = {
    "schema": "e69_gate3_confirmatory_v1",
    "parent_protocol_sha256": digest(sys.argv[2]),
    "execution_protocol_sha256": digest(sys.argv[3]),
    "gate2_identity_sha256": digest(sys.argv[4]),
    "gate2_terminal_audit_sha256": digest(sys.argv[5]),
    "launcher_sha256": digest(sys.argv[6]),
    "source_hash": os.environ["SOURCE_HASH"],
    "execution_surface_hash": os.environ["EXECUTION_HASH"],
    "data_tree_sha256": dict(zip(domains, sys.argv[7:12])),
    "new_manifest_sha256": manifest_hashes,
    "jobs": jobs,
    "attempt_selection": "exact_manifest_job_ids_with_exact_gate2_seed43_reuse",
    "model": (
        "Qwen2.5-0.5B-Instruct@"
        "7ae557604adf67be50417f59c2c2f167def9a775"
    ),
    "seeds": [43, 44, 45],
    "new_seeds": [44, 45],
    "passes": 6,
    "num_samples": 16,
    "reused_physical_jobs": 10,
    "new_physical_jobs": 20,
    "confirmatory_physical_jobs": 30,
    "modebench_arms": ["grpo", "verified_route_successor"],
    "math_arms": ["grpo", "verified_first_global_replay_canonical"],
    "fixed_compute": {
        "neutral_groups_per_prompt": 1,
        "proposal_control_groups_per_prompt": 3,
        "rows_per_group": 16,
        "sampled_rows_per_prompt": 64,
        "replay_groups_per_update": 1,
        "replay_capacity": 16,
        "replay_score_passes": 2,
        "drgrpo_replay_gradient": 0,
    },
    "reporting_passes": [0, 1, 2, 3, 4, 5, 6],
    "checkpoint_selection": "terminal_pass_6_only",
    "bootstrap": {"replicates": 10000, "seed": 690301},
    "math500_sealed": True,
}
identity_path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(
    prefix=f".{identity_path.name}.",
    dir=identity_path.parent,
)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, identity_path)
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
export OAT_ZERO_TRAIN_SEEDS=44,45
export OAT_ZERO_XDR_TAUS=""
export OAT_ZERO_DRGRPO_VARIANT=grpo_compute_matched

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
  VERIFIED_COUNTERFACTUAL_CANONICAL VERIFIED_FIRST_GLOBAL_REPLAY_CANONICAL \
  VERIFIED_ENTROPY_GATED_SINGLETON_ESCAPE_CANONICAL \
  VERIFIED_ROUTE_SUCCESSOR; do
  export "OAT_ZERO_INCLUDE_${flag}_ARM=0"
done

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
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_ALPHA=0.10
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_CAPACITY=16
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=1
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_BOOTSTRAP_STEPS=0
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_WARMUP_STEPS=64
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_EMA_DECAY=0.90
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_ALPHA=0.10
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_WARMUP_STEPS=64
export OAT_ZERO_ONLINE_CANONICAL_REPLAY_MASS_EMA_DECAY=0.90
export OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_FIXED_CONTROL_GROUPS=3
export OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_MAX_ATTEMPTS=3
export OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SAMPLING_TEMPERATURE=1.0
export OAT_ZERO_VERIFIED_ROUTE_REPLAY_CAPACITY_PER_ROUTE=16
export OAT_ZERO_VERIFIED_ROUTE_RECURRING_MIN_NEUTRAL_PROMPTS=2
export OAT_ZERO_VERIFIED_ROUTE_PROPOSAL_MAX_MEAN_LOGPROB_DROP=2.0

export OAT_ZERO_NUM_SAMPLES=16
export OAT_ZERO_LEARNING_RATE=0.0000002
export OAT_ZERO_MAX_QUERIES=100000000
export OAT_ZERO_MAX_PROMPT_EPOCHS=6
export OAT_ZERO_NUM_PROMPT_EPOCH=6
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
export OAT_ZERO_INPUT_KEY=problem
export OAT_ZERO_OUTPUT_KEY=answer
export OAT_ZERO_EVAL_INPUT_KEY=problem
export OAT_ZERO_EVAL_OUTPUT_KEY=answer
export OAT_ZERO_TEMPERATURE=1
export OAT_ZERO_TOP_P=1
export OAT_ZERO_EVAL_TEMPERATURE=0
export OAT_ZERO_EVAL_MODE_COVERAGE_K=8
export OAT_ZERO_EVAL_MODE_COVERAGE_TEMPERATURE=1
export OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=1
export OAT_ZERO_EVAL_BATCH_SIZE=64
export OAT_ZERO_ALLOW_SPARSE_EVAL=1
export OAT_ZERO_SYNC_PARAMS_EVERY=1
export OAT_ZERO_CANONICAL_ACTION_TASK=none
export OAT_ZERO_CANONICAL_GRAPH_ACTIONS=0
export OAT_ZERO_REPLICATED_FREEFORM_SAMPLING=1
export OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC=1
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
export OAT_ZERO_MAX_RESUME_NUM=2
export OAT_ZERO_PRUNE_RESUME_ON_SUCCESS=0
export OAT_ZERO_AUTO_RESUME=1
export OAT_ZERO_WATCHDOG_REQUEUE=1
export OAT_ZERO_WATCHDOG_MAX_RESTARTS=8
export OAT_ZERO_TRAIN_CPUS_PER_TASK=8
export OAT_ZERO_TRAIN_MEMORY=64G

submit_domain() {
  local domain="$1"
  export OAT_ZERO_SEMANTIC_SHANNON_COEF=0.10
  export OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0.50
  export OAT_ZERO_ONLINE_CANONICAL_KEY_MODE=modebench_outcome
  export OAT_ZERO_ONLY_ARMS="grpo,verified_route_successor"
  export OAT_ZERO_INCLUDE_VERIFIED_FIRST_GLOBAL_REPLAY_CANONICAL_ARM=0
  export OAT_ZERO_INCLUDE_VERIFIED_ROUTE_SUCCESSOR_ARM=1
  export OAT_ZERO_PROMPT_TEMPLATE=qwen_boxed
  export OAT_ZERO_TEST_SPLIT=multi_answer
  export OAT_ZERO_VERIFIER_VERSION=fast
  export OAT_ZERO_PROMPT_MAX_LENGTH=256
  export OAT_ZERO_EVAL_MODE_COVERAGE_DRAWS=1
  export OAT_ZERO_TRAIN_TIME_LIMIT=3-00:00:00
  case "$domain" in
    graph_coloring)
      export RUN_STAMP_PREFIX="$GRAPH_PREFIX"
      export OAT_ZERO_COMPARATIVE_TASK=graph_coloring
      export OAT_ZERO_COMPARATIVE_DATA_ROOT="$GRAPH_DATA"
      export OAT_ZERO_MAX_TRAIN=192
      export OAT_ZERO_EVAL_PROMPT_INTERVAL=192
      export OAT_ZERO_SAVE_STEPS=192
      export OAT_ZERO_SAVE_FROM=192
      export OAT_ZERO_RESUME_STEPS=192
      export OAT_ZERO_RESUME_FROM=192
      export OAT_ZERO_GENERATE_MAX_LENGTH=192
      export OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=192
      export OAT_ZERO_MAX_MODEL_LEN=512
      export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=690201
      export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E69_GRAPH_NODELIST:-node105,node202,node203,node204}"
      export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E69_GRAPH_GRES:-gpu:a5000:1}"
      export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E69_GRAPH_PARTITION:-lowprio}"
      export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E69_GRAPH_ACCOUNT:-mltheory}"
      ;;
    countdown)
      export RUN_STAMP_PREFIX="$COUNTDOWN_PREFIX"
      export OAT_ZERO_COMPARATIVE_TASK=countdown
      export OAT_ZERO_COMPARATIVE_DATA_PRESET=easy3
      export OAT_ZERO_COMPARATIVE_DATA_ROOT="$COUNTDOWN_DATA"
      export OAT_ZERO_MAX_TRAIN=384
      export OAT_ZERO_EVAL_PROMPT_INTERVAL=384
      export OAT_ZERO_SAVE_STEPS=384
      export OAT_ZERO_SAVE_FROM=384
      export OAT_ZERO_RESUME_STEPS=384
      export OAT_ZERO_RESUME_FROM=384
      export OAT_ZERO_GENERATE_MAX_LENGTH=192
      export OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=192
      export OAT_ZERO_MAX_MODEL_LEN=512
      export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=690202
      export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E69_A5000_NODELIST:-node105,node202,node203,node204}"
      export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E69_A5000_GRES:-gpu:a5000:1}"
      export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E69_A5000_PARTITION:-lowprio}"
      export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E69_3090_ACCOUNT:-mltheory}"
      ;;
    python_factor)
      export RUN_STAMP_PREFIX="$PYTHON_PREFIX"
      export OAT_ZERO_COMPARATIVE_TASK=python_factor
      export OAT_ZERO_COMPARATIVE_DATA_ROOT="$PYTHON_DATA"
      export OAT_ZERO_MAX_TRAIN=384
      export OAT_ZERO_EVAL_PROMPT_INTERVAL=384
      export OAT_ZERO_SAVE_STEPS=384
      export OAT_ZERO_SAVE_FROM=384
      export OAT_ZERO_RESUME_STEPS=384
      export OAT_ZERO_RESUME_FROM=384
      export OAT_ZERO_GENERATE_MAX_LENGTH=192
      export OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=192
      export OAT_ZERO_MAX_MODEL_LEN=512
      export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=690203
      export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E69_A5000_NODELIST:-node105,node202,node203,node204}"
      export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E69_A5000_GRES:-gpu:a5000:1}"
      export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E69_A5000_PARTITION:-lowprio}"
      export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E69_3090_ACCOUNT:-mltheory}"
      ;;
    mathir)
      export RUN_STAMP_PREFIX="$MATHIR_PREFIX"
      export OAT_ZERO_COMPARATIVE_TASK=math
      export OAT_ZERO_COMPARATIVE_DATA_ROOT="$MATHIR_DATA"
      export OAT_ZERO_MAX_TRAIN=384
      export OAT_ZERO_EVAL_PROMPT_INTERVAL=384
      export OAT_ZERO_SAVE_STEPS=384
      export OAT_ZERO_SAVE_FROM=384
      export OAT_ZERO_RESUME_STEPS=384
      export OAT_ZERO_RESUME_FROM=384
      export OAT_ZERO_GENERATE_MAX_LENGTH=64
      export OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=64
      export OAT_ZERO_MAX_MODEL_LEN=384
      export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=690204
      export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E69_A100_NODELIST:-node302}"
      export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E69_A100_GRES:-gpu:a100:1}"
      export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E69_A100_PARTITION:-lowprio}"
      export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E69_A100_ACCOUNT:-mltheory}"
      ;;
    math_dev)
      export RUN_STAMP_PREFIX="$MATH_PREFIX"
      export OAT_ZERO_COMPARATIVE_TASK=math
      export OAT_ZERO_COMPARATIVE_DATA_ROOT="$MATH_DATA"
      export OAT_ZERO_ONLY_ARMS="grpo,verified_first_global_replay_canonical"
      export OAT_ZERO_INCLUDE_VERIFIED_FIRST_GLOBAL_REPLAY_CANONICAL_ARM=1
      export OAT_ZERO_INCLUDE_VERIFIED_ROUTE_SUCCESSOR_ARM=0
      # Inherit the recorded Gate 2 startup repair: the E66 endpoint variant
      # enables separated semantic advantage and therefore requires its
      # prospectively defined positive coefficient.
      export OAT_ZERO_SEMANTIC_SHANNON_COEF=0.10
      export OAT_ZERO_ONLINE_CANONICAL_NOVELTY_BETA=0
      export OAT_ZERO_ONLINE_CANONICAL_KEY_MODE=math_verified_answer
      export OAT_ZERO_PROMPT_TEMPLATE=qwen_math
      export OAT_ZERO_TEST_SPLIT=math
      export OAT_ZERO_VERIFIER_VERSION=math_verify
      export OAT_ZERO_MAX_TRAIN=384
      export OAT_ZERO_EVAL_PROMPT_INTERVAL=384
      export OAT_ZERO_SAVE_STEPS=384
      export OAT_ZERO_SAVE_FROM=384
      export OAT_ZERO_RESUME_STEPS=384
      export OAT_ZERO_RESUME_FROM=384
      export OAT_ZERO_PROMPT_MAX_LENGTH=1024
      export OAT_ZERO_GENERATE_MAX_LENGTH=1024
      export OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=1024
      export OAT_ZERO_MAX_MODEL_LEN=2048
      export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=690205
      export OAT_ZERO_TRAIN_TIME_LIMIT=7-00:00:00
      export OAT_ZERO_TRAIN_NODELIST="${OAT_ZERO_E69_A100_NODELIST:-node302}"
      export OAT_ZERO_TRAIN_GRES="${OAT_ZERO_E69_A100_GRES:-gpu:a100:1}"
      export OAT_ZERO_TRAIN_PARTITION="${OAT_ZERO_E69_A100_PARTITION:-lowprio}"
      export OAT_ZERO_TRAIN_ACCOUNT="${OAT_ZERO_E69_A100_ACCOUNT:-mltheory}"
      ;;
    *)
      echo "Unknown E69 Gate 3 domain: $domain" >&2
      exit 1
      ;;
  esac
  "$OPS_ROOT/submit_countdown_comparative.sh"
}

if [[ "$phase" == config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  export OAT_ZERO_SBATCH_HOLD=0
  for domain in graph_coloring countdown python_factor mathir math_dev; do
    submit_domain "$domain"
  done
  echo "[e69-gate3] all five domain configurations passed"
  exit 0
fi

export OAT_ZERO_PROTOCOL_IDENTITY="$IDENTITY"
export OAT_ZERO_SBATCH_HOLD=1
released=0
job_ids=()
transition_job_id=""
cleanup_held() {
  local status="$?"
  trap - EXIT
  if [[ "$released" != "1" ]]; then
    if [[ -n "$transition_job_id" ]]; then
      scancel "$transition_job_id" 2>/dev/null || true
      echo "[e69-gate3] cancelled incomplete transition: $transition_job_id" >&2
    fi
    if [[ "${#job_ids[@]}" -gt 0 ]]; then
      scancel "${job_ids[@]}" 2>/dev/null || true
      echo "[e69-gate3] cancelled incomplete held cohort: ${job_ids[*]}" >&2
    fi
  fi
  exit "$status"
}
trap cleanup_held EXIT

for domain in graph_coloring countdown python_factor mathir math_dev; do
  submit_domain "$domain"
done

for manifest in \
  "$GRAPH_MANIFEST" "$COUNTDOWN_MANIFEST" "$PYTHON_MANIFEST" \
  "$MATHIR_MANIFEST" "$MATH_MANIFEST"; do
  mapfile -t manifest_jobs < <(
    awk -F $'\t' 'NR > 1 && $3 ~ /^[0-9]+$/ {print $3}' "$manifest"
  )
  if [[ "${#manifest_jobs[@]}" -ne 4 ]]; then
    echo "E69 Gate 3 manifest $manifest has ${#manifest_jobs[@]} jobs; expected 4" >&2
    exit 1
  fi
  job_ids+=("${manifest_jobs[@]}")
done
if [[ "${#job_ids[@]}" -ne 20 ]]; then
  echo "E69 Gate 3 new cohort has ${#job_ids[@]} jobs; expected 20" >&2
  exit 1
fi

write_identity

for job_id in "${job_ids[@]}"; do
  record="$(scontrol show job "$job_id" -o)"
  for required in \
    'JobState=PENDING' 'Reason=JobHeldUser' \
    'OAT_ZERO_MAX_PROMPT_EPOCHS=6' \
    'OAT_ZERO_NUM_PROMPT_EPOCH=6' \
    'OAT_ZERO_NUM_SAMPLES=16' \
    'OAT_ZERO_REPLICATED_FREEFORM_SAMPLING=1' \
    'OAT_ZERO_LOCAL_ACTOR_WEIGHT_SYNC=1' \
    'OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_FIXED_CONTROL_GROUPS=3' \
    'OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=1' \
    "OAT_ZERO_SOURCE_ROOT=${SOURCE_ROOT}" \
    "OAT_ZERO_OPS_SNAPSHOT_ROOT=${OPS_ROOT}" \
    "OAT_ZERO_PROTOCOL_IDENTITY=${IDENTITY}"; do
    if [[ "$record" != *"$required"* ]]; then
      echo "E69 Gate 3 held-job audit failed for $job_id: missing $required" >&2
      exit 1
    fi
  done
  if [[ "$record" == *'OAT_ZERO_VARIANT=grpo_compute_matched'* ]]; then
    for required in \
      'OAT_ZERO_ONLINE_CANONICAL_REPLAY=1' \
      'OAT_ZERO_ONLINE_CANONICAL_REPLAY_COMPUTE_ONLY=1' \
      'OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS=0'; do
      [[ "$record" == *"$required"* ]] || {
        echo "E69 Gate 3 Dr.GRPO audit failed for $job_id: missing $required" >&2
        exit 1
      }
    done
  elif [[ "$record" == *'OAT_ZERO_VARIANT=verified_route_successor'* ]]; then
    for required in \
      'OAT_ZERO_ONLINE_CANONICAL_KEY_MODE=verified_route' \
      'OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_PROPOSALS=1' \
      'OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_SEPARATE_OBJECTIVE_SUPPORT=1'; do
      [[ "$record" == *"$required"* ]] || {
        echo "E69 Gate 3 route audit failed for $job_id: missing $required" >&2
        exit 1
      }
    done
  fi
done

original_ifs="$IFS"
IFS=:
dependency="${job_ids[*]}"
IFS="$original_ifs"
mkdir -p "$ROOT_DIR/var/artifacts/logs"
transition_job_id="$(
  sbatch \
    --parsable \
    --dependency="afterany:${dependency}" \
    --output="$ROOT_DIR/var/artifacts/logs/e69_gate3_to_gate4-%j.out" \
    --error="$ROOT_DIR/var/artifacts/logs/e69_gate3_to_gate4-%j.err" \
    "$TRANSITION_SLURM"
)"
transition_job_id="${transition_job_id%%;*}"
if [[ ! "$transition_job_id" =~ ^[0-9]+$ ]]; then
  echo "Invalid E69 Gate 3 transition job id: $transition_job_id" >&2
  exit 1
fi
transition_record="$(scontrol show job "$transition_job_id" -o)"
for required in 'JobState=PENDING' 'Reason=Dependency' 'Dependency=afterany:'; do
  if [[ "$transition_record" != *"$required"* ]]; then
    echo \
      "E69 Gate 3 transition audit failed for $transition_job_id: missing $required" \
      >&2
    exit 1
  fi
done
for job_id in "${job_ids[@]}"; do
  if [[ "$transition_record" != *"$job_id"* ]]; then
    echo \
      "E69 Gate 3 transition $transition_job_id is missing dependency $job_id" \
      >&2
    exit 1
  fi
done
"$PYTHON_BIN" - "$TRANSITION_RECORD" "$transition_job_id" "${job_ids[@]}" <<'PY'
import json
import os
import pathlib
import sys
import tempfile

path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "e69_gate3_to_gate4_transition_job_v1",
    "job_id": int(sys.argv[2]),
    "dependency_kind": "afterany",
    "gate3_new_job_ids": [int(raw) for raw in sys.argv[3:]],
    "fail_closed": True,
}
descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY

scontrol release "${job_ids[@]}"
released=1
trap - EXIT
echo "[e69-gate3] released 20 new seed-44/45 jobs: ${job_ids[*]}"
echo "[e69-gate3] fail-closed Gate 3 -> Gate 4 transition: $transition_job_id"
echo "[e69-gate3] identity=$IDENTITY"
