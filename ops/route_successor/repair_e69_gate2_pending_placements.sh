#!/usr/bin/env bash
# Atomically replace Gate 2's 12 never-started executable-domain jobs.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/ops/repo_env.sh"

phase="${1:-}"
case "$phase" in
  config|full|graph-a5000-config|graph-a5000-full) ;;
  *)
    echo "Usage: $0 {config|full|graph-a5000-config|graph-a5000-full}" >&2
    exit 1
    ;;
esac
GRAPH_A5000_REPAIR=0
if [[ "$phase" == graph-a5000-* ]]; then
  GRAPH_A5000_REPAIR=1
fi

PYTHON_BIN="${OAT_ZERO_PYTHON:-$ROOT_DIR/var/seed_paper_eval/paper310/bin/python}"
ORIGINAL_IDENTITY="$ROOT_DIR/var/artifacts/e69_gate2_compute_matched_screen_identity.json"
AMENDMENT="$ROOT_DIR/paper/preregistration/e69_gate2_pending_placement_repair_20260728.md"
MODEL_ROOT="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
GRAPH_DATA="$ROOT_DIR/var/data/exact_answer_mode_probe"
COUNTDOWN_DATA="$ROOT_DIR/var/data/exact_countdown_easy3_probe"
PYTHON_DATA="$ROOT_DIR/var/data/python_factor_modebench_v1"
REPAIR_IDENTITY="$ROOT_DIR/var/artifacts/e69_gate2_pending_placement_repair_identity.json"
GRAPH_PREFIX=gce69_gate2_placement_repair
GRAPH_A5000_AMENDMENT="$ROOT_DIR/paper/preregistration/e69_gate2_graph_a5000_preemption_repair_20260728.md"
GRAPH_A5000_REPAIR_IDENTITY="$ROOT_DIR/var/artifacts/e69_gate2_graph_a5000_preemption_repair_identity.json"
GRAPH_A5000_ORIGINAL_JOBS=(30160181 30160182 30160183 30160184)
if [[ "$GRAPH_A5000_REPAIR" == "1" ]]; then
  GRAPH_PREFIX=gce69_gate2_graph_a5000_repair
fi
COUNTDOWN_PREFIX=cde69_gate2_placement_repair
PYTHON_PREFIX=pye69_gate2_placement_repair
GRAPH_MANIFEST="$ROOT_DIR/var/artifacts/${GRAPH_PREFIX}_comparative_jobs.tsv"
COUNTDOWN_MANIFEST="$ROOT_DIR/var/artifacts/${COUNTDOWN_PREFIX}_comparative_jobs.tsv"
PYTHON_MANIFEST="$ROOT_DIR/var/artifacts/${PYTHON_PREFIX}_comparative_jobs.tsv"
ORIGINAL_JOBS=(
  30159713 30159714 30159715 30159716
  30159717 30159718 30159719 30159720
  30159721 30159722 30159723 30159724
)

for required in \
  "$ORIGINAL_IDENTITY" "$AMENDMENT" "$MODEL_ROOT/config.json" \
  "$GRAPH_DATA/train/dataset_dict.json" "$GRAPH_DATA/eval/dataset_dict.json" \
  "$COUNTDOWN_DATA/train/dataset_dict.json" "$COUNTDOWN_DATA/eval/dataset_dict.json" \
  "$PYTHON_DATA/train/dataset_dict.json" "$PYTHON_DATA/eval/dataset_dict.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing E69 placement-repair prerequisite: $required" >&2
    exit 1
  fi
done
if [[ "$GRAPH_A5000_REPAIR" == "1" ]]; then
  for required in "$GRAPH_A5000_AMENDMENT" "$REPAIR_IDENTITY"; do
    if [[ ! -e "$required" ]]; then
      echo "Missing E69 Graph A5000 repair prerequisite: $required" >&2
      exit 1
    fi
  done
fi

readarray -t frozen_hashes < <(
  "$PYTHON_BIN" - "$ORIGINAL_IDENTITY" <<'PY'
import json
import pathlib
import sys

identity = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if identity.get("schema") != "e69_gate2_compute_matched_screen_v1":
    raise SystemExit("unexpected E69 Gate 2 identity schema")
expected = set(range(30159713, 30159725))
observed = {
    int(row["job_id"])
    for domain in ("graph_coloring", "countdown", "python_factor")
    for row in identity["jobs"][domain]
}
if observed != expected:
    raise SystemExit("original Gate 2 executable pending cohort mismatch")
print(identity["source_hash"])
print(identity["execution_surface_hash"])
PY
)
SOURCE_HASH="${frozen_hashes[0]}"
EXECUTION_HASH="${frozen_hashes[1]}"
SOURCE_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e69_gate2_${SOURCE_HASH}/src"
OPS_ROOT="$ROOT_DIR/var/artifacts/source_snapshots/e69_gate2_ops_${EXECUTION_HASH}/ops"

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
if [[ "$(hash_tree "$SOURCE_ROOT")" != "$SOURCE_HASH" ]]; then
  echo "E69 placement-repair source snapshot mismatch" >&2
  exit 1
fi
if [[ "$(hash_tree "$OPS_ROOT")" != "$EXECUTION_HASH" ]]; then
  echo "E69 placement-repair execution snapshot mismatch" >&2
  exit 1
fi

if [[ "$phase" == full ]]; then
  for fresh in \
    "$REPAIR_IDENTITY" "$GRAPH_MANIFEST" "$COUNTDOWN_MANIFEST" \
    "$PYTHON_MANIFEST"; do
    if [[ -e "$fresh" ]]; then
      echo "Fresh E69 placement-repair artifact required: $fresh" >&2
      exit 1
    fi
  done
  for job_id in "${ORIGINAL_JOBS[@]}"; do
    record="$(scontrol show job "$job_id" -o)"
    if [[ "$record" != *'JobState=PENDING'* \
        || "$record" != *'RunTime=00:00:00'* ]]; then
      echo "Original placement-repair job is not never-started pending: $job_id" >&2
      exit 1
    fi
    if find "$ROOT_DIR/var/data" -maxdepth 2 -type d \
        -path "*job${job_id}" -print -quit | grep -q .; then
      echo "Original pending job unexpectedly has a run directory: $job_id" >&2
      exit 1
    fi
  done
fi
if [[ "$phase" == graph-a5000-full ]]; then
  for fresh in "$GRAPH_A5000_REPAIR_IDENTITY" "$GRAPH_MANIFEST"; do
    if [[ -e "$fresh" ]]; then
      echo "Fresh E69 Graph A5000 repair artifact required: $fresh" >&2
      exit 1
    fi
  done
  for job_id in "${GRAPH_A5000_ORIGINAL_JOBS[@]}"; do
    record="$(scontrol show job "$job_id" -o)"
    if [[ "$record" != *'JobState=PENDING'* \
        && "$record" != *'JobState=RUNNING'* ]]; then
      echo "Graph repair source job is not pending/running: $job_id" >&2
      exit 1
    fi
    run_dir="$(
      find "$ROOT_DIR/var/data" -maxdepth 2 -type d \
        -path "*job${job_id}" -print -quit
    )"
    if [[ -z "$run_dir" || -d "$run_dir/saved_models" ]]; then
      echo "Graph repair source job is missing or terminal: $job_id" >&2
      exit 1
    fi
  done
fi

export OAT_ZERO_REPO_ROOT="$ROOT_DIR"
export OAT_ZERO_OPS_SNAPSHOT_ROOT="$OPS_ROOT"
export OAT_ZERO_CAMPAIGN_SOURCE_ROOT="$SOURCE_ROOT"
export OAT_ZERO_PRETRAIN="$MODEL_ROOT"
export OAT_ZERO_REQUIRE_EXISTING_DATA=1
export OAT_ZERO_COMPARATIVE_REBUILD=0
export OAT_ZERO_APPEND_MANIFEST=0
export OAT_ZERO_COMPARATIVE_MODEL=qwen2.5-0.5b-instruct
export OAT_ZERO_TRAIN_SEEDS=43
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
export OAT_ZERO_INCLUDE_VERIFIED_FIRST_GLOBAL_REPLAY_CANONICAL_ARM=1
export OAT_ZERO_INCLUDE_VERIFIED_ENTROPY_GATED_SINGLETON_ESCAPE_CANONICAL_ARM=1
export OAT_ZERO_INCLUDE_VERIFIED_ROUTE_SUCCESSOR_ARM=1
export OAT_ZERO_ONLY_ARMS="grpo,verified_first_global_replay_canonical,verified_entropy_gated_singleton_escape_canonical,verified_route_successor"

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
export OAT_ZERO_TRAIN_TIME_LIMIT=3-00:00:00
export OAT_ZERO_PROMPT_TEMPLATE=qwen_boxed
export OAT_ZERO_TEST_SPLIT=multi_answer
export OAT_ZERO_VERIFIER_VERSION=fast
export OAT_ZERO_PROMPT_MAX_LENGTH=256
export OAT_ZERO_GENERATE_MAX_LENGTH=192
export OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=192
export OAT_ZERO_MAX_MODEL_LEN=512

submit_domain() {
  local domain="$1"
  export OAT_ZERO_ONLINE_CANONICAL_KEY_MODE=modebench_outcome
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
      export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=690201
      if [[ "$GRAPH_A5000_REPAIR" == "1" ]]; then
        export OAT_ZERO_TRAIN_NODELIST=node202
        export OAT_ZERO_TRAIN_GRES=gpu:a5000:1
      else
        export OAT_ZERO_TRAIN_NODELIST=node101
        export OAT_ZERO_TRAIN_GRES=gpu:a40:1
      fi
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
      export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=690202
      export OAT_ZERO_TRAIN_NODELIST=node105
      export OAT_ZERO_TRAIN_GRES=gpu:a5000:1
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
      export OAT_ZERO_EVAL_MODE_COVERAGE_SEED=690203
      export OAT_ZERO_TRAIN_NODELIST=node105
      export OAT_ZERO_TRAIN_GRES=gpu:a5000:1
      ;;
    *)
      echo "Unknown placement-repair domain: $domain" >&2
      exit 1
      ;;
  esac
  export OAT_ZERO_TRAIN_PARTITION=lowprio
  export OAT_ZERO_TRAIN_ACCOUNT=mltheory
  "$OPS_ROOT/submit_countdown_comparative.sh"
}

if [[ "$phase" == config || "$phase" == graph-a5000-config ]]; then
  export OAT_ZERO_COMPARATIVE_CONFIG_ONLY=1
  export OAT_ZERO_SBATCH_HOLD=0
  if [[ "$GRAPH_A5000_REPAIR" == "1" ]]; then
    submit_domain graph_coloring
    echo "[e69-graph-a5000-repair] configuration passed"
  else
    for domain in graph_coloring countdown python_factor; do
      submit_domain "$domain"
    done
    echo "[e69-placement-repair] all three configurations passed"
  fi
  exit 0
fi

if [[ "$phase" == graph-a5000-full ]]; then
  export OAT_ZERO_PROTOCOL_IDENTITY="$ORIGINAL_IDENTITY"
  export OAT_ZERO_SBATCH_HOLD=1
  graph_replacements=()
  graph_released=0
  graph_originals_cancelled=0
  cleanup_graph_held() {
    local status="$?"
    trap - EXIT
    if [[ "$graph_released" != "1" && "$graph_originals_cancelled" != "1" \
        && "${#graph_replacements[@]}" -gt 0 ]]; then
      scancel "${graph_replacements[@]}" 2>/dev/null || true
      echo "[e69-graph-a5000-repair] cancelled incomplete held replacements" >&2
    elif [[ "$graph_released" != "1" \
        && "$graph_originals_cancelled" == "1" ]]; then
      echo "[e69-graph-a5000-repair] originals cancelled; replacements remain held" >&2
    fi
    exit "$status"
  }
  trap cleanup_graph_held EXIT

  submit_domain graph_coloring
  mapfile -t graph_replacements < <(
    awk -F $'\t' 'NR > 1 && $3 ~ /^[0-9]+$/ {print $3}' "$GRAPH_MANIFEST"
  )
  if [[ "${#graph_replacements[@]}" -ne 4 ]]; then
    echo "Graph A5000 repair manifest must contain four jobs" >&2
    exit 1
  fi
  for job_id in "${graph_replacements[@]}"; do
    record="$(scontrol show job "$job_id" -o)"
    for required in \
      'JobState=PENDING' 'Reason=JobHeldUser' \
      'ReqNodeList=node202' 'gres/gpu:a5000=1' \
      'OAT_ZERO_MAX_PROMPT_EPOCHS=6' \
      'OAT_ZERO_NUM_SAMPLES=16' \
      'OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_FIXED_CONTROL_GROUPS=3' \
      'OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=1' \
      "OAT_ZERO_SOURCE_ROOT=${SOURCE_ROOT}" \
      "OAT_ZERO_OPS_SNAPSHOT_ROOT=${OPS_ROOT}" \
      "OAT_ZERO_PROTOCOL_IDENTITY=${ORIGINAL_IDENTITY}"; do
      if [[ "$record" != *"$required"* ]]; then
        echo "Graph A5000 held-job audit $job_id missing: $required" >&2
        exit 1
      fi
    done
  done

  export SOURCE_HASH EXECUTION_HASH
  "$PYTHON_BIN" - \
    "$GRAPH_A5000_REPAIR_IDENTITY" "$ORIGINAL_IDENTITY" "$REPAIR_IDENTITY" \
    "$GRAPH_A5000_AMENDMENT" "$0" "$GRAPH_MANIFEST" <<'PY'
import csv
import hashlib
import json
import os
import pathlib
import sys
import tempfile

def digest(raw):
    return hashlib.sha256(pathlib.Path(raw).read_bytes()).hexdigest()

def max_step(path):
    maximum = -1
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
            value = row.get("trainer/global_step", row.get("misc/global_step", -1))
            maximum = max(maximum, int(float(value)))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return maximum

path = pathlib.Path(sys.argv[1])
parent = json.loads(pathlib.Path(sys.argv[3]).read_text(encoding="utf-8"))
replacements = list(csv.DictReader(pathlib.Path(sys.argv[6]).open(), delimiter="\t"))
sources = parent["mappings"]["graph_coloring"]
replacement_by_arm = {row["arm"]: row for row in replacements}
if len(sources) != 4 or len(replacements) != 4:
    raise SystemExit("Graph A5000 repair is not a four-cell mapping")
if set(replacement_by_arm) != {row["arm"] for row in sources}:
    raise SystemExit("Graph A5000 replacement arms differ")

mappings = []
invalid_attempts = []
data_root = pathlib.Path(sys.argv[1]).resolve().parents[1] / "data"
for source in sources:
    invalid_job = int(source["replacement_job_id"])
    matches = sorted(data_root.glob(f"*_{source['run_stamp']}/debug_job{invalid_job}"))
    if len(matches) != 1:
        raise SystemExit(f"missing unique Graph source attempt {invalid_job}")
    run_dir = matches[0]
    if (run_dir / "saved_models").exists():
        raise SystemExit(f"Graph source attempt unexpectedly terminal: {invalid_job}")
    metric_path = run_dir / "train_metrics.jsonl"
    invalid_attempts.append(
        {
            "arm": source["arm"],
            "job_id": invalid_job,
            "latest_step": max_step(metric_path),
            "run_dir": str(run_dir),
            "terminal": False,
        }
    )
    replacement = replacement_by_arm[source["arm"]]
    mappings.append(
        {
            "arm": source["arm"],
            "seed": int(source["seed"]),
            "invalid_job_id": invalid_job,
            "replacement_job_id": int(replacement["job_id"]),
            "run_stamp": replacement["run_stamp"],
        }
    )

payload = {
    "schema": "e69_gate2_graph_a5000_preemption_repair_v1",
    "original_identity_sha256": digest(sys.argv[2]),
    "parent_placement_repair_identity_sha256": digest(sys.argv[3]),
    "amendment_sha256": digest(sys.argv[4]),
    "launcher_sha256": digest(sys.argv[5]),
    "manifest_sha256": digest(sys.argv[6]),
    "source_hash": os.environ["SOURCE_HASH"],
    "execution_surface_hash": os.environ["EXECUTION_HASH"],
    "mappings": mappings,
    "invalid_attempts": invalid_attempts,
    "placement": {
        "node": "node202",
        "gres": "gpu:a5000:1",
        "partition": "lowprio",
        "account": "mltheory",
    },
    "attempt_selection": "replace_all_four_partial_graph_cells_after_preemption",
    "repair_decision_basis": "scheduler_preemption_and_node_inventory_only",
    "nonterminal_outcomes_available_before_repair": True,
    "terminal_outcomes_observed_before_repair": False,
    "math500_sealed": True,
}
path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY

  scancel "${GRAPH_A5000_ORIGINAL_JOBS[@]}"
  graph_originals_cancelled=1
  for job_id in "${GRAPH_A5000_ORIGINAL_JOBS[@]}"; do
    state="$(scontrol show job "$job_id" -o)"
    if [[ "$state" != *'JobState=CANCELLED'* ]]; then
      echo "Graph source job did not enter CANCELLED state: $job_id" >&2
      exit 1
    fi
  done
  scontrol release "${graph_replacements[@]}"
  graph_released=1
  trap - EXIT
  echo "[e69-graph-a5000-repair] cancelled originals: ${GRAPH_A5000_ORIGINAL_JOBS[*]}"
  echo "[e69-graph-a5000-repair] released replacements: ${graph_replacements[*]}"
  echo "[e69-graph-a5000-repair] identity=$GRAPH_A5000_REPAIR_IDENTITY"
  exit 0
fi

export OAT_ZERO_PROTOCOL_IDENTITY="$ORIGINAL_IDENTITY"
export OAT_ZERO_SBATCH_HOLD=1
released=0
originals_cancelled=0
replacement_jobs=()
cleanup_held() {
  local status="$?"
  trap - EXIT
  if [[ "$released" != "1" && "$originals_cancelled" != "1" \
      && "${#replacement_jobs[@]}" -gt 0 ]]; then
    scancel "${replacement_jobs[@]}" 2>/dev/null || true
    echo "[e69-placement-repair] cancelled incomplete held replacements" >&2
  elif [[ "$released" != "1" && "$originals_cancelled" == "1" ]]; then
    echo "[e69-placement-repair] originals cancelled; replacements remain held" >&2
  fi
  exit "$status"
}
trap cleanup_held EXIT

for domain in graph_coloring countdown python_factor; do
  submit_domain "$domain"
done
for manifest in "$GRAPH_MANIFEST" "$COUNTDOWN_MANIFEST" "$PYTHON_MANIFEST"; do
  mapfile -t manifest_jobs < <(
    awk -F $'\t' 'NR > 1 && $3 ~ /^[0-9]+$/ {print $3}' "$manifest"
  )
  if [[ "${#manifest_jobs[@]}" -ne 4 ]]; then
    echo "Placement-repair manifest must contain four jobs: $manifest" >&2
    exit 1
  fi
  replacement_jobs+=("${manifest_jobs[@]}")
done
if [[ "${#replacement_jobs[@]}" -ne 12 ]]; then
  echo "Placement-repair cohort must contain 12 jobs" >&2
  exit 1
fi

for job_id in "${replacement_jobs[@]}"; do
  record="$(scontrol show job "$job_id" -o)"
  for required in \
    'JobState=PENDING' 'Reason=JobHeldUser' \
    'OAT_ZERO_MAX_PROMPT_EPOCHS=6' \
    'OAT_ZERO_NUM_SAMPLES=16' \
    'OAT_ZERO_ONLINE_CANONICAL_COUNTERFACTUAL_FIXED_CONTROL_GROUPS=3' \
    'OAT_ZERO_ONLINE_CANONICAL_REPLAY_GLOBAL_GROUPS_PER_STEP=1' \
    "OAT_ZERO_SOURCE_ROOT=${SOURCE_ROOT}" \
    "OAT_ZERO_OPS_SNAPSHOT_ROOT=${OPS_ROOT}" \
    "OAT_ZERO_PROTOCOL_IDENTITY=${ORIGINAL_IDENTITY}"; do
    if [[ "$record" != *"$required"* ]]; then
      echo "Placement-repair held-job audit $job_id missing: $required" >&2
      exit 1
    fi
  done
done

export SOURCE_HASH EXECUTION_HASH
"$PYTHON_BIN" - \
  "$REPAIR_IDENTITY" "$ORIGINAL_IDENTITY" "$AMENDMENT" "$0" \
  "$GRAPH_MANIFEST" "$COUNTDOWN_MANIFEST" "$PYTHON_MANIFEST" <<'PY'
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
original = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
domains = ("graph_coloring", "countdown", "python_factor")
manifests = [pathlib.Path(raw) for raw in sys.argv[5:8]]
mappings = {}
for domain, manifest in zip(domains, manifests):
    replacements = list(csv.DictReader(manifest.open(), delimiter="\t"))
    originals = original["jobs"][domain]
    if len(replacements) != 4 or len(originals) != 4:
        raise SystemExit(f"{domain} repair is not a four-cell mapping")
    replacement_by_arm = {row["arm"]: row for row in replacements}
    if set(replacement_by_arm) != {row["arm"] for row in originals}:
        raise SystemExit(f"{domain} replacement arms differ")
    mappings[domain] = [
        {
            "arm": row["arm"],
            "seed": 43,
            "invalid_job_id": int(row["job_id"]),
            "replacement_job_id": int(replacement_by_arm[row["arm"]]["job_id"]),
            "run_stamp": replacement_by_arm[row["arm"]]["run_stamp"],
        }
        for row in originals
    ]

payload = {
    "schema": "e69_gate2_pending_placement_repair_v1",
    "original_identity_sha256": digest(sys.argv[2]),
    "amendment_sha256": digest(sys.argv[3]),
    "launcher_sha256": digest(sys.argv[4]),
    "manifest_sha256": {
        domain: digest(manifest)
        for domain, manifest in zip(domains, manifests)
    },
    "source_hash": os.environ["SOURCE_HASH"],
    "execution_surface_hash": os.environ["EXECUTION_HASH"],
    "mappings": mappings,
    "placement": {
        "graph_coloring": {
            "node": "node101",
            "gres": "gpu:a40:1",
            "partition": "lowprio",
            "account": "mltheory",
        },
        "countdown": {
            "node": "node105",
            "gres": "gpu:a5000:1",
            "partition": "lowprio",
            "account": "mltheory",
        },
        "python_factor": {
            "node": "node105",
            "gres": "gpu:a5000:1",
            "partition": "lowprio",
            "account": "mltheory",
        },
    },
    "invalid_jobs_optimizer_records": 0,
    "attempt_selection": "replace_exact_never_started_pending_cells",
    "outcomes_observed_before_repair": False,
    "math500_sealed": True,
}
path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, path)
PY

scancel "${ORIGINAL_JOBS[@]}"
originals_cancelled=1
for job_id in "${ORIGINAL_JOBS[@]}"; do
  state="$(scontrol show job "$job_id" -o)"
  if [[ "$state" != *'JobState=CANCELLED'* ]]; then
    echo "Original job did not enter CANCELLED state: $job_id" >&2
    exit 1
  fi
done
scontrol release "${replacement_jobs[@]}"
released=1
trap - EXIT
echo "[e69-placement-repair] cancelled originals: ${ORIGINAL_JOBS[*]}"
echo "[e69-placement-repair] released replacements: ${replacement_jobs[*]}"
echo "[e69-placement-repair] identity=$REPAIR_IDENTITY"
