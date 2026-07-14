#!/usr/bin/env bash
# Submit the paper's matched comparative on the exact multi-answer ModeBench
# data: Dr.GRPO (baseline) vs xDr.GRPO (tau sweep), plus the Token-MaxEnt
# control arm. All arms share the same data, model, prompt format, rollout
# budget G, and optimization settings; only the aggregation weight / extra
# regularizer differs. Dr.GRPO is the xdr tau=inf endpoint of the same code
# path (the grpo arm simply leaves OAT_ZERO_XDR_TAU=inf).
#
# After the training jobs finish, evaluate with:
#   ops/run_countdown_comparative_eval.sh   (see header there)
# and analyze with:
#   ops/analyze_countdown_comparative.py
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP_PREFIX="${RUN_STAMP_PREFIX:-countdown_comparative_$(date +%Y%m%d_%H%M%S)}"
TASK="${OAT_ZERO_COMPARATIVE_TASK:-countdown}"
# Model scale for every arm (tiny-probe model preset name).
MODEL="${OAT_ZERO_COMPARATIVE_MODEL:-qwen2.5-0.5b-instruct}"
# countdown data preset: easy3 (3 numbers, 2-8 modes, 384/128 — calibrated for
# 0.5B) or countdown4 (4 numbers, 2-8 modes, 1024/256 — the larger pool for
# 3B-scale models; a 0.5B gets no reward signal on it).
DATA_PRESET="${OAT_ZERO_COMPARATIVE_DATA_PRESET:-easy3}"
TRAIN_SEEDS_CSV="${OAT_ZERO_TRAIN_SEEDS:-43,44,45}"
XDR_TAUS_CSV="${OAT_ZERO_XDR_TAUS:-0.05,0.1,0.25,0.5,1,2}"
INCLUDE_TOKEN_ENTROPY_ARM="${OAT_ZERO_INCLUDE_TOKEN_ENTROPY_ARM:-1}"
INCLUDE_SEED_ARM="${OAT_ZERO_INCLUDE_SEED_ARM:-1}"
# Mode-adaptive tempering arm: tau_x = tau0/log(1+kappa_x) from observed
# distinct correct modes per group.
INCLUDE_XDR_ADAPT_ARM="${OAT_ZERO_INCLUDE_XDR_ADAPT_ARM:-0}"
XDR_ADAPT_TAU0="${OAT_ZERO_XDR_ADAPT_TAU0:-0.05}"
SEED_ENTROPY_ALPHA="${OAT_ZERO_SEED_ENTROPY_ALPHA:-1.0}"
# Stable recipe from the stabilization sweep (stab_sweep_1432): at lr 5e-6
# the policy's success rate collapses to zero within a run; at 2e-7 it rises
# and holds. The 3x prompt budget compensates the smaller steps.
LEARNING_RATE="${OAT_ZERO_LEARNING_RATE:-0.0000002}"
MAX_TRAIN="${OAT_ZERO_MAX_TRAIN:-9216}"
MAX_QUERIES="${OAT_ZERO_MAX_QUERIES:-100000}"
# G = 16 rollouts per prompt for every arm (G=8 leaves most groups without a
# correct sample at this model scale, starving the group-relative signal).
NUM_SAMPLES="${OAT_ZERO_NUM_SAMPLES:-16}"
TRAIN_NODELIST="${OAT_ZERO_TRAIN_NODELIST:-node105,node302}"
TRAIN_GRES="${OAT_ZERO_TRAIN_GRES:-gpu:1}"
# Short walltime: these 0.5B arms finish in a few hours, and a tight limit
# lets jobs backfill ahead of maintenance reservations instead of pending on
# the slurm template's 24h default.
TRAIN_TIME_LIMIT="${OAT_ZERO_TRAIN_TIME_LIMIT:-08:00:00}"
REBUILD_DATA="${OAT_ZERO_COMPARATIVE_REBUILD:-0}"

case "$TASK" in
  countdown)
    case "$DATA_PRESET" in
      easy3)
        DATA_ROOT="${OAT_ZERO_COMPARATIVE_DATA_ROOT:-$ROOT_DIR/var/data/exact_countdown_easy3_probe}"
        CD_TRAIN_SIZE="${OAT_ZERO_COUNTDOWN_MODE_TRAIN_SIZE:-384}"
        CD_EVAL_SIZE="${OAT_ZERO_COUNTDOWN_MODE_EVAL_SIZE:-128}"
        CD_NUMBER_COUNT="${OAT_ZERO_COUNTDOWN_MODE_NUMBER_COUNT:-3}"
        ;;
      countdown4)
        DATA_ROOT="${OAT_ZERO_COMPARATIVE_DATA_ROOT:-$ROOT_DIR/var/data/exact_countdown4_probe}"
        CD_TRAIN_SIZE="${OAT_ZERO_COUNTDOWN_MODE_TRAIN_SIZE:-1024}"
        CD_EVAL_SIZE="${OAT_ZERO_COUNTDOWN_MODE_EVAL_SIZE:-256}"
        CD_NUMBER_COUNT="${OAT_ZERO_COUNTDOWN_MODE_NUMBER_COUNT:-4}"
        ;;
      *)
        echo "Unknown OAT_ZERO_COMPARATIVE_DATA_PRESET=${DATA_PRESET}; use easy3 or countdown4." >&2
        exit 1
        ;;
    esac
    ;;
  graph_coloring)
    DATA_ROOT="${OAT_ZERO_COMPARATIVE_DATA_ROOT:-$ROOT_DIR/var/data/exact_answer_mode_probe}"
    ;;
  *)
    echo "Unknown OAT_ZERO_COMPARATIVE_TASK=${TASK}; use countdown or graph_coloring." >&2
    exit 1
    ;;
esac

mkdir -p "$ROOT_DIR/var/artifacts/logs"

if [[ "$REBUILD_DATA" == "1" ]] || [[ ! -f "$DATA_ROOT/train/dataset_dict.json" ]] || [[ ! -f "$DATA_ROOT/eval/dataset_dict.json" ]]; then
  case "$TASK" in
    countdown)
      "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
        "$ROOT_DIR/ops/make_exact_countdown_mode_data.py" \
        --output-root "$DATA_ROOT" \
        --train-size "$CD_TRAIN_SIZE" \
        --eval-size "$CD_EVAL_SIZE" \
        --number-count "$CD_NUMBER_COUNT" \
        --max-value "${OAT_ZERO_COUNTDOWN_MODE_MAX_VALUE:-12}" \
        --multi-min-modes "${OAT_ZERO_COUNTDOWN_MODE_MULTI_MIN_MODES:-2}" \
        --multi-max-modes "${OAT_ZERO_COUNTDOWN_MODE_MULTI_MAX_MODES:-8}" \
        --seed "${OAT_ZERO_COUNTDOWN_MODE_DATA_SEED:-0}" \
        --overwrite
      ;;
    graph_coloring)
      "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
        "$ROOT_DIR/ops/make_exact_answer_mode_data.py" \
        --output-root "$DATA_ROOT" \
        --train-size "${OAT_ZERO_ANSWER_MODE_TRAIN_SIZE:-192}" \
        --eval-size "${OAT_ZERO_ANSWER_MODE_EVAL_SIZE:-96}" \
        --seed "${OAT_ZERO_ANSWER_MODE_DATA_SEED:-0}" \
        --overwrite
      ;;
  esac
fi

# arm label -> tiny-probe variant + arm-specific env.
submit_arm() {
  local arm="$1"
  local variant="$2"
  local seed="$3"
  local xdr_tau="${4:-}"
  local seed_alpha="${5:-}"
  local run_stamp="${STAMP_PREFIX}_${arm}_s${seed}"
  local local_root="/tmp/${USER}/maxent-grpo-oat-zero-${run_stamp}"
  local export_vars
  export_vars="ALL"
  export_vars+=",RUN_STAMP=${run_stamp}"
  export_vars+=",OAT_ZERO_SEED=${seed}"
  export_vars+=",OAT_ZERO_TINY_VARIANT=${variant}"
  export_vars+=",OAT_ZERO_TINY_MODEL=${MODEL}"
  export_vars+=",OAT_ZERO_TINY_DATA_ROOT=${DATA_ROOT}"
  export_vars+=",OAT_ZERO_NUM_SAMPLES=${NUM_SAMPLES}"
  export_vars+=",OAT_ZERO_MAX_TRAIN=${MAX_TRAIN}"
  export_vars+=",OAT_ZERO_MAX_QUERIES=${MAX_QUERIES}"
  export_vars+=",OAT_ZERO_NUM_PROMPT_EPOCH=${OAT_ZERO_NUM_PROMPT_EPOCH:-1}"
  export_vars+=",OAT_ZERO_EVAL_STEPS=${OAT_ZERO_EVAL_STEPS:-64}"
  export_vars+=",OAT_ZERO_SAVE_STEPS=${OAT_ZERO_SAVE_STEPS:-64}"
  export_vars+=",OAT_ZERO_SAVE_FROM=${OAT_ZERO_SAVE_FROM:-64}"
  export_vars+=",OAT_ZERO_SAVE_CKPT=${OAT_ZERO_SAVE_CKPT:-0}"
  export_vars+=",OAT_ZERO_SAVE_INITIAL_MODEL=${OAT_ZERO_SAVE_INITIAL_MODEL:-0}"
  export_vars+=",OAT_ZERO_MAX_SAVE_NUM=${OAT_ZERO_MAX_SAVE_NUM:-4}"
  export_vars+=",OAT_ZERO_REQUIRE_FULL_EVAL_CHECKPOINTS=${OAT_ZERO_REQUIRE_FULL_EVAL_CHECKPOINTS:-0}"
  export_vars+=",OAT_ZERO_USE_WB=0"
  export_vars+=",OAT_ZERO_PROMPT_TEMPLATE=qwen_boxed"
  export_vars+=",OAT_ZERO_PROMPT_MAX_LENGTH=${OAT_ZERO_PROMPT_MAX_LENGTH:-256}"
  export_vars+=",OAT_ZERO_GENERATE_MAX_LENGTH=${OAT_ZERO_GENERATE_MAX_LENGTH:-192}"
  export_vars+=",OAT_ZERO_EVAL_GENERATE_MAX_LENGTH=${OAT_ZERO_EVAL_GENERATE_MAX_LENGTH:-192}"
  export_vars+=",OAT_ZERO_EVAL_BATCH_SIZE=${OAT_ZERO_EVAL_BATCH_SIZE:-64}"
  export_vars+=",OAT_ZERO_LOCAL_ROOT=${local_root}"
  # Pin every method knob on every arm: --export=ALL would otherwise let a
  # stray export in the submitting shell silently change an arm's objective
  # (e.g. give the baseline an entropy bonus, or flip arms onto the maxent
  # code path), voiding the matched design.
  export_vars+=",OAT_ZERO_OBJECTIVE=grpo"
  export_vars+=",OAT_ZERO_RND_SEED=0"
  export_vars+=",OAT_ZERO_LEARNING_RATE=${LEARNING_RATE}"
  if [[ "$variant" == "xdr_adapt" ]]; then
    export_vars+=",OAT_ZERO_XDR_MODE_ADAPTIVE=1"
  else
    export_vars+=",OAT_ZERO_XDR_MODE_ADAPTIVE=0"
  fi
  if [[ "$variant" == "grpo_entropy" ]]; then
    export_vars+=",OAT_ZERO_POLICY_ENTROPY_COEF=${OAT_ZERO_TOKEN_ENTROPY_COEF:-0.01}"
  else
    export_vars+=",OAT_ZERO_POLICY_ENTROPY_COEF=0.0"
  fi
  if [[ -n "$xdr_tau" ]]; then
    export_vars+=",OAT_ZERO_XDR_TAU=${xdr_tau}"
  else
    export_vars+=",OAT_ZERO_XDR_TAU=inf"
  fi
  if [[ -n "$seed_alpha" ]]; then
    export_vars+=",OAT_ZERO_SEED_ENTROPY_ALPHA=${seed_alpha}"
  else
    export_vars+=",OAT_ZERO_SEED_ENTROPY_ALPHA=0.0"
  fi

  sbatch --parsable "--export=${export_vars}" "--nodelist=${TRAIN_NODELIST}" \
    "--gres=${TRAIN_GRES}" "--time=${TRAIN_TIME_LIMIT}" \
    "$ROOT_DIR/ops/slurm/train_tiny_probe_node302.slurm"
}

IFS=',' read -r -a train_seeds <<< "$TRAIN_SEEDS_CSV"
IFS=',' read -r -a xdr_taus <<< "$XDR_TAUS_CSV"

echo "[comparative] stamp_prefix=${STAMP_PREFIX}"
echo "[comparative] task=${TASK}"
echo "[comparative] model=${MODEL}"
echo "[comparative] data_preset=${DATA_PRESET}"
echo "[comparative] data_root=${DATA_ROOT}"
echo "[comparative] train_seeds=${TRAIN_SEEDS_CSV}"
echo "[comparative] xdr_taus=${XDR_TAUS_CSV}"
echo "[comparative] num_samples=${NUM_SAMPLES}"
echo "[comparative] include_token_entropy_arm=${INCLUDE_TOKEN_ENTROPY_ARM}"

manifest="$ROOT_DIR/var/artifacts/${STAMP_PREFIX}_comparative_jobs.tsv"
if [[ -f "$manifest" ]]; then
  echo "Manifest ${manifest} already exists; pick a fresh RUN_STAMP_PREFIX" >&2
  echo "instead of resubmitting under an old stamp (mixed-environment runs" >&2
  echo "would be indistinguishable in the analysis)." >&2
  exit 1
fi
printf "arm\tseed\tjob_id\trun_stamp\n" > "$manifest"

for seed in "${train_seeds[@]}"; do
  job_id="$(submit_arm grpo grpo "$seed")"
  printf "grpo\t%s\t%s\t%s\n" "$seed" "$job_id" "${STAMP_PREFIX}_grpo_s${seed}" >> "$manifest"
  echo "[comparative] grpo seed=${seed} job=${job_id}"

  for tau in "${xdr_taus[@]}"; do
    arm="xdr_tau${tau//./p}"
    job_id="$(submit_arm "$arm" xdr "$seed" "$tau")"
    printf "%s\t%s\t%s\t%s\n" "$arm" "$seed" "$job_id" "${STAMP_PREFIX}_${arm}_s${seed}" >> "$manifest"
    echo "[comparative] ${arm} seed=${seed} job=${job_id}"
  done

  if [[ "$INCLUDE_TOKEN_ENTROPY_ARM" == "1" ]]; then
    job_id="$(submit_arm grpo_entropy grpo_entropy "$seed")"
    printf "grpo_entropy\t%s\t%s\t%s\n" "$seed" "$job_id" "${STAMP_PREFIX}_grpo_entropy_s${seed}" >> "$manifest"
    echo "[comparative] grpo_entropy seed=${seed} job=${job_id}"
  fi

  if [[ "$INCLUDE_XDR_ADAPT_ARM" == "1" ]]; then
    job_id="$(submit_arm xdr_adapt xdr_adapt "$seed" "$XDR_ADAPT_TAU0")"
    printf "xdr_adapt\t%s\t%s\t%s\n" "$seed" "$job_id" "${STAMP_PREFIX}_xdr_adapt_s${seed}" >> "$manifest"
    echo "[comparative] xdr_adapt seed=${seed} job=${job_id}"
  fi

  if [[ "$INCLUDE_SEED_ARM" == "1" ]]; then
    job_id="$(submit_arm seed seed "$seed" "" "$SEED_ENTROPY_ALPHA")"
    printf "seed\t%s\t%s\t%s\n" "$seed" "$job_id" "${STAMP_PREFIX}_seed_s${seed}" >> "$manifest"
    echo "[comparative] seed seed=${seed} job=${job_id}"
  fi
done

echo "[comparative] manifest=${manifest}"
echo "[comparative] next: RUN_STAMP_PREFIX=${STAMP_PREFIX} ops/run_countdown_comparative_eval.sh (on a GPU node)"
