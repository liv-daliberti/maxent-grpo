#!/bin/bash
# Submit the Level-5 frozen base-model grid, one array per model scale.
#
# Resources are the ones the Level 1-4 receipts record for the same models:
# tensor parallelism 1 up to 14B, 2 at 32B, 4 at 72B, all float16 at
# max_model_len 2048 and gpu_memory_utilization .82. 14B and larger need 48 GB
# cards, so the big scales pin a6000. Wall times are first estimates for
# 128 prompts x 4 groups x 8 draws at <=192 tokens; tighten them from
# `seff <jobid>` after the first cell rather than padding.
#
# Usage:  RUN_ROOT=artifacts/modebench_base_level_grid_level5_<date> \
#         DOMAIN_LIST="countdown" ops/submit_modebench_level5_base_grid.sh
# Add --dry-run (default) to print submissions; pass --submit to send them.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CACHE="$ROOT_DIR/var/cache/huggingface/transformers"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required (a dated artifacts run directory)}"
DOMAIN_LIST="${DOMAIN_LIST:?DOMAIN_LIST is required, space separated}"
MODE="${1:---dry-run}"
read -r -a DOMAINS <<< "$DOMAIN_LIST"
LAST=$(( ${#DOMAINS[@]} - 1 ))

snapshot () { ls -d "$CACHE/models--Qwen--Qwen2.5-$1-Instruct/snapshots/"*/ | head -1; }

# label  hf-size  tp  gres              mem    time
# Wall times re-calibrated 2026-09-15 from a measured 0.5B countdown cell
# (job 31290482): 13 of 64 batches in ~16 min including load, i.e. ~67 min for a
# full cell against the original 40 min estimate. The original figures were first
# estimates and under-ran by about 1.75x, so each is scaled accordingly. --resume
# means a cell that still over-runs continues from its saved batches.
CELLS=(
  # The rtx_3090 nodes reject any job over 60 minutes ("Requested node
  # configuration is not available" at submit time, verified by bisection on
  # 2026-09-15: 01:00:00 accepted, 01:05:00 rejected). A full cell needs ~67 min,
  # so the two small scales move to a6000, which accepts the longer wall time.
  "05b      0.5B  1  gpu:a6000:1        64G   01:20:00"
  # 1.5B was missing from this list when Level 5 was launched, so the scale that
  # carries the PCMD peak in Graph, Python and Pantry had no Level-5 cell at all.
  # Sized between the 0.5B and 3B rows, which both run on one card in 64G.
  "qwen15b  1.5B  1  gpu:a6000:1        64G   01:30:00"
  "3b       3B    1  gpu:a6000:1        64G   01:30:00"
  "7b       7B    1  gpu:a6000:1        64G   01:40:00"
  "14b      14B   1  gpu:a6000:1        64G   02:00:00"
  "qwen32b  32B   2  gpu:a6000:2        96G   02:30:00"
  "qwen72b  72B   4  gpu:a6000:4       160G   03:30:00"
)

# Optional: restrict to one or more scales, so a gap can be filled without
# resubmitting cells that already have receipts. Empty means every scale.
MODEL_FILTER="${MODEL_FILTER:-}"

for row in "${CELLS[@]}"; do
  read -r label size tp gres mem time <<< "$row"
  if [[ -n "$MODEL_FILTER" && ! " $MODEL_FILTER " == *" $label "* ]]; then continue; fi
  path="$(snapshot "$size")"
  cmd=(sbatch --job-name="mb-l5-$label" --array="0-$LAST" --gres="$gres"
       --mem="$mem" --time="$time"
       --export="ALL,RUN_ROOT=$RUN_ROOT,MODEL_LABEL=$label,MODEL_PATH=$path,TP=$tp,DOMAIN_LIST=$DOMAIN_LIST"
       "$ROOT_DIR/ops/slurm/evaluate_modebench_level5_base_grid.slurm")
  if [[ "$MODE" == "--submit" ]]; then "${cmd[@]}"; else printf '%s\n\n' "${cmd[*]}"; fi
done
