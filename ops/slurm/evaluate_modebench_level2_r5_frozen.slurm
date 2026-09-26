#!/usr/bin/env bash
#SBATCH --job-name=mb-r5-final
#SBATCH --array=0-9%6
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=var/logs/modebench_level2_viability/%A_%a.out
#SBATCH --error=var/logs/modebench_level2_viability/%A_%a.err
set -euo pipefail
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
source "$ROOT_DIR/ops/repo_env.sh"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
QWEN="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
FALCON="$ROOT_DIR/var/cache/huggingface/transformers/models--tiiuae--Falcon3-1B-Instruct/snapshots/28ba2251970a01dd1edc7ba7dad2eb71216ccfdf"
DOMAINS=(countdown graph_coloring python_factors mathir pantry)
MODELS=(qwen-0.5b falcon-1b)
idx=${SLURM_ARRAY_TASK_ID:?}; model_i=$((idx / 5)); domain_i=$((idx % 5))
model_label=${MODELS[$model_i]}; domain=${DOMAINS[$domain_i]}
if [[ "$model_label" == qwen-0.5b ]]; then model=$QWEN; base=5317100; else model=$FALCON; base=5317200; fi
profile=hybrid_solver_v4; syntax=domain_legal_v1; max_len=2048
case "$domain" in
 countdown)
  seed=$base; syntax=countdown_legal_v3
  if [[ "$model_label" == falcon-1b ]]; then profile=countdown_shallow_v6; fi
  ;;
 graph_coloring)
  profile=boxed_direct_v1; syntax=none; max_len=1024
  if [[ "$model_label" == qwen-0.5b ]]; then seed=5217101; else seed=5217201; fi
  ;;
 python_factors) seed=$((base+1));;
 mathir) seed=$((base+2));;
 pantry) seed=$((base+3));;
 *) exit 2;;
esac
results_dir="${RESULTS_DIR:-$ROOT_DIR/var/results/modebench_level2_r5_frozen}"
out="$results_dir/${model_label}__${domain}.json"
mkdir -p "$(dirname "$out")" "$ROOT_DIR/var/logs/modebench_level2_viability"
exec "$PYTHON_BIN" "$ROOT_DIR/ops/evaluate_modebench_level2_viability.py" \
 --level2-root "$ROOT_DIR/var/data/modebench_harder_v2_matched_r5" \
 --domain "$domain" --model "$model" --model-label "$model_label" --output "$out" \
 --seed "$seed" --temperature 1 --top-p 1 --max-tokens 192 --max-model-len "$max_len" \
 --batch-size 8 --dtype float16 --prompt-profile "$profile" --row-limit 0 \
 --syntax-profile "$syntax"
