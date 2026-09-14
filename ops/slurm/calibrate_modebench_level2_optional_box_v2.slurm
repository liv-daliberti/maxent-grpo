#!/usr/bin/env bash
#SBATCH --job-name=mb-l2-pass8
#SBATCH --array=0-3%2
#SBATCH --gres=gpu:a6000:1
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
DOMAINS=(countdown mathir)
MODELS=(qwen-0.5b falcon-1b)
idx=${SLURM_ARRAY_TASK_ID:?}; model_i=$((idx / 2)); domain_i=$((idx % 2)); offsets=(0 2)
model_label=${MODELS[$model_i]}; domain=${DOMAINS[$domain_i]}
if [[ "$model_label" == qwen-0.5b ]]; then model=$QWEN; seed=$((5317100+offsets[domain_i])); else model=$FALCON; seed=$((5317200+offsets[domain_i])); fi
out="$ROOT_DIR/var/results/modebench_level2_optional_box_v2/${model_label}__${domain}.json"
mkdir -p "$(dirname "$out")" "$ROOT_DIR/var/logs/modebench_level2_viability"
exec "$PYTHON_BIN" "$ROOT_DIR/ops/evaluate_modebench_level2_viability.py" --domain "$domain" --model "$model" --model-label "$model_label" --output "$out" --seed "$seed" --temperature 1 --top-p 1 --max-tokens 192 --max-model-len 2048 --batch-size 8 --dtype float16 --prompt-profile hybrid_solver_v4 --row-limit 32 --syntax-profile domain_legal_v2
