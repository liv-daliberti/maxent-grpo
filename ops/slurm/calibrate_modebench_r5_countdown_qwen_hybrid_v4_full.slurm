#!/usr/bin/env bash
#SBATCH --job-name=mb-r5-qwen-v4
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=var/logs/modebench_level2_viability/%j.out
#SBATCH --error=var/logs/modebench_level2_viability/%j.err
set -euo pipefail
ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
source "$ROOT_DIR/ops/repo_env.sh"
PYTHON_BIN="$ROOT_DIR/var/seed_paper_eval/paper310/bin/python"
MODEL="$ROOT_DIR/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
out="$ROOT_DIR/var/results/modebench_r5_countdown_qwen_hybrid_v4_full/qwen-0.5b__countdown.json"
mkdir -p "$(dirname "$out")" "$ROOT_DIR/var/logs/modebench_level2_viability"
exec "$PYTHON_BIN" "$ROOT_DIR/ops/evaluate_modebench_level2_viability.py" \
 --level2-root "$ROOT_DIR/var/data/modebench_harder_v2_matched_r5" \
 --domain countdown --model "$MODEL" --model-label qwen-0.5b --output "$out" \
 --seed 5317100 --temperature 1 --top-p 1 --max-tokens 192 --max-model-len 2048 \
 --batch-size 8 --dtype float16 --prompt-profile hybrid_solver_v4 --row-limit 128 \
 --syntax-profile countdown_legal_v3
