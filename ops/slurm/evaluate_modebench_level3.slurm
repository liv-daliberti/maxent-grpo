#!/usr/bin/env bash
#SBATCH --job-name=mb-level3
#SBATCH --partition=all
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=var/logs/modebench_level3/%j.out
#SBATCH --error=var/logs/modebench_level3/%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source ops/repo_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V1=0
export OMP_NUM_THREADS=4
label=${1:?model label 05b or 3b required}
tasks=${2:?task manifest required}
case "$label" in
  05b) model="$PWD/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775" ;;
  3b) model="$PWD/var/cache/huggingface/transformers/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1" ;;
  *) echo "unknown model label: $label" >&2; exit 2 ;;
esac
shift 2
exec var/seed_paper_eval/paper310/bin/python ops/evaluate_modebench_level3.py \
  --model "$model" --model-label "$label" --tasks-json "$tasks" --resume "$@"
