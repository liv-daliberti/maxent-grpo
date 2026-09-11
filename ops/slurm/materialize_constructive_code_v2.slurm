#!/bin/bash
#SBATCH --job-name=constructive_code_v2_materialize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --partition=all
#SBATCH --time=04:00:00
#SBATCH --output=var/artifacts/logs/constructive-code-v2-materialize-%j.out
#SBATCH --error=var/artifacts/logs/constructive-code-v2-materialize-%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

exec /n/fs/similarity/maxent-grpo/var/seed_paper_eval/paper310/bin/python \
  "$ROOT_DIR/ops/materialize_constructive_code_v2.py"
