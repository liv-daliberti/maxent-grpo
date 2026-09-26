#!/bin/bash
#SBATCH --job-name=launch_constructive_v2_replay
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=all
#SBATCH --time=01:00:00
#SBATCH --output=var/artifacts/logs/launch-constructive-v2-replay-%j.out
#SBATCH --error=var/artifacts/logs/launch-constructive-v2-replay-%j.err

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

exec "$ROOT_DIR/ops/exp_scaling/launch_constructive_code_v2_replay.sh" run
