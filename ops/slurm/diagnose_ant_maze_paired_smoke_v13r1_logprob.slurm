#!/bin/bash
#SBATCH --job-name=diagnose-ant-logprob-v13r1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=all
#SBATCH --account=allcs
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=00:30:00
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err

set -euo pipefail
ROOT_DIR="${ROOT_DIR:?ROOT_DIR is required}"
source "$ROOT_DIR/ops/repo_env.sh"
export TOKENIZERS_PARALLELISM=false

exec "$ROOT_DIR/var/seed_paper_eval/paper310/bin/python" \
  "$ROOT_DIR/ops/diagnose_ant_maze_paired_smoke_v13r1_logprob.py" \
  --model "$ROOT_DIR/var/models/ant_maze_interactive_warmstart_v13" \
  --identity "$ROOT_DIR/var/artifacts/ant_maze_interactive_paired_smoke_v13r1_identity.json" \
  --metrics "$ROOT_DIR/var/artifacts/ant_maze_interactive_paired_smoke_v13r1_grpo.metrics.jsonl" \
  --state-replay "$ROOT_DIR/var/artifacts/ant_maze_interactive_paired_smoke_v13r1_grpo.state_replay.jsonl" \
  --output "$ROOT_DIR/var/artifacts/ant_maze_interactive_paired_smoke_v13r1_logprob_batch_diagnostic.json"
