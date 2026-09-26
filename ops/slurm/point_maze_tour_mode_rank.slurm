#!/bin/bash
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G
#SBATCH --output=var/artifacts/logs/%x-%j.out
#SBATCH --error=var/artifacts/logs/%x-%j.err
set -euo pipefail
cd "$ROOT_DIR"; source ops/repo_env.sh
export PYTHONPATH="$ROOT_DIR/ops:$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OAT_ZERO_REPO_ROOT="$ROOT_DIR" OAT_ZERO_SOURCE_ROOT="$ROOT_DIR/src" OAT_ZERO_EXECUTION_ROOT="$ROOT_DIR/ops"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl
run () {  # model label [prompt_format]
  var/seed_paper_eval/paper310/bin/python ops/measure_point_maze_tour_mode_rank.py \
    --model "$1" --data-root var/data/point_maze_tour_v1r1 \
    --worker-python var/maze_runtime/venv/bin/python \
    --label "$2" --output "var/artifacts/tour_rank/$2.json"
}
run "$M3B"                                        base3b_pass0
run "$FAL"                                        basefal_pass0
run var/checkpoints/tour_e93pt_control_s43/model  control_s43_pass8
run var/checkpoints/tour_e93pt_replay_s43/model   replay_s43_pass8
run var/checkpoints/tour_e93pt_control_s45/model  control_s45_pass8
run var/checkpoints/tour_e93pt_replay_s45/model   replay_s45_pass8
