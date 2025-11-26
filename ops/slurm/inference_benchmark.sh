export DATASETS="math_500,aime24,aime25,amc,minerva"

for MODEL_ROOT in \
  var/data/Qwen2.5-1.5B-Open-R1-MaxEnt-GRPO-math-v1 \
  var/data/Qwen2.5-1.5B-Open-R1-MaxEnt-GRPO-BASELINE-math-v1
do
  find "$MODEL_ROOT" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | while read -r CKPT; do
    echo "Launching $CKPT on $DATASETS"
    sbatch ops/slurm/infer_math.slurm --model "$CKPT" --datasets "$DATASETS"
  done
done
