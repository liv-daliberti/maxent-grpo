#!/usr/bin/env bash
set -euo pipefail

required=(
  E113R4_ROOT E113R4_RUNTIME_SNAPSHOT E113R4_VERL_ROOT E113R4_VERIFIER_SITE E113R4_IMAGE
  E113R4_IMAGE_SIZE
  E113R4_FAMILY E113R4_DOMAIN E113R4_SEED E113R4_OUTPUT
  E113R4_TRAIN_FILE E113R4_TRAIN_SHA256 E113R4_VAL_FILE E113R4_VAL_SHA256
  E113R4_MODEL E113R4_PROMPT_LENGTH
  E113R4_RESPONSE_LENGTH E113R4_OVERLONG_BUFFER E113R4_TOTAL_STEPS
  E113R4_MAX_EPOCHS
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing required environment variable: ${name}" >&2
    exit 2
  fi
done

if [[ "$(stat -c '%s' "$E113R4_IMAGE")" != "$E113R4_IMAGE_SIZE" ]]; then
  echo "pinned E113-R4 image size drifted" >&2
  exit 2
fi
actual_train_sha="$(sha256sum "$E113R4_TRAIN_FILE" | awk '{print $1}')"
actual_val_sha="$(sha256sum "$E113R4_VAL_FILE" | awk '{print $1}')"
if [[ "$actual_train_sha" != "$E113R4_TRAIN_SHA256" ]]; then
  echo "E113-R4 train parquet hash drifted" >&2
  exit 2
fi
if [[ "$actual_val_sha" != "$E113R4_VAL_SHA256" ]]; then
  echo "E113-R4 validation parquet hash drifted" >&2
  exit 2
fi

if [[ ! -x "$(command -v apptainer)" ]]; then
  echo "apptainer is unavailable" >&2
  exit 2
fi
for path in \
  "$E113R4_IMAGE" \
  "$E113R4_TRAIN_FILE" \
  "$E113R4_VAL_FILE" \
  "$E113R4_VERIFIER_SITE/latex2sympy2_extended/__init__.py" \
  "$E113R4_VERIFIER_SITE/math_verify/__init__.py" \
  "$E113R4_MODEL/config.json" \
  "$E113R4_VERL_ROOT/recipe/dapo/src/main_dapo.py" \
  "$E113R4_RUNTIME_SNAPSHOT/src/oat_drgrpo/math_grader.py" \
  "$E113R4_RUNTIME_SNAPSHOT/ops/exp_scaling/e113r4_verl_reward.py"; do
  if [[ ! -f "$path" ]]; then
    echo "missing frozen E113-R4 input: ${path}" >&2
    exit 2
  fi
done

mkdir -p \
  "$E113R4_OUTPUT" \
  "$E113R4_OUTPUT/checkpoints" \
  "$E113R4_OUTPUT/hydra" \
  "$E113R4_OUTPUT/tmp"

# Ray appends a timestamped session directory and Unix-domain socket names to
# its temporary root. Keeping that root under the long scientific output path
# exceeded Linux's 107-byte AF_UNIX limit before either R4 smoke could start.
# Use a unique, short, node-local directory for process-local state while all
# scientific outputs and checkpoints remain in E113R4_OUTPUT.
job_tmp="$(mktemp -d "/tmp/e113r4-${SLURM_JOB_ID:-manual}-XXXXXX")"
cleanup_job_tmp() {
  rm -rf -- "$job_tmp"
}
trap cleanup_job_tmp EXIT

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=true
export WANDB_MODE=disabled
export PYTHONHASHSEED="$E113R4_SEED"
export TMPDIR="$job_tmp"
export RAY_TMPDIR="$job_tmp"
export NCCL_DEBUG=WARN
unset RAY_ADDRESS || true

max_token_length=$((E113R4_PROMPT_LENGTH + E113R4_RESPONSE_LENGTH))
# Ten generation batches remains the frozen default. Exact-cell liveness
# recoveries may raise this bounded attempt budget without changing the
# requirement that every optimizer update contain 128 non-constant groups.
max_num_gen_batches="${E113R4_MAX_NUM_GEN_BATCHES:-10}"
if [[ ! "$max_num_gen_batches" =~ ^[1-9][0-9]*$ ]]; then
  echo "E113R4_MAX_NUM_GEN_BATCHES must be a positive integer" >&2
  exit 2
fi
# The official DAPO launcher caps aggregate scheduled tokens at one full
# sequence, while verl's pinned rollout config permits up to 1,024 concurrent
# sequences. Published DAPO's 22,528-token context satisfies vLLM's required
# max_num_batched_tokens >= max_num_seqs invariant implicitly; the scaled
# ModeBench contexts do not. Preserve verl's concurrency setting and lift
# only the aggregate scheduler cap when the scientific context is shorter.
rollout_max_num_seqs=1024
rollout_max_num_batched_tokens="$max_token_length"
if (( rollout_max_num_batched_tokens < rollout_max_num_seqs )); then
  rollout_max_num_batched_tokens="$rollout_max_num_seqs"
fi
experiment_name="$(basename "$E113R4_OUTPUT")"
reward_path="$E113R4_RUNTIME_SNAPSHOT/ops/exp_scaling/e113r4_verl_reward.py"
python_path="$E113R4_VERIFIER_SITE:$E113R4_RUNTIME_SNAPSHOT/src:$E113R4_VERL_ROOT"

apptainer exec \
  --nv \
  --bind "$E113R4_ROOT:$E113R4_ROOT" \
  --bind "$job_tmp:$job_tmp" \
  --pwd "$E113R4_VERL_ROOT" \
  --env "PYTHONPATH=$python_path" \
  --env "HF_HUB_OFFLINE=1" \
  --env "TRANSFORMERS_OFFLINE=1" \
  --env "TOKENIZERS_PARALLELISM=true" \
  --env "WANDB_MODE=disabled" \
  --env "PYTHONHASHSEED=$E113R4_SEED" \
  --env "TMPDIR=$TMPDIR" \
  --env "RAY_TMPDIR=$RAY_TMPDIR" \
  "$E113R4_IMAGE" \
  python3 -m recipe.dapo.src.main_dapo \
    "data.train_files=$E113R4_TRAIN_FILE" \
    "data.val_files=$E113R4_VAL_FILE" \
    data.prompt_key=prompt \
    data.truncation=error \
    data.filter_overlong_prompts=False \
    data.shuffle=True \
    "+data.seed=$E113R4_SEED" \
    "data.max_prompt_length=$E113R4_PROMPT_LENGTH" \
    "data.max_response_length=$E113R4_RESPONSE_LENGTH" \
    data.gen_batch_size=384 \
    data.train_batch_size=128 \
    actor_rollout_ref.rollout.n=16 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.20 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=True \
    "algorithm.filter_groups.max_num_gen_batches=$max_num_gen_batches" \
    algorithm.filter_groups.metric=acc \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$max_token_length" \
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$max_token_length" \
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$max_token_length" \
    "actor_rollout_ref.model.path=$E113R4_MODEL" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    "actor_rollout_ref.rollout.max_num_batched_tokens=$rollout_max_num_batched_tokens" \
    "actor_rollout_ref.rollout.max_num_seqs=$rollout_max_num_seqs" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    "actor_rollout_ref.rollout.engine_seed=$E113R4_SEED" \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=True \
    "reward_model.overlong_buffer.len=$E113R4_OVERLONG_BUFFER" \
    reward_model.overlong_buffer.penalty_factor=1.0 \
    "custom_reward_function.path=$reward_path" \
    custom_reward_function.name=compute_score \
    "trainer.logger=['console']" \
    trainer.project_name=E113-R4-official-verl-DAPO \
    "trainer.experiment_name=$experiment_name" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.test_freq=5 \
    trainer.save_freq=5 \
    "trainer.total_epochs=$E113R4_MAX_EPOCHS" \
    "trainer.total_training_steps=$E113R4_TOTAL_STEPS" \
    "trainer.default_local_dir=$E113R4_OUTPUT/checkpoints" \
    trainer.resume_mode=auto \
    "hydra.run.dir=$E113R4_OUTPUT/hydra" \
    hydra.output_subdir=null

python3 "$E113R4_RUNTIME_SNAPSHOT/ops/exp_scaling/e113r4_write_receipt.py" \
  --output "$E113R4_OUTPUT" \
  --family "$E113R4_FAMILY" \
  --domain "$E113R4_DOMAIN" \
  --seed "$E113R4_SEED" \
  --steps "$E113R4_TOTAL_STEPS" \
  --snapshot "$E113R4_RUNTIME_SNAPSHOT"

