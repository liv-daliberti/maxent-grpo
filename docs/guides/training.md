Training
========

This guide covers the common flows for GRPO training.

One‑Command Training
--------------------

Slurm (recommended):

```bash
sbatch training.sh
```

Local (for smoke tests):

```bash
bash training.sh
```

What the launcher does
----------------------

- Configures a repo‑local conda env and installs dependencies from `environment.yml`.
- Repins `huggingface-hub` to `< 1.0` for compatibility with Transformers/TRL.
- Patches Accelerate `num_processes` to the number of training GPUs.
- Starts vLLM on GPU 0, health‑checks `/health`, then launches GRPO training on GPUs 1–6.
- Writes logs to `logs/` (Slurm stdout/err, vLLM server log, training log).

Entrypoint and vLLM parameters
------------------------------

- Choose trainer:
  - Default: `TRAIN_ENTRYPOINT=src/grpo.py`
  - MaxEnt: set `MAXENT=1` (or `TRAIN_ENTRYPOINT=src/maxent-grpo.py`)
- Customize vLLM without editing the script:
  - `VLLM_MODEL` (default `Qwen/Qwen2.5-1.5B-Instruct`)
  - `VLLM_PORT` (default `8000`), `VLLM_TP_SIZE` (default `1`),
  - `VLLM_MAX_LEN` (default `2048`), `VLLM_MEM_UTIL` (default `0.90`)

Slurm specifics
---------------

The script includes an SBATCH header tailored for `mltheory` with 7× A100s. Adjust for your cluster as needed:

```bash
#SBATCH --job-name=OpenR1_GRPO
#SBATCH --account=mltheory
#SBATCH --partition=mltheory
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:7
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --exclusive
```

Inside the batch step, it runs a single `srun` that launches vLLM and the trainer:

```bash
srun --gres=gpu:7 --cpus-per-task=64 bash -c '
  set -euo pipefail

  # vLLM on GPU 0
  export CUDA_VISIBLE_DEVICES=0
  trl vllm-serve \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --dtype float16 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.90 \
    > "$SERVER_LOG" 2>&1 &
  VLLM_PID=$!

  until curl -sf http://localhost:8000/health >/dev/null; do sleep 2; done

  # Training on GPUs 1–6
  export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
  accelerate launch \
    --main_process_port 29525 \
    --config_file "$CONFIG_FILE" \
    src/grpo.py \
    --config "$CONFIG" \
    --use_vllm \
    --run_name "${RUN_NAME}-${TIMESTAMP}" \
    --ignore_data_skip \
    --overwrite_output_dir false \
    --seed 42 \
    > "$TRAINING_LOG" 2>&1

  wait $VLLM_PID
'
```

Key Files
---------

- `src/grpo.py` — trainer wiring (dataset → tokenizer/model → TRL GRPOTrainer)
- `src/configs.py` — configuration dataclasses (ScriptArguments, GRPOConfig, …)
- `recipes/` — ready‑to‑use YAML configs; see the Recipes page

Datasets
--------

You can train from a single dataset or a mixture. The mixture form lets you blend multiple sources with explicit columns and weights.

See: `src/configs.py:ScriptArguments` and `src/configs.py:DatasetMixtureConfig`.

Rewards
-------

The default reward is exact‑match for math answers found within `<answer>…</answer>` tags. See: `src/rewards.py` for details and extension points.

CLI / YAML Config
-----------------

Most options can be provided by CLI via TRL’s `TrlParser` or by a YAML recipe.

- Example knobs: `--system_prompt`, `--chat_template`, `--benchmarks`, W&B and Hub settings, MaxEnt controls (`--maxent_tau`, …)
- See API Reference → `configs` for all fields.

Logging & Checkpoints
---------------------

- Slurm stdout/err are in `logs/slurm_%j.out` and `logs/slurm_%j.err`.
- vLLM server logs to `logs/liv_vllm_<RUN>_<TIMESTAMP>.log`.
- Trainer logs to `logs/liv_train_<RUN>_<TIMESTAMP>.log`.
- Weights & Biases integration is available via `wandb_*` fields.
- Checkpoints are handled by TRL’s GRPOConfig as usual.

Common Flags
------------

- `--dataset_name` or `dataset_mixture` (YAML) to define training data
- `--system_prompt` and `--chat_template` to control prompting
- `--num_generations` and `--max_completion_length` for candidate sampling
- `--init_kl_coeff`, `--kl_target`, `--kl_horizon` for trust region
- `--report_to wandb` plus `wandb_*` fields for logging
- MaxEnt extras (when using `src/maxent-grpo.py`): `--maxent_tau`, `--maxent_q_temperature`, `--maxent_q_epsilon`, `--maxent_length_normalize_ref`

Troubleshooting
---------------

- vLLM not healthy / empty server log: verify GPUs were allocated (see `nvidia-smi -L` in Slurm log) and SBATCH GRES/partition match your cluster.
- GPU OOM early in training: reduce `per_device_train_batch_size` or `num_generations`, enable `gradient_checkpointing`.
- Very high KL: lower `init_kl_coeff` or raise `kl_target`; verify chat template and system prompt.
- Slow data loading: pre‑filter columns via `DatasetConfig.columns` and disable unused features.
- RTD or local docs errors: ensure `pip install -r docs/requirements.txt` and rebuild with `make docs`.
