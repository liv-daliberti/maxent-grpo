Training
========

This guide covers the common flows for GRPO training.

One‑Command Training
--------------------

Slurm (recommended):

```bash
sbatch slurm/train.slurm \
  --model Qwen2.5-1.5B-Instruct \
  --task grpo \
  --config math \
  --accelerator zero3 \
  --args "--run_name demo --report_to wandb"
```

Quick flags:

- `--task maxent` launches `src/maxent-grpo.py`.
- `--dp/--tp` set vLLM data/tensor parallel sizes.
- `--vllm-port` / `--vllm-group-port` override RPC ports when needed.
- `--args "…" ` passes raw CLI to the trainer (quote the entire string).
- Authenticate with Hugging Face ahead of time (`huggingface-cli login` or `export HF_TOKEN=...`); the launcher forwards `HF_TOKEN` to every node for gated repos.
- See every option via `sbatch slurm/train.slurm --help`.

Local smoke tests (no Slurm) can invoke the trainer directly. Example:

```bash
accelerate launch \
  --config_file recipes/accelerate_configs/zero1.yaml \
  src/grpo.py \
  --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_math.yaml \
  --max_steps 5 --overwrite_output_dir true
```

What the Slurm launcher does
----------------------------

`slurm/train.slurm` is the orchestrator. It:

- Boots a conda env (default `./openr1`) or arbitrary venv depending on `ENV_MODE`.
- Loads CUDA 12.6 modules (editable) and exports `CUDA_HOME`, `PATH`, `LD_LIBRARY_PATH`.
- Redirects all Hugging Face / pip / torch caches into the repository so jobs stay self‑contained.
- Parses your YAML recipe to discover `use_vllm`, the model ID, gradient accumulation, and vLLM mode.
- Splits nodes between training and vLLM automatically:
  - single node → GPU 0 hosts vLLM, remaining GPUs run Accelerate.
  - multi‑node → reserves the last node for vLLM (data/tensor parallel controlled by `--dp/--tp`).
- Launches `trl.scripts.vllm_serve` with health checks, then fans out `accelerate launch` across remaining nodes/GPUs.
- Auto‑wires `--vllm_server_host/port` into the trainer if you did not pass them manually.
- Streams logs to `logs/train_<jobid>.log` and `logs/vllm-<jobid>.out` alongside the Slurm stdout/err files.

Entrypoint and vLLM parameters
------------------------------

- `--task grpo|maxent` toggles the trainer file (`src/grpo.py` vs `src/maxent-grpo.py`).
- `--model/--config` pair selects the YAML under `recipes/<model>/<task>/config_<suffix>.yaml`.
- `--accelerator` chooses the Accelerate config under `recipes/accelerate_configs/`.
- `--args` lets you append any CLI flag supported by the trainer.
- vLLM knobs:
  - `--dp`, `--tp` set data/tensor parallel width on the server node.
  - `--vllm-port` and `--vllm-group-port` control HTTP and NCCL RPC ports.
  - Environment variables like `VLLM_MODEL`, `VLLM_MODE`, `VLLM_MAX_LEN`, `VLLM_MEM_UTIL` are honoured when exporting extra options.
- The launcher reads `use_vllm` from the YAML; disable it in the recipe to keep everything inline.

Slurm specifics
---------------

`slurm/train.slurm` ships with a conservative SBATCH header. Update account, partition, GPU type/count, and walltime for your site.

```bash
#SBATCH --job-name=open_r1
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --account=mltheory
#SBATCH --partition=mltheory
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=128:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
```

On submission the script:

- Reserves GPUs/nodes based on the header.
- Optionally spins up a dedicated vLLM node (multi‑node) or a single GPU on the training node (inline).
- Launches Accelerate with `--num_machines`/`--num_processes` derived from the available training GPUs.

Key Files
---------

- `src/grpo.py` — trainer wiring (dataset → tokenizer/model → TRL GRPOTrainer)
- `src/configs.py` — configuration dataclasses (ScriptArguments, GRPOConfig, …)
- `recipes/` — ready‑to‑use YAML configs; see the Recipes page

Datasets
--------

You can train from a single dataset or a mixture. The mixture form lets you blend multiple sources with explicit columns and weights.

See: `src/configs.py:ScriptArguments` and `src/configs.py:DatasetMixtureConfig`.

Prompts longer than 2,048 characters are clipped before requests are sent to vLLM to avoid HTTP payload failures. Override the limit with `MAX_PROMPT_CHARS` if your setup requires more context.

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

- Slurm stdout/err live in `logs/<job-name>-<jobid>.out|err`.
- vLLM server logs to `logs/vllm-<jobid>.out` (or inline when sharing a node).
- Trainer logs to `logs/train_<jobid>.log`.
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
