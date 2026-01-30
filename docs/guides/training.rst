Training
========

This guide covers the common flows for GRPO training.

One‑Command Training
--------------------

Slurm (recommended):

.. code-block:: bash

   sbatch ops/slurm/train.slurm \
     --model Qwen2.5-1.5B-Instruct \
     --task grpo \
     --config math \
     --accelerator zero3 \
     --args "--run_name demo --report_to wandb"

Quick flags:

- ``--task maxent`` launches the shared MaxEnt/GRPO pipeline. When ``train_grpo_objective=false`` the run uses the custom loop under ``src/maxent_grpo/training/loop.py``; when ``train_grpo_objective=true`` it defaults to TRL’s GRPOTrainer unless you set ``force_custom_loop=true`` or enable the meta-controller. Paired GRPO recipes set ``force_custom_loop: true`` so GRPO and MaxEnt execute the same loop for parity.
- ``--task infoseed`` (or a recipe with ``info_seed_enabled=true``) routes through ``src/maxent_grpo/pipelines/training/infoseed.py`` which always uses the custom loop so the auxiliary seed loss can tap the same hooks as MaxEnt. Hydra validation enforces ``info_seed_enabled`` for this command, so keep the flag true unless you switch back to the baseline/MaxEnt recipes.
- ``--dp/--tp`` set vLLM data/tensor parallel sizes.
- ``--vllm-port`` / ``--vllm-group-port`` override RPC ports when needed.
- ``--args "…"`` passes raw CLI to the trainer (quote the entire string).
- Authenticate with Hugging Face ahead of time (``huggingface-cli login`` or ``export HF_TOKEN=...``); the launcher forwards ``HF_TOKEN`` to every node for gated repos.
- See every option via ``sbatch ops/slurm/train.slurm --help``.

Local smoke tests (no Slurm) can use the Hydra console scripts. Examples:

.. code-block:: bash

   # Baseline GRPO with inline overrides
   maxent-grpo-baseline command=train-baseline training.output_dir=var/data/out

   # MaxEnt-GRPO using a YAML recipe
   GRPO_RECIPE=configs/recipes/Qwen2.5-1.5B-Instruct/maxent-grpo/config_math.yaml \
     maxent-grpo-maxent

Recipe pairing (reproducible GRPO_RECIPE runs)
----------------------------------------------

- The baseline and MaxEnt math recipes are paired to stay comparable: both use ``open-r1/OpenR1-Math-220k`` for training and ``HuggingFaceH4/MATH-500`` (``test`` split, ``problem``/``answer`` columns) for evaluation with the same seed (``42``) and eval cadence (``evaluation_strategy=steps``, ``eval_steps=25``, ``per_device_eval_batch_size=8``).
- Baseline GRPO recipe: ``configs/recipes/Qwen2.5-1.5B-Instruct/grpo/config_math.yaml``
- MaxEnt-GRPO recipe: ``configs/recipes/Qwen2.5-1.5B-Instruct/maxent-grpo/config_math.yaml``
- Paired GRPO recipes set ``force_custom_loop: true`` and ``maxent_reference_logprobs_source: model`` so GRPO runs through the same custom loop with a frozen reference anchor.
- InfoSeed recipe (custom loop + auxiliary loss): ``configs/recipes/<model>/infoseed/config_math.yaml``. Those keep ``info_seed_enabled: true`` (required for ``train-infoseed``) and default to the InfoSeed variant of the pipeline; flip the flag (or use the MaxEnt/GRPO recipes) to disable seed conditioning entirely.
- Hydra console wrappers reference the same pair: ``configs/recipes/hydra/baseline_math.yaml`` and ``configs/recipes/hydra/maxent_math.yaml`` so ``GRPO_RECIPE=… maxent-grpo-{baseline,maxent}`` will read the intended sibling. For a custom-loop GRPO run, use ``configs/recipes/hydra/grpo_custom_math.yaml``.

What the Slurm launcher does
----------------------------

``ops/slurm/train.slurm`` is the orchestrator. It:

- Boots a conda env (default ``./var/openr1``) or arbitrary venv depending on ``ENV_MODE``.
- Loads CUDA 12.6 modules (editable) and exports ``CUDA_HOME``, ``PATH``, ``LD_LIBRARY_PATH``.
- Redirects all Hugging Face / pip / torch caches into ``var/`` so jobs stay self‑contained.
- Parses your YAML recipe to discover ``use_vllm``, the model ID, gradient accumulation, and vLLM mode.
- Splits nodes between training and vLLM automatically:
  - single node → GPU 0 hosts vLLM, remaining GPUs run Accelerate.
  - multi‑node → reserves the last node for vLLM (data/tensor parallel controlled by ``--dp/--tp``).
- Launches ``trl.scripts.vllm_serve`` with health checks, then fans out ``accelerate launch`` across remaining nodes/GPUs.
- Auto‑wires ``--vllm_server_host/port`` into the trainer if you did not pass them manually.
- Streams logs to ``var/artifacts/logs/train_<jobid>.log`` and ``var/artifacts/logs/vllm-<jobid>.out`` alongside the Slurm stdout/err files.

Entrypoint and vLLM parameters
------------------------------

- ``--task grpo|maxent`` toggles the trainer file (``src/maxent_grpo/grpo.py`` vs ``src/maxent_grpo/maxent_grpo.py``).
- ``--model/--config`` pair selects the YAML under ``configs/recipes/<model>/<task>/config_<suffix>.yaml``.
- ``--accelerator`` chooses the Accelerate config under ``configs/recipes/accelerate_configs/``.
- ``--args`` lets you append any CLI flag supported by the trainer.
- vLLM knobs:
  - ``--dp``, ``--tp`` set data/tensor parallel width on the server node.
  - ``--vllm-port`` and ``--vllm-group-port`` control HTTP and NCCL RPC ports.
  - Environment variables like ``VLLM_MODEL``, ``VLLM_MODE``, ``VLLM_MAX_LEN``, ``VLLM_MEM_UTIL`` are honoured when exporting extra options.
- The launcher reads ``use_vllm`` from the YAML; disable it in the recipe to keep everything inline.

Slurm specifics
---------------

``ops/slurm/train.slurm`` ships with a conservative SBATCH header. Update account, partition, GPU type/count, and walltime for your site.

.. code-block:: bash

   #SBATCH --job-name=open_r1
   #SBATCH --nodes=1
   #SBATCH --gres=gpu:8
   #SBATCH --account=mltheory
   #SBATCH --partition=mltheory
   #SBATCH --ntasks-per-node=1
   #SBATCH --exclusive
   #SBATCH --time=128:00:00
   #SBATCH --output=var/artifacts/logs/%x-%j.out
   #SBATCH --error=var/artifacts/logs/%x-%j.err

On submission the script:

- Reserves GPUs/nodes based on the header.
- Optionally spins up a dedicated vLLM node (multi‑node) or a single GPU on the training node (inline).
- Launches Accelerate with ``--num_machines``/``--num_processes`` derived from the available training GPUs.

Key Files
---------

- ``src/maxent_grpo/grpo.py`` — trainer wiring (dataset → tokenizer/model → TRL GRPOTrainer)
- ``src/maxent_grpo/maxent_grpo.py`` — MaxEnt entrypoint that detects ``train_grpo_objective`` and either wraps TRL’s trainer (GRPO mode) or drives the fully custom controller-aware loop (MaxEnt mode, or GRPO when ``force_custom_loop`` is set).
- ``src/maxent_grpo/pipelines/training/infoseed.py`` — InfoSeed pipeline that always runs through the custom loop so auxiliary losses can inspect weight stats.
- ``src/maxent_grpo/config/`` — configuration dataclasses (ScriptArguments, GRPOConfig, …)
- ``configs/recipes/`` — ready-to-use YAML configs; see the Recipes page

Datasets
--------

You can train from a single dataset or a mixture. The mixture form lets you blend multiple sources with explicit columns and weights.

See: ``src/maxent_grpo/config/dataset.py:ScriptArguments`` and ``src/maxent_grpo/config/dataset.py:DatasetMixtureConfig``.

Prompts longer than 2,048 characters are clipped before requests are sent to vLLM to avoid HTTP payload failures. Override the limit with ``MAX_PROMPT_CHARS`` if your setup requires more context.

Shared vLLM servers
--------------------

When multiple trainer ranks share a single vLLM HTTP server every request is
tagged with a ``client_tag`` so responses can be sharded. The helper derives a
stable identifier from ``Accelerate.process_index`` (falling back to
``RANK``/``WORLD_SIZE``) and automatically attaches it to:

1. the JSON payload (``client_tag`` field), and
2. the ``X-VLLM-Client-Tag`` HTTP header.

Override the value when needed by exporting ``VLLM_CLIENT_TAG`` before launching
the trainer. This is useful when several jobs share a proxy that fans out to
multiple vLLM backends and you want deterministic routing.

**Server requirement:** the vLLM server (or proxy) must echo the tag in its JSON
response. The client inspects ``results[*].metadata.client_tag`` and falls back
to nested ``outputs[*].metadata.client_tag``; any group/output whose tag does
not match the current rank is dropped before rewards/weighting run. If the echo
is missing the client discards every completion, leading to warnings such as
``Skipping policy loss group due to empty log-probs`` followed by zero losses
and ``train/kl`` = ``NaN``.

**Troubleshooting checklist**

- The vLLM warning ``vLLM raw groups=96 ...`` immediately followed by ``Score
  batch built | total_sequences=16`` indicates that one rank received all
  completions. Confirm that each HTTP response carries your tag, or run a
  per-rank vLLM worker to keep requests isolated.
- When filtering works you should see log entries like ``Filtered 80
  extraneous vLLM result groups for client_tag=rank-0`` exactly once per batch
  (DEBUG level) and the controller metrics ``train/delta_tau`` /
  ``train/delta_beta`` stop being pinned at zero.
- For ad-hoc debugging set ``VLLM_CLIENT_TAG`` to a descriptive value and enable
  ``MAXENT_GRPO_LOGLEVEL=DEBUG`` so the helper prints which tags were removed.

Rewards
-------

The default reward is exact‑match for math answers found within ``<answer>…</answer>`` tags. See: ``src/maxent_grpo/rewards.py`` for details and extension points.

CLI / YAML Config
-----------------

Most options can be provided by CLI via TRL’s ``TrlParser`` or by a YAML recipe.

- Example knobs: ``--system_prompt``, ``--chat_template``, ``--benchmarks``, W&B and Hub settings, MaxEnt controls (``--maxent_tau``, …)
- See API Reference → ``maxent_grpo.config`` for all fields.

Logging & Checkpoints
---------------------

- Slurm stdout/err live in ``var/artifacts/logs/<job-name>-<jobid>.out|err``.
- vLLM server logs to ``var/artifacts/logs/vllm-<jobid>.out`` (or inline when sharing a node).
- Trainer logs to ``var/artifacts/logs/train_<jobid>.log``.
- Run headers emit ``run/git_sha`` and ``run/recipe_path`` (from ``GRPO_RECIPE``/``GRPO_RECIPE_USED``) and the same keys are logged to metrics/W&B so cross-run comparisons stay schema-stable.
- Weights & Biases integration is available via ``wandb_*`` fields.
- Checkpoints are handled by TRL’s GRPOConfig as usual.
- When DeepSpeed is enabled we default to DeepSpeed’s native ``save_checkpoint``
  (mirroring HF Trainer) and fall back to Accelerate’s ``save_state`` only when
  explicitly allowed. Set ``MAXENT_SKIP_DEEPSPEED_STATE_SAVE=true`` to skip both,
  or ``MAXENT_ALLOW_DEEPSPEED_STATE_SAVE=true`` if you want to allow the fallback
  ``accelerator.save_state`` path after a DeepSpeed failure. To bypass DeepSpeed
  entirely and always use Accelerate, set
  ``MAXENT_PREFER_ACCELERATE_STATE_SAVE=true``.
- Periodic checkpoints in the Slurm recipe default to a *shallow* HF-style mode
  that writes DeepSpeed ZeRO shards plus small HF metadata and copies static
  config/tokenizer files from a base directory, but skips ``save_pretrained``
  on the model/tokenizer at every save. This avoids rewriting large
  ``model.safetensors`` files on each checkpoint and relies primarily on
  DeepSpeed + ``zero_to_fp32.py`` for weights. If you prefer standard HF/TRL
  behavior where each ``checkpoint-N`` is a standalone HF checkpoint, set
  ``MAXENT_CHECKPOINT_METADATA_MODE=full``. In shallow mode, periodic checkpoints:

  - still call the DeepSpeed/Accelerate state save, but
  - skip ``save_pretrained`` on the model/tokenizer and instead copy small static
    files (config/tokenizer/merges) from a base snapshot directory. By default
    this is inferred from ``output_dir`` (e.g., replacing ``MaxEnt-GRPO`` with
    ``GRPO`` for the sibling Open‑R1 GRPO run), but you can override it via
    ``MAXENT_CHECKPOINT_METADATA_SOURCE``.

  The final checkpoint always performs a full ``save_pretrained`` regardless of
  this setting.

Common Flags
------------

- ``--dataset_name`` or ``dataset_mixture`` (YAML) to define training data
- ``--system_prompt`` and ``--chat_template`` to control prompting
- ``--num_generations`` and ``--max_completion_length`` for candidate sampling
- ``--init_kl_coeff``, ``--kl_target``, ``--kl_horizon`` for trust region
- ``--report_to wandb`` plus ``wandb_*`` fields for logging
- InfoSeed extras (only when ``info_seed_enabled``): ``info_seed_num_seeds``, ``info_seed_lambda``, ``info_seed_temperature``, ``info_seed_prompt_template``, ``info_seed_loss_type`` – set ``info_seed_enabled=false`` (CLI or YAML) to drop the augmentation entirely.
- MaxEnt extras (when using ``src/maxent_grpo/maxent_grpo.py``): ``--maxent_tau``, ``--maxent_q_temperature``, ``--maxent_q_epsilon``, ``--maxent_length_normalize_ref``, plus optional controllers below.

Adaptive Controllers (MaxEnt)
-----------------------------

- **β (reverse KL) controller**: ``init_kl_coeff`` is the starting β. ``kl_target``,
  ``kl_horizon``, and ``kl_ctl_step_size`` enable the TRL-style multiplicative
  controller that increases β when measured KL exceeds the target and decreases
  otherwise. Leave the target/horizon at zero to disable adaptation.

- **τ (weight entropy) controller**: ``maxent_target_weight_entropy`` activates
  τ adaptation. ``maxent_tau_lr``, ``maxent_tau_min``, ``maxent_tau_max``, and
  ``maxent_tau_warmup_steps`` control the log-space optimizer, ensuring τ stays
  within bounds after an optional warmup. Omit ``maxent_target_weight_entropy``
  (or set it to null) to keep τ fixed.

- Controller persistence: β/τ values are stored in
  ``<output_dir>/controller_state.json``. When resuming via
  ``--resume_from_checkpoint``, the trainer restores that snapshot; otherwise the
  controllers restart from the recipe defaults.

Meta-Controller (τ/β)
---------------------

- Enable the meta-controller by setting ``controller_meta_enabled=true`` in your
  recipe (or ``--controller_meta_enabled true`` on TRL's CLI). The default mode
  uses a lightweight analytic update; flip ``controller_meta_method`` to
  ``first_order`` or ``truncated_backprop`` to run the meta-optimizer loop and
  differentiate through the controller loss.
- ``controller_meta_lr`` sets the meta learning rate. ``controller_meta_update_interval``
  controls how often meta steps run (every N policy steps). ``controller_meta_objective``
  is currently informational (the regularized potential is always optimized),
  ``controller_meta_optimizer`` chooses the meta optimizer (currently ``sgd``),
  ``controller_meta_truncation_steps`` (or ``controller_meta_analytic_steps``)
  controls the truncated horizon, and ``controller_meta_use_hessian`` toggles
  whether second-order approximations are permitted.
  Example Hydra overrides:

  .. code-block:: bash

     # Disable meta-controller in the MaxEnt Hydra template
     maxent-grpo-maxent maxent.training.controller_meta_enabled=false

     # Enable first-order meta updates for the baseline GRPO template
     maxent-grpo-baseline \
       baseline.training.controller_meta_enabled=true \
       baseline.training.controller_meta_method=first_order \
       baseline.training.controller_meta_lr=0.02
- When the meta-controller is enabled, the training loop caches per-step weight
  entropy and KL metrics, builds a controller objective (analytic or truncated
  backprop), and applies the resulting τ/β gradients through the meta-optimizer.
  The controller snapshot now includes the meta configuration so resuming runs
  with the flag turned on preserves the same behaviour. Metrics prefixed with
  ``train/meta/*`` record the latest meta loss, gradients, projected updates,
  and runtime configuration so dashboards can track the “strictly concave”
  controller dynamics alongside the main loss.
- Disable the meta-controller (or leave ``controller_meta_enabled=false``) to
  fall back to the standard EMA-based τ controller and KL feedback loop. In this
  mode the new CLI knobs are ignored and checkpoints omit meta diagnostics.

Troubleshooting
---------------

- vLLM not healthy / empty server log: verify GPUs were allocated (see ``nvidia-smi -L`` in Slurm log) and SBATCH GRES/partition match your cluster.
- GPU OOM early in training: reduce ``per_device_train_batch_size`` or ``num_generations``, enable ``gradient_checkpointing``.
- Very high KL: lower ``init_kl_coeff`` or raise ``kl_target``; verify chat template and system prompt.
- Slow data loading: pre‑filter columns via ``DatasetConfig.columns`` and disable unused features.
- RTD or local docs errors: ensure ``pip install -r docs/requirements.txt`` and rebuild with ``make docs``.
