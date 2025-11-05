Training
========

This guide covers the common flows for GRPO training.

One‑Command Training
--------------------

```bash
./training.sh
```

The launcher:

- Configures a local conda env under the repo
- Starts vLLM (for candidate generation when needed)
- Runs GRPO using TRL with your selected config/recipe

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

- Weights & Biases integration is available via `wandb_*` fields
- Checkpoints are handled by TRL’s GRPOConfig as usual

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

- GPU OOM early in training: reduce `per_device_train_batch_size` or `num_generations`, enable `gradient_checkpointing`.
- Very high KL: lower `init_kl_coeff` or raise `kl_target`; verify chat template and system prompt.
- Slow data loading: pre‑filter columns via `DatasetConfig.columns` and disable unused features.
- RTD or local docs errors: ensure `pip install -r docs/requirements.txt` and rebuild with `make docs`.
