Recipes
=======

Use YAML recipes to keep runs reproducible and readable. The configs mirror the dataclasses in ``maxent_grpo.config`` and TRL’s GRPO settings.

Recipe Layout
-------------

Recipes are **flat** YAML mappings. Fields are routed automatically into three
config objects based on their dataclass field names:

- ``GRPOScriptArguments`` (dataset, evaluation, reward-related script knobs)
- ``GRPOConfig`` (training, MaxEnt/InfoSeed, vLLM, logging)
- TRL ``ModelConfig`` (model name, dtype, revision, etc.)

Keys that do not match script or training fields are forwarded to the TRL
``ModelConfig`` when possible. Any remaining keys are ignored (aside from a few
compatibility aliases described below).

Minimal example:

.. code-block:: yaml

   model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct
   dataset_name: open-r1/OpenR1-Math-220k
   output_dir: var/data/out
   train_grpo_objective: false
   maxent_tau: 0.2

Loading and Validation
----------------------

``maxent_grpo.config.load_grpo_recipe`` loads YAML with OmegaConf (or PyYAML),
sets ``GRPO_RECIPE_USED`` to the resolved path, and applies a few compatibility
rules:

- ``init_kl_coeff`` / ``init_kl_coef`` / ``kl_penalty_beta`` map to ``beta`` if
  ``beta`` is not set.
- When ``use_vllm: true`` and ``vllm_mode: server``, missing
  ``vllm_server_base_url`` / host / port are inferred from ``vllm_url``.
- ``MAXENT_LOG_LEVEL`` overrides ``log_level`` in the training config.

For flat recipes (no top-level ``script`` / ``training`` / ``model`` keys),
schema validation enforces:

- Baseline recipes must set ``beta`` or a KL alias.
- MaxEnt recipes must set ``train_grpo_objective: false``.
- InfoSeed recipes must set ``info_seed_enabled: true``.

Validation is skipped during tests and only applies to flat recipe files (not
Hydra configs).

Dataset Mixtures
----------------

To blend multiple datasets, set ``dataset_mixture`` instead of ``dataset_name``:

.. code-block:: yaml

   dataset_mixture:
     seed: 42
     test_split_size: 0.02
     datasets:
       - id: open-r1/OpenR1-Math-220k
         split: train
         columns: [problem, answer]
         weight: 1.0
       - id: some/other-dataset
         split: train
         weight: 0.5

Each dataset can define a split, optional columns, and a sampling weight. The
mixture loader validates column consistency and can carve out a test split on
the combined dataset.

Math GRPO (Qwen 1.5B)
---------------------

.. literalinclude:: ../configs/recipes/Qwen2.5-1.5B-Instruct/grpo/config_math.yaml
   :language: yaml
   :linenos:

Paired Recipes (GRPO vs MaxEnt)
-------------------------------

For reproducible comparisons, each model family ships **paired** GRPO and
MaxEnt recipes that keep sampling, optimizer, and evaluation settings aligned.
Use the GRPO recipe under ``grpo/`` and the MaxEnt counterpart under
``maxent-grpo/``:

- GRPO: ``configs/recipes/<model>/grpo/config_math.yaml``
- MaxEnt: ``configs/recipes/<model>/maxent-grpo/config_math.yaml``

Paired GRPO recipes set ``force_custom_loop: true`` so GRPO runs through the
same custom loop used by MaxEnt, and they pin
``maxent_reference_logprobs_source: model`` so both objectives use a frozen
reference anchor for KL.

Tips
----

- Adjust ``num_generations`` and ``max_completion_length`` to trade off speed vs. diversity
- Set ``hub_model_id`` to point at your namespace
- Toggle ``use_vllm`` depending on your setup

Hydra recipes
-------------

- Baseline: ``configs/recipes/hydra/baseline_math.yaml``
- GRPO (custom loop parity): ``configs/recipes/hydra/grpo_custom_math.yaml``
- MaxEnt-GRPO: ``configs/recipes/hydra/maxent_math.yaml``
- InfoSeed: ``configs/recipes/hydra/infoseed_math.yaml``

Hydra configs bundle ``command=...`` with a recipe path and optional overrides
under ``baseline`` / ``maxent`` / ``infoseed``. They are a convenient way to
share fully-specified CLI runs without long command lines.
