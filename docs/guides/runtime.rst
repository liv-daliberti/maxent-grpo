Runtime Setup
=============

This page summarizes runtime dependencies and the main environment knobs used
by MaxEnt-GRPO.

Dependencies
------------

Training requires the usual ML stack (torch, transformers, trl, datasets) plus
Accelerate for multi-GPU launches. Optional integrations include:

- DeepSpeed (ZeRO and large-scale training)
- vLLM (external generation server)
- Weights & Biases (logging)
- peft / bitsandbytes (adapter or quantized training)

Optional imports are guarded in ``maxent_grpo.utils.imports`` and
``maxent_grpo.training.runtime`` so missing dependencies raise a descriptive
error rather than a generic ``ImportError``.

Accelerate and DeepSpeed
------------------------

The Slurm launcher ``ops/slurm/train.slurm`` expects an Accelerate config file
under ``configs/recipes/accelerate_configs/<name>.yaml``. The ``--accelerator``
flag in the launcher selects the file, and the resulting config controls
process counts, mixed precision, and DeepSpeed integration.

vLLM Connectivity
-----------------

To generate with vLLM during training:

- Set ``use_vllm: true`` and ``vllm_mode: server`` in the recipe.
- Provide a ``vllm_url`` (e.g., ``http://host:port/generate``).

When these are set, the recipe loader infers ``vllm_server_base_url`` (plus host
and port) from ``vllm_url`` if the server-specific fields are omitted.

For standalone generation, use the distilabel CLI:

.. code-block:: bash

   python -m maxent_grpo.generate --help

Environment Variables
---------------------

Common overrides you may want to set:

- ``GRPO_RECIPE``: default recipe path for the Hydra CLI.
- ``MAXENT_LOG_LEVEL``: overrides ``log_level`` in training configs.
- ``MAXENT_TAU`` / ``MAXENT_Q_TEMPERATURE`` / ``MAXENT_Q_EPS`` /
  ``MAXENT_LENGTH_NORM_REF``: defaults for MaxEnt runtime options.
- ``MAXENT_DATASET_CACHE_DIR``: base directory for cached dataset transforms.
- ``MAXENT_HF_DATASET_RETRIES`` / ``MAXENT_HF_DATASET_RETRY_SLEEP`` /
  ``MAXENT_HF_DATASET_RETRY_MAX_SLEEP``: dataset download retry policy.
- ``MAXENT_FAULTHANDLER``: enable Python faulthandler in the MaxEnt entrypoint.

There are additional toggles throughout the codebase; search for ``MAXENT_`` in
``src/maxent_grpo`` to discover them.
