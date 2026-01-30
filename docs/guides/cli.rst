CLI Usage
=========

The project ships two user-facing CLIs:

- A Hydra multi-command CLI in ``maxent_grpo.cli.hydra_cli`` (training, inference,
  generation) exposed via console scripts.
- A lightweight generation CLI in ``maxent_grpo.generate`` for the distilabel
  pipeline (Typer + argparse).

Entry Points
------------

Console scripts are defined in ``pyproject.toml``:

- ``maxent-grpo``: top-level Hydra CLI (requires ``command=...``).
- ``maxent-grpo-baseline``: wrapper for ``command=train-baseline``.
- ``maxent-grpo-maxent``: wrapper for ``command=train-maxent``.
- ``maxent-grpo-infoseed``: wrapper for ``command=train-infoseed``.
- ``maxent-grpo-generate``: wrapper for ``command=generate``.
- ``maxent-grpo-inference``: wrapper for ``command=inference``.
- ``maxent-grpo-math-eval``: wrapper for ``command=math-eval``.

Command Routing
---------------

The Hydra CLI accepts these commands:

- ``train-baseline``: baseline GRPO training.
- ``train-maxent``: MaxEnt-GRPO training.
- ``train-infoseed``: InfoSeed training.
- ``generate``: distilabel generation pipeline.
- ``inference`` / ``math-eval``: math inference evaluation.

Wrapper scripts inject the matching ``command=...`` argument for convenience.
If you use ``maxent-grpo`` directly, include the command explicitly.

Recipes and Overrides
---------------------

Training commands can load a YAML recipe via:

- ``$GRPO_RECIPE`` (environment variable), or
- ``baseline.recipe=...`` / ``maxent.recipe=...`` / ``infoseed.recipe=...``
  (command-specific field).

After loading a recipe, overrides are applied from the command-specific
``script``/``training``/``model`` sections. For example, with the baseline
wrapper:

.. code-block:: bash

   GRPO_RECIPE=configs/recipes/Qwen2.5-1.5B-Instruct/grpo/config_math.yaml \
     maxent-grpo-baseline baseline.training.output_dir=var/data/out

Or using the top-level CLI:

.. code-block:: bash

   maxent-grpo command=train-maxent \
     maxent.recipe=configs/recipes/Qwen2.5-1.5B-Instruct/maxent-grpo/config_math.yaml \
     maxent.training.maxent_tau=0.2

Generation and inference use the ``generate`` / ``inference`` config blocks:

.. code-block:: bash

   maxent-grpo command=generate \
     generate.args.hf_dataset=your/dataset \
     generate.args.model=Qwen/Qwen2.5-1.5B-Instruct \
     generate.args.vllm_server_url=http://localhost:8000/v1

If you prefer a dedicated generation CLI, use:

.. code-block:: bash

   python -m maxent_grpo.generate --help

Validation
----------

Before launching, ``maxent_grpo.cli.config_validation`` checks that:

- MaxEnt-specific overrides are only used with ``train_grpo_objective=false``.
- InfoSeed overrides require ``info_seed_enabled=true``.
- Generation and inference inputs meet minimal requirements (e.g., vLLM URL).

Examples
--------

Hydra recipe presets live under ``configs/recipes/hydra/`` and show how to
combine a recipe path with command-specific overrides.

For custom-loop GRPO parity runs, use the dedicated preset:
``configs/recipes/hydra/grpo_custom_math.yaml``.
