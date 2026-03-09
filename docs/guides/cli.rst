CLI Usage
=========

The project exposes a single Hydra CLI surface focused on training:

- ``maxent-grpo``: top-level CLI (set ``command=...`` explicitly).
- ``maxent-grpo-baseline``: convenience wrapper for ``command=train-baseline``.

Command Routing
---------------

Supported commands:

- ``train-baseline``: baseline GRPO training.
- ``train-maxent``: MaxEnt-GRPO training.

Recipes and Overrides
---------------------

Training commands can load YAML recipes via:

- ``$GRPO_RECIPE`` (environment variable), or
- ``baseline.recipe=...`` / ``maxent.recipe=...`` command fields.

After loading a recipe, overrides are applied from command-specific
``script``/``training``/``model`` sections.

.. code-block:: bash

   GRPO_RECIPE=configs/recipes/Qwen2.5-1.5B-Instruct/grpo/config_math.yaml \
     maxent-grpo-baseline baseline.training.output_dir=var/data/out

.. code-block:: bash

   maxent-grpo command=train-maxent \
     maxent.recipe=configs/recipes/Qwen2.5-1.5B-Instruct/maxent-grpo/config_math.yaml \
     maxent.training.maxent_tau=0.2

Coding pipeline example (MBPP + test-based reward):

.. code-block:: bash

   maxent-grpo-baseline \
     baseline.recipe=configs/recipes/Qwen2.5-0.5B-Instruct/grpo/config_code_mbpp.yaml

Validation
----------

Before launch, ``maxent_grpo.cli.config_validation`` ensures MaxEnt overrides
are only used with ``train_grpo_objective=false`` except for GRPO + entropy-bonus
runs where ``policy_entropy_bonus_coef>0``.

Examples
--------

Hydra recipe presets live under ``configs/recipes/hydra/``.
For custom-loop GRPO parity runs, use
``configs/recipes/hydra/grpo_custom_math.yaml``.
