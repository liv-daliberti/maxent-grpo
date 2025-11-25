Recipes
=======

Use YAML recipes to keep runs reproducible and readable. The configs mirror the dataclasses in ``maxent_grpo.config`` and TRL’s GRPO settings.

Math GRPO (Qwen 1.5B)
---------------------

.. literalinclude:: ../configs/recipes/Qwen2.5-1.5B-Instruct/grpo/config_math.yaml
   :language: yaml
   :linenos:

Tips
----

- Adjust ``num_generations`` and ``max_completion_length`` to trade off speed vs. diversity
- Set ``hub_model_id`` to point at your namespace
- Toggle ``use_vllm`` depending on your setup

Hydra recipes
-------------

- Baseline: ``configs/recipes/hydra/baseline_math.yaml``
- MaxEnt-GRPO: ``configs/recipes/hydra/maxent_math.yaml``
