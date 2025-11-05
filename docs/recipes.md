Recipes
=======

Use YAML recipes to keep runs reproducible and readable. The configs mirror the dataclasses in `src/configs.py` and TRL’s GRPO settings.

Math GRPO (Qwen 1.5B)
---------------------

```{literalinclude} ../recipes/Qwen2.5-1.5B-Instruct/grpo/config_math.yaml
:language: yaml
:linenos:
```

Tips
----

- Adjust `num_generations` and `max_completion_length` to trade off speed vs. diversity
- Set `hub_model_id` to point at your namespace
- Toggle `use_vllm` depending on your setup

