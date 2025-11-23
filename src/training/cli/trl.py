"""Helpers for parsing TRL-powered CLI arguments."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Tuple

from maxent_grpo.config import GRPOConfig, GRPOScriptArguments, load_grpo_recipe

if TYPE_CHECKING:
    from trl import ModelConfig


def parse_grpo_args(
    recipe_path: str | None = None,
) -> Tuple[GRPOScriptArguments, GRPOConfig, ModelConfig]:
    """Parse GRPO CLI arguments or load them from a YAML recipe.

    When ``recipe_path`` (or ``$GRPO_RECIPE``) is provided, the YAML is loaded
    via OmegaConf and converted into config dataclasses so orchestration code
    remains recipe-agnostic.
    """
    recipe_path = recipe_path or os.environ.get("GRPO_RECIPE")
    try:  # pragma: no cover - optional dependency for CLI
        from trl import ModelConfig, TrlParser
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "Parsing GRPO configs requires TRL. Install it via `pip install trl`."
        ) from exc
    if recipe_path:
        return load_grpo_recipe(recipe_path, model_config_cls=ModelConfig)
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    return parser.parse_args_and_config()


__all__ = ["parse_grpo_args"]
