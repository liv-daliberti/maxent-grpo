"""
Copyright 2025 Liv d'Aliberti

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Helpers for parsing TRL-powered CLI arguments.
"""

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