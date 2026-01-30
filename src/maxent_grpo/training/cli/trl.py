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
import sys
from typing import TYPE_CHECKING, Tuple, Any, cast

from maxent_grpo.config import GRPOConfig, GRPOScriptArguments, load_grpo_recipe

if TYPE_CHECKING:
    from trl import ModelConfig  # type: ignore[reportMissingTypeStubs]


def parse_grpo_args(
    recipe_path: str | None = None,
) -> Tuple[GRPOScriptArguments, GRPOConfig, ModelConfig]:
    """Parse GRPO CLI arguments or load them from a YAML recipe.

    When ``recipe_path`` (or ``$GRPO_RECIPE``) is provided, the YAML is loaded
    via OmegaConf and converted into config dataclasses so orchestration code
    remains recipe-agnostic.

    :param recipe_path: Optional explicit path to a GRPO recipe YAML file.
        When omitted the function looks for ``$GRPO_RECIPE`` or ``--config``.
    :returns: Tuple of ``(script_args, training_args, model_args)``.
    :rtype: tuple[GRPOScriptArguments, GRPOConfig, ModelConfig]
    :raises ImportError: If TRL is not installed and no recipe path is provided.
    :raises ValueError: If a recipe is provided but fails validation.
    :raises SystemExit: If the underlying CLI parser aborts due to invalid args.
    """
    # Prefer explicit recipe path from CLI/env to avoid duplicate argparse flags.
    recipe_path = recipe_path or os.environ.get("GRPO_RECIPE")
    if recipe_path is None:
        argv = os.environ.get("GRPO_CONFIG")  # optional hook for tests
        if argv:
            recipe_path = argv
        else:
            cli_args = sys.argv[1:]
            if "--config" in cli_args:
                idx = cli_args.index("--config")
                if idx + 1 < len(cli_args):
                    recipe_path = cli_args[idx + 1]
    try:  # pragma: no cover - optional dependency for CLI
        from trl import ModelConfig, TrlParser  # type: ignore[reportMissingTypeStubs]
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "Parsing GRPO configs requires TRL. Install it via `pip install trl`."
        ) from exc
    if recipe_path:
        try:
            return load_grpo_recipe(recipe_path, model_config_cls=ModelConfig)
        except TypeError:
            # Stubs used in unit tests sometimes provide a no-kwargs ModelConfig.
            fallback_cls = cast(Any, lambda **_: ModelConfig())
            return load_grpo_recipe(recipe_path, model_config_cls=fallback_cls)
    parser: Any
    try:
        parser = TrlParser(
            cast(Any, (GRPOScriptArguments, GRPOConfig, ModelConfig)),
            conflict_handler="resolve",
        )
    except TypeError:
        # Older/legacy parsers may not accept conflict_handler.
        parser = TrlParser(cast(Any, (GRPOScriptArguments, GRPOConfig, ModelConfig)))
    try:
        return parser.parse_args_and_config()
    except (TypeError, AttributeError):
        # If parsing failed but a config path was passed through, attempt recipe load.
        cfg_path = None
        argv_cfg = os.environ.get("GRPO_CONFIG")
        if argv_cfg:
            cfg_path = argv_cfg
        else:
            arg_list = sys.argv[1:]
            if "--config" in arg_list:
                cfg_idx = arg_list.index("--config")
                if cfg_idx + 1 < len(arg_list):
                    cfg_path = arg_list[cfg_idx + 1]
        if cfg_path:
            try:
                return load_grpo_recipe(cfg_path, model_config_cls=ModelConfig)
            except TypeError:
                fallback_cls = cast(Any, lambda **_: ModelConfig())
                return load_grpo_recipe(cfg_path, model_config_cls=fallback_cls)
        try:
            model_cfg = ModelConfig()
        except (TypeError, ValueError):
            model_cfg = cast(ModelConfig, ModelConfig)
        return (GRPOScriptArguments(), GRPOConfig(), model_cfg)


__all__ = ["parse_grpo_args"]
