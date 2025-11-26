"""
Helper utilities to load YAML recipes into GRPO dataclasses.

Recipes are resolved with OmegaConf (or PyYAML when OmegaConf is unavailable),
split into script/training/model sections, and instantiated into the
corresponding config objects used by the training pipeline. Optional
dependencies are guarded to keep doc builds and unit tests lightweight.

License
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
"""

from __future__ import annotations

import os
from dataclasses import fields
from typing import Any, Dict, Tuple, Type

from .grpo import GRPOConfig, GRPOScriptArguments
from .defaults import INFO_SEED_DEFAULTS

try:  # pragma: no cover - optional dependency
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover
    OmegaConf = None
try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def _dataclass_field_names(cls: Type[Any]) -> set[str]:
    """Return dataclass field names for filtering dictionaries.

    :param cls: Dataclass type whose fields should be inspected.
    :returns: Set of field names defined on the dataclass.
    """

    return {f.name for f in fields(cls)}


def _split_recipe_payload(
    payload: Dict[str, Any],
    model_cls: Type[Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Split a recipe dict into script/training/model/other sections.

    :param payload: Raw mapping loaded from a recipe file.
    :param model_cls: Model config class used to route model kwargs.
    :returns: Tuple of script, training, model, and passthrough kwargs dictionaries.
    """

    script_fields = _dataclass_field_names(GRPOScriptArguments)
    training_fields = _dataclass_field_names(GRPOConfig)
    model_fields = set(getattr(model_cls, "__dataclass_fields__", {}).keys())

    script_kwargs: Dict[str, Any] = {}
    training_kwargs: Dict[str, Any] = {}
    model_kwargs: Dict[str, Any] = {}
    other_kwargs: Dict[str, Any] = {}
    for key, value in payload.items():
        if key in script_fields:
            script_kwargs[key] = value
        elif key in training_fields:
            training_kwargs[key] = value
        elif not model_fields or key in model_fields:
            model_kwargs[key] = value
        else:
            other_kwargs[key] = value
    return script_kwargs, training_kwargs, model_kwargs, other_kwargs


def load_grpo_recipe(
    recipe_path: str,
    *,
    model_config_cls: Type[Any],
) -> Tuple[GRPOScriptArguments, GRPOConfig, Any]:
    """Load a GRPO recipe YAML into config dataclasses.

    :param recipe_path: Path to the YAML recipe under ``configs/recipes``.
    :param model_config_cls: TRL ``ModelConfig`` class used for model kwargs.
    :returns: Tuple containing script arguments, training config, and model config.
    :raises ImportError: If neither OmegaConf nor PyYAML is available.
    :raises ValueError: If the resolved recipe payload is not a mapping.
    """

    resolved_path = os.path.expanduser(recipe_path)
    if OmegaConf is not None:
        cfg = OmegaConf.to_container(OmegaConf.load(resolved_path), resolve=True)
    elif yaml is not None:
        with open(resolved_path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
    else:
        raise ImportError("OmegaConf or PyYAML is required to load recipe YAMLs.")

    if not isinstance(cfg, dict):
        raise ValueError(f"Recipe {recipe_path} did not resolve to a mapping.")

    # Persist the resolved path so downstream logging can surface it consistently.
    os.environ.setdefault("GRPO_RECIPE_USED", resolved_path)

    (
        script_kwargs,
        training_kwargs,
        model_kwargs,
        other_kwargs,
    ) = _split_recipe_payload(cfg, model_config_cls)
    for key, default_val in INFO_SEED_DEFAULTS.items():
        training_kwargs.setdefault(key, other_kwargs.get(key, default_val))
    script_args = GRPOScriptArguments(**script_kwargs)
    training_args = GRPOConfig(**training_kwargs)
    model_args = model_config_cls(**model_kwargs)
    # Attach the recipe path for logging/telemetry consumers (best-effort).
    for obj in (script_args, training_args, model_args):
        try:
            setattr(obj, "recipe_path", resolved_path)
        except (AttributeError, TypeError):
            pass
    return (script_args, training_args, model_args)
