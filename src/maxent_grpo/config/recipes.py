"""Helper to load YAML recipes via OmegaConf to dataclass instances."""

from __future__ import annotations

import os
from dataclasses import fields
from typing import Any, Dict, Tuple, Type

from .grpo import GRPOConfig, GRPOScriptArguments

try:  # pragma: no cover - optional dependency
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover
    OmegaConf = None  # type: ignore[assignment]


def _dataclass_field_names(cls: Type[Any]) -> set[str]:
    """Return dataclass field names for filtering dictionaries."""

    return {f.name for f in fields(cls)}


def _split_recipe_payload(
    payload: Dict[str, Any],
    model_cls: Type[Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Split a recipe dict into script/training/model sections."""

    script_fields = _dataclass_field_names(GRPOScriptArguments)
    training_fields = _dataclass_field_names(GRPOConfig)
    model_fields = set(getattr(model_cls, "__dataclass_fields__", {}).keys())

    script_kwargs: Dict[str, Any] = {}
    training_kwargs: Dict[str, Any] = {}
    model_kwargs: Dict[str, Any] = {}
    for key, value in payload.items():
        if key in script_fields:
            script_kwargs[key] = value
        elif key in training_fields:
            training_kwargs[key] = value
        elif not model_fields or key in model_fields:
            model_kwargs[key] = value
        else:
            training_kwargs[key] = value
    return script_kwargs, training_kwargs, model_kwargs


def load_grpo_recipe(
    recipe_path: str,
    *,
    model_config_cls: Type[Any],
) -> Tuple[GRPOScriptArguments, GRPOConfig, Any]:
    """Load a GRPO recipe YAML into config dataclasses.

    :param recipe_path: Path to the YAML recipe under ``configs/recipes``.
    :param model_config_cls: TRL ``ModelConfig`` class used for model kwargs.
    """

    if OmegaConf is None:
        raise ImportError(
            "OmegaConf is required to load recipe YAMLs. "
            "Install it via `pip install omegaconf` or use CLI args directly."
        )
    resolved_path = os.path.expanduser(recipe_path)
    cfg = OmegaConf.to_container(OmegaConf.load(resolved_path), resolve=True)
    if not isinstance(cfg, dict):
        raise ValueError(f"Recipe {recipe_path} did not resolve to a mapping.")

    script_kwargs, training_kwargs, model_kwargs = _split_recipe_payload(
        cfg, model_config_cls
    )
    return (
        GRPOScriptArguments(**script_kwargs),
        GRPOConfig(**training_kwargs),
        model_config_cls(**model_kwargs),
    )
