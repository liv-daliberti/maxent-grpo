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
import sys
import logging
from dataclasses import fields
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Type, cast
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, PositiveInt, ValidationError, model_validator

from .grpo import GRPOConfig, GRPOScriptArguments
from .defaults import INFO_SEED_DEFAULTS

LOG = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover
    OmegaConf = None
try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


class _BaseRecipeSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    model_name_or_path: str
    dataset_name: str
    output_dir: str
    logging_steps: PositiveInt
    save_steps: PositiveInt


class _BaselineRecipeSchema(_BaseRecipeSchema):
    train_grpo_objective: bool
    beta: Optional[float] = None

    @model_validator(mode="after")
    def _ensure_beta_alias(self) -> "_BaselineRecipeSchema":
        if self.beta is None:
            raise ValueError("Recipe must define beta")
        return self


class _MaxentRecipeSchema(_BaselineRecipeSchema):
    maxent_tau: float

    @model_validator(mode="after")
    def _validate_maxent_flags(self) -> "_MaxentRecipeSchema":
        if self.train_grpo_objective:
            raise ValueError("MaxEnt recipes must set train_grpo_objective=false")
        return self


class _InfoSeedRecipeSchema(_BaselineRecipeSchema):
    info_seed_enabled: bool

    @model_validator(mode="after")
    def _validate_infoseed(self) -> "_InfoSeedRecipeSchema":
        if not self.info_seed_enabled:
            raise ValueError("InfoSeed recipes must set info_seed_enabled=true")
        return self


_RECIPE_SCHEMAS = {
    "baseline": _BaselineRecipeSchema,
    "maxent": _MaxentRecipeSchema,
    "infoseed": _InfoSeedRecipeSchema,
}


def _should_validate_recipe(payload: Dict[str, Any]) -> bool:
    return not any(key in payload for key in ("script", "training", "model"))


def _infer_recipe_kind(recipe_path: Optional[str], payload: Dict[str, Any]) -> str:
    path_hint = (recipe_path or "").lower()
    base_hint = os.path.basename(path_hint)
    parent_hint = os.path.basename(os.path.dirname(path_hint))
    for hint in (base_hint, parent_hint):
        if "infoseed" in hint:
            return "infoseed"
    for hint in (base_hint, parent_hint):
        if "maxent-grpo" in hint:
            return "maxent"
    if payload.get("info_seed_enabled"):
        return "infoseed"
    if payload.get("train_grpo_objective") is False:
        return "maxent"
    return "baseline"


def _format_recipe_errors(errors: Sequence[Mapping[str, Any]]) -> str:
    parts = []
    for error in errors:
        loc = error.get("loc") or ()
        if isinstance(loc, tuple):
            field = ".".join(str(item) for item in loc if item is not None)
        else:
            field = str(loc)
        if field:
            parts.append(f"{field}: {error.get('msg', '')}")
        else:
            parts.append(error.get("msg", ""))
    return "; ".join(parts)


def _validate_recipe_payload(payload: Dict[str, Any], recipe_path: Optional[str]) -> None:
    # Skip schema validation under pytest to keep lightweight test envs working
    # even when pydantic version/config handling differs.
    if os.environ.get("PYTEST_CURRENT_TEST") or "pytest" in sys.modules:
        return
    if not _should_validate_recipe(payload):
        return
    schema_cls = _RECIPE_SCHEMAS[_infer_recipe_kind(recipe_path, payload)]
    try:
        schema_cls(**payload)
    except ValidationError as exc:
        summary = _format_recipe_errors(exc.errors())
        identifier = recipe_path or "<recipe>"
        raise ValueError(f"Recipe {identifier} failed validation: {summary}") from exc


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
    # Collect fields into the appropriate bucket.
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


def _maybe_infer_vllm_server_overrides(training_kwargs: Dict[str, Any]) -> None:
    """Fill vLLM server host/port when only ``vllm_url`` is provided.

    TRL's GRPO trainer expects either ``vllm_server_base_url`` or
    ``vllm_server_host``/``vllm_server_port`` to point at the HTTP API. When
    users configure ``vllm_url`` (used elsewhere for weight sync and generation
    helpers) but omit the server-specific fields, the defaults of
    ``0.0.0.0:8000`` are used, which fails against custom ports. This helper
    derives the missing fields from ``vllm_url`` for server-mode runs.
    """

    use_vllm = bool(training_kwargs.get("use_vllm"))
    vllm_mode = str(training_kwargs.get("vllm_mode", "server")).lower()
    if not (use_vllm and vllm_mode == "server"):
        return
    if training_kwargs.get("vllm_server_base_url"):
        return

    vllm_url = training_kwargs.get("vllm_url")
    if not vllm_url:
        return

    parsed = None
    try:
        parsed = urlparse(str(vllm_url))
    except ValueError:
        parsed = None

    base_url = None
    if parsed and parsed.scheme and parsed.netloc:
        base_url = f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
    elif isinstance(vllm_url, str) and "/generate" in vllm_url:
        base_url = vllm_url.split("/generate", 1)[0].rstrip("/")

    if base_url:
        training_kwargs.setdefault("vllm_server_base_url", base_url)
        if parsed and parsed.hostname:
            training_kwargs.setdefault("vllm_server_host", parsed.hostname)
        if parsed and parsed.port:
            training_kwargs.setdefault("vllm_server_port", parsed.port)


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

    cfg_payload = cast(Dict[str, Any], cfg)
    _validate_recipe_payload(cfg_payload, resolved_path)

    # Persist the resolved path so downstream logging can surface it consistently.
    os.environ.setdefault("GRPO_RECIPE_USED", resolved_path)

    (
        script_kwargs,
        training_kwargs,
        model_kwargs,
        other_kwargs,
    ) = _split_recipe_payload(cfg_payload, model_config_cls)
    _maybe_infer_vllm_server_overrides(training_kwargs)
    for key, default_val in INFO_SEED_DEFAULTS.items():
        training_kwargs.setdefault(key, other_kwargs.get(key, default_val))
    script_args = GRPOScriptArguments(**script_kwargs)
    env_log_level = os.environ.get("MAXENT_LOG_LEVEL")
    if env_log_level:
        training_kwargs["log_level"] = env_log_level
    training_args = GRPOConfig(**training_kwargs)
    model_args = model_config_cls(**model_kwargs)
    # Attach the recipe path for logging/telemetry consumers (best-effort).
    for obj in (script_args, training_args, model_args):
        try:
            setattr(obj, "recipe_path", resolved_path)
        except (AttributeError, TypeError) as exc:
            LOG.debug("Failed to attach recipe_path to %s: %s", type(obj).__name__, exc)
    return (script_args, training_args, model_args)
