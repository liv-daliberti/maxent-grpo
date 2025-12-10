"""Pydantic-powered validation for Hydra training configs.

This module inspects the resolved training arguments before a pipeline is
launched so accidental MaxEnt toggles are caught early. The validator is kept
lightweight and only depends on :mod:`pydantic`, which is already part of the
runtime toolchain for several other components. Future guardrails can extend
this module by adding additional schema checks.
"""

from __future__ import annotations

import warnings
from dataclasses import MISSING, asdict, fields, is_dataclass
from typing import Any, Mapping, MutableMapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    ValidationError,
    model_validator,
)
from pydantic.warnings import UnsupportedFieldAttributeWarning

from maxent_grpo.config import GRPOConfig
from maxent_grpo.pipelines.inference.inference import resolve_inference_dataset

__all__ = [
    "validate_training_config",
    "validate_generation_config",
    "validate_inference_config",
]

warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)


def _field_default(field) -> Any:
    if field.default is not MISSING:
        return field.default
    if field.default_factory is not MISSING:  # type: ignore[attr-defined]
        return field.default_factory()  # type: ignore[misc]
    return None


_MAXENT_DEFAULTS = {
    field.name: _field_default(field)
    for field in fields(GRPOConfig)
    if field.name.startswith("maxent_")
}

_INFO_SEED_DEFAULTS = {
    field.name: _field_default(field)
    for field in fields(GRPOConfig)
    if field.name.startswith("info_seed_") and field.name != "info_seed_enabled"
}

_DEFAULT_OBJECTIVE_BY_COMMAND = {
    "train-baseline": True,
    "train-infoseed": True,
    "train-maxent": False,
}

_INFO_SEED_SUPPORTED = {"train-maxent", "train-infoseed"}


class _TrainingSchema(BaseModel):
    """Minimal schema capturing the knobs that need cross-field validation."""

    model_config = ConfigDict(extra="forbid")

    train_grpo_objective: bool | None = None
    default_objective: bool | None = Field(default=None)
    maxent_overrides: dict[str, Any] = Field(default_factory=dict)
    info_seed_enabled: bool | None = None
    info_seed_overrides: dict[str, Any] = Field(default_factory=dict)
    allow_info_seed: bool = True
    require_info_seed: bool = False

    @model_validator(mode="after")
    def _check_maxent_conflicts(self) -> "_TrainingSchema":
        effective = (
            self.train_grpo_objective
            if self.train_grpo_objective is not None
            else self.default_objective
        )
        if effective is not False and self.maxent_overrides:
            knobs = ", ".join(sorted(self.maxent_overrides))
            raise ValueError(
                "MaxEnt overrides (%s) require train_grpo_objective=false" % knobs
            )
        if not self.allow_info_seed and self.info_seed_enabled:
            raise ValueError("InfoSeed is not supported for this command")
        if self.require_info_seed and not self.info_seed_enabled:
            raise ValueError("train-infoseed requires info_seed_enabled=true")
        if (not self.info_seed_enabled) and self.info_seed_overrides:
            knobs = ", ".join(sorted(self.info_seed_overrides))
            raise ValueError(
                "InfoSeed overrides (%s) require info_seed_enabled=true" % knobs
            )
        return self


class _GenerationSchema(BaseModel):
    """Schema enforcing minimal requirements for generation jobs."""

    model_config = ConfigDict(extra="allow")

    hf_dataset: str
    model: str
    vllm_server_url: str = "http://localhost:8000/v1"
    num_generations: PositiveInt = 1
    max_new_tokens: PositiveInt = 1
    input_batch_size: PositiveInt = 1
    client_replicas: PositiveInt = 1

    @model_validator(mode="after")
    def _check_url(self) -> "_GenerationSchema":
        if not str(self.vllm_server_url).strip():
            raise ValueError("vllm_server_url must be provided for generation jobs")
        return self


class _InferenceModelSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    model_name_or_path: str


class _InferenceSchema(BaseModel):
    """Schema validating inference command payloads."""

    model_config = ConfigDict(extra="allow")

    models: list[_InferenceModelSchema]
    dataset: str = "math_500"
    eval: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_dataset(self) -> "_InferenceSchema":
        if not self.models:
            raise ValueError("inference.models must contain at least one model spec")
        try:
            resolve_inference_dataset(self.dataset, self.eval)
        except ValueError as exc:  # pragma: no cover - errors exercised in tests
            raise ValueError(str(exc)) from exc
        return self


def _training_values(payload: Any) -> MutableMapping[str, Any]:
    """Return a mapping containing the knobs relevant to validation."""

    if isinstance(payload, Mapping):
        return {key: payload[key] for key in payload}
    values: MutableMapping[str, Any] = {}
    attr_names = set(_MAXENT_DEFAULTS)
    attr_names |= {"train_grpo_objective", "info_seed_enabled"}
    attr_names |= set(_INFO_SEED_DEFAULTS)
    for name in attr_names:
        if hasattr(payload, name):
            values[name] = getattr(payload, name)
    return values


def _maxent_overrides(values: Mapping[str, Any]) -> dict[str, Any]:
    """Return MaxEnt fields whose values differ from their defaults."""

    overrides: dict[str, Any] = {}
    for name, default in _MAXENT_DEFAULTS.items():
        if name not in values:
            continue
        value = values[name]
        if value is None and default is None:
            continue
        if value == default:
            continue
        overrides[name] = value
    return overrides


def _info_seed_overrides(values: Mapping[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for name, default in _INFO_SEED_DEFAULTS.items():
        if name not in values:
            continue
        value = values[name]
        if value == default:
            continue
        overrides[name] = value
    return overrides


def _payload_mapping(payload: Any) -> MutableMapping[str, Any]:
    if isinstance(payload, Mapping):
        return {key: payload[key] for key in payload}
    if is_dataclass(payload):
        return asdict(payload)
    if hasattr(payload, "__dict__"):
        return dict(payload.__dict__)
    return {}


def _source_hint(command: str, *, recipe: str | None, training_args: Any) -> str:
    """Return a short string pointing at the config origin for error messages."""

    hints: list[str] = [command]
    recipe_path = recipe or getattr(training_args, "recipe_path", None)
    if recipe_path:
        hints.append(str(recipe_path))
    return " | ".join(hints)


def _command_hint(command: str, source: str | None = None) -> str:
    hints = [command]
    if source:
        hints.append(source)
    return " | ".join(hints)


def _format_validation_errors(errors: list[dict[str, Any]]) -> str:
    parts = []
    for error in errors:
        loc = error.get("loc") or ()
        if isinstance(loc, tuple):
            path = ".".join(str(item) for item in loc if item is not None)
        else:
            path = str(loc)
        if path:
            parts.append(f"{path}: {error.get('msg', '')}")
        else:
            parts.append(error.get("msg", ""))
    return "; ".join(parts)


def validate_training_config(
    training_args: GRPOConfig | Mapping[str, Any],
    *,
    command: str,
    source: str | None = None,
) -> None:
    """Validate Hydrated training knobs before dispatching to a pipeline.

    The validator ensures that the GRPO objective flag matches the presence of
    MaxEnt-specific options. When MaxEnt knobs are supplied while
    ``train_grpo_objective`` resolves to ``True`` (the vanilla GRPO objective), a
    :class:`ValueError` is raised so the job fails fast.

    :param training_args: Training dataclass or mapping derived from Hydra.
    :param command: CLI command being executed (e.g., ``train-baseline``).
    :param source: Optional user-facing hint (recipe path, override description).
    :raises ValueError: If incompatible knob combinations are detected.
    """

    values = _training_values(training_args)
    schema_payload = {
        "train_grpo_objective": values.get("train_grpo_objective"),
        "default_objective": _DEFAULT_OBJECTIVE_BY_COMMAND.get(command),
        "maxent_overrides": _maxent_overrides(values),
        "info_seed_enabled": values.get("info_seed_enabled"),
        "info_seed_overrides": _info_seed_overrides(values),
        "allow_info_seed": command in _INFO_SEED_SUPPORTED,
        "require_info_seed": command == "train-infoseed",
    }
    try:
        _TrainingSchema(**schema_payload)
    except ValidationError as exc:
        message = _format_validation_errors(exc.errors())
        hint = _source_hint(command, recipe=source, training_args=training_args)
        raise ValueError(f"{hint}: {message}") from exc


def validate_generation_config(
    config: Mapping[str, Any] | Any,
    *,
    command: str = "generate",
    source: str | None = None,
) -> None:
    """Validate generation command payloads before invoking pipelines."""

    values = _payload_mapping(config)
    try:
        _GenerationSchema(**values)
    except ValidationError as exc:
        message = _format_validation_errors(exc.errors())
        hint = _command_hint(command, source)
        raise ValueError(f"{hint}: {message}") from exc


def validate_inference_config(
    config: Mapping[str, Any] | Any,
    *,
    command: str = "inference",
    source: str | None = None,
) -> None:
    """Validate inference command payloads prior to execution."""

    values = _payload_mapping(config)
    try:
        _InferenceSchema(**values)
    except ValidationError as exc:
        message = _format_validation_errors(exc.errors())
        hint = _command_hint(command, source)
        raise ValueError(f"{hint}: {message}") from exc
