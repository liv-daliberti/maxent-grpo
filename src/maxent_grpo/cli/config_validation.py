"""Pydantic-powered validation for Hydra training configs.

This module inspects the resolved training arguments before a pipeline is
launched so accidental MaxEnt toggles are caught early. The validator is kept
lightweight and only depends on :mod:`pydantic`, which is already part of the
runtime toolchain for several other components. Future guardrails can extend
this module by adding additional schema checks.
"""

from __future__ import annotations

import warnings
import importlib
import inspect
from dataclasses import MISSING, asdict, fields, is_dataclass
from types import SimpleNamespace
from typing import Any, Mapping, MutableMapping


class _FallbackBaseModel:  # pragma: no cover - used when pydantic is missing
    """Minimal stub for linting/optional pydantic environments."""

    def __init__(self, **_kwargs: Any) -> None:
        pass


class _FallbackValidationError(Exception):
    """Fallback when pydantic isn't importable."""


class _FallbackUnsupportedFieldAttributeWarning(Warning):
    """Fallback warning category when pydantic isn't importable."""


def _fallback_field(default: Any = None, **_kwargs: Any) -> Any:
    return default


def _fallback_config_dict(**kwargs: Any) -> dict[str, Any]:
    return dict(kwargs)


def _fallback_model_validator(*_args: Any, **_kwargs: Any):
    def _decorator(fn):
        return fn

    return _decorator


def _build_model_validator_from_root(root_validator):
    """Return a pydantic v1-compatible model_validator shim."""

    def _invoke(fn, values: dict[str, Any]):
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return fn(SimpleNamespace(**values))
        params = list(sig.parameters.values())
        if len(params) == 1:
            return fn(SimpleNamespace(**values))
        return fn(None, values)

    def _adapter(*_args: Any, mode: str = "after", **_kwargs: Any):
        def _decorator(fn):
            def _root_validator(_cls, values):
                if mode == "before":
                    result = _invoke(fn, values)
                    return result if isinstance(result, dict) else values
                result = _invoke(fn, values)
                return result if isinstance(result, dict) else values

            return root_validator(pre=(mode == "before"), skip_on_failure=True)(
                _root_validator
            )

        return _decorator

    return _adapter


def _load_pydantic():
    try:
        return importlib.import_module("pydantic")
    except ImportError:
        return None


_PYDANTIC = _load_pydantic()
BaseModel = _FallbackBaseModel
ConfigDict = _fallback_config_dict
Field = _fallback_field
PositiveInt = int
ValidationError = _FallbackValidationError
model_validator = _fallback_model_validator
if _PYDANTIC is not None:
    BaseModel = getattr(_PYDANTIC, "BaseModel", _FallbackBaseModel)
    ConfigDict = getattr(_PYDANTIC, "ConfigDict", _fallback_config_dict)
    Field = getattr(_PYDANTIC, "Field", _fallback_field)
    PositiveInt = getattr(_PYDANTIC, "PositiveInt", int)
    ValidationError = getattr(_PYDANTIC, "ValidationError", _FallbackValidationError)
    model_validator = getattr(_PYDANTIC, "model_validator", None)
    if model_validator is None:
        root_validator_fn = getattr(_PYDANTIC, "root_validator", None)
        if root_validator_fn is not None:
            model_validator = _build_model_validator_from_root(root_validator_fn)
        else:
            model_validator = _fallback_model_validator

try:
    _pydantic_warnings = importlib.import_module("pydantic.warnings")
    UnsupportedFieldAttributeWarning = getattr(
        _pydantic_warnings,
        "UnsupportedFieldAttributeWarning",
        _FallbackUnsupportedFieldAttributeWarning,
    )
except ImportError:
    UnsupportedFieldAttributeWarning = _FallbackUnsupportedFieldAttributeWarning

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
    :returns: ``None``. Raises on invalid or incompatible configurations.
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
    """Validate generation command payloads before invoking pipelines.

    :param config: Mapping or object carrying generation options (e.g., a Hydra
        config block or :class:`DistilabelGenerationConfig`-like object).
    :param command: Command label included in error messages.
    :param source: Optional hint describing where the config came from.
    :returns: ``None``. Raises on invalid payloads.
    :raises ValueError: If required fields are missing or invalid.
    """

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
    """Validate inference command payloads prior to execution.

    :param config: Mapping or object carrying inference options (Hydra config or
        an object with attributes like ``models`` and ``dataset``).
    :param command: Command label included in error messages.
    :param source: Optional hint describing where the config came from.
    :returns: ``None``. Raises on invalid payloads.
    :raises ValueError: If required fields are missing or invalid.
    """

    values = _payload_mapping(config)
    try:
        _InferenceSchema(**values)
    except ValidationError as exc:
        message = _format_validation_errors(exc.errors())
        hint = _command_hint(command, source)
        raise ValueError(f"{hint}: {message}") from exc
