"""Pydantic-powered validation for Hydra training configs.

This module inspects the resolved training arguments before a pipeline is
launched so accidental MaxEnt toggles are caught early. The validator is kept
lightweight and only depends on :mod:`pydantic`, which is already part of the
runtime toolchain for several other components. Future guardrails can extend
this module by adding additional schema checks (including GRPO + entropy-bonus
overrides under ``train-maxent``).
"""

from __future__ import annotations

import warnings
from dataclasses import MISSING, Field as DataclassField, fields
from typing import Any, Mapping, MutableMapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic.warnings import UnsupportedFieldAttributeWarning

from maxent_grpo.config import GRPOConfig

__all__ = [
    "validate_training_config",
]

warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)


def _field_default(field: DataclassField[object]) -> object | None:
    if field.default is not MISSING:
        return field.default
    if field.default_factory is not MISSING:
        return field.default_factory()
    return None


_MAXENT_DEFAULTS = {
    field.name: _field_default(field)
    for field in fields(GRPOConfig)
    if field.name.startswith("maxent_")
}

_DEFAULT_OBJECTIVE_BY_COMMAND = {
    "train-baseline": True,
    "train-maxent": False,
}


class _TrainingSchema(BaseModel):
    """Minimal schema capturing the knobs that need cross-field validation."""

    model_config = ConfigDict(extra="forbid")

    train_grpo_objective: bool | None = None
    default_objective: bool | None = Field(default=None)
    maxent_overrides: dict[str, Any] = Field(default_factory=dict)
    allow_grpo_with_maxent_overrides: bool = False

    @model_validator(mode="after")
    def _check_maxent_conflicts(self) -> "_TrainingSchema":
        effective = (
            self.train_grpo_objective
            if self.train_grpo_objective is not None
            else self.default_objective
        )
        if (
            effective is not False
            and self.maxent_overrides
            and not self.allow_grpo_with_maxent_overrides
        ):
            knobs = ", ".join(sorted(self.maxent_overrides))
            raise ValueError(
                "MaxEnt overrides (%s) require train_grpo_objective=false" % knobs
            )
        return self


def _training_values(payload: Any) -> MutableMapping[str, Any]:
    """Return a mapping containing the knobs relevant to validation."""

    if isinstance(payload, Mapping):
        return {key: payload[key] for key in payload}
    values: MutableMapping[str, Any] = {}
    attr_names = set(_MAXENT_DEFAULTS)
    attr_names |= {
        "train_grpo_objective",
        "policy_entropy_bonus_coef",
    }
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


def _source_hint(command: str, *, recipe: str | None, training_args: Any) -> str:
    """Return a short string pointing at the config origin for error messages."""

    hints: list[str] = [command]
    recipe_path = recipe or getattr(training_args, "recipe_path", None)
    if recipe_path:
        hints.append(str(recipe_path))
    return " | ".join(hints)


def _format_validation_errors(errors: Sequence[Mapping[str, Any]]) -> str:
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
    :class:`ValueError` is raised so the job fails fast, except when running
    ``train-maxent`` with ``policy_entropy_bonus_coef>0`` to enable GRPO + entropy bonus.

    :param training_args: Training dataclass or mapping derived from Hydra.
    :param command: CLI command being executed (e.g., ``train-baseline``).
    :param source: Optional user-facing hint (recipe path, override description).
    :returns: ``None``. Raises on invalid or incompatible configurations.
    :raises ValueError: If incompatible knob combinations are detected.
    """

    values = _training_values(training_args)
    bonus_coef = values.get("policy_entropy_bonus_coef", 0.0)
    allow_grpo_with_maxent_overrides = False
    try:
        allow_grpo_with_maxent_overrides = (
            command == "train-maxent" and float(bonus_coef) > 0.0
        )
    except (TypeError, ValueError):
        allow_grpo_with_maxent_overrides = False
    schema_payload = {
        "train_grpo_objective": values.get("train_grpo_objective"),
        "default_objective": _DEFAULT_OBJECTIVE_BY_COMMAND.get(command),
        "maxent_overrides": _maxent_overrides(values),
        "allow_grpo_with_maxent_overrides": allow_grpo_with_maxent_overrides,
    }
    try:
        _TrainingSchema(**schema_payload)
    except ValidationError as exc:
        message = _format_validation_errors(exc.errors())
        hint = _source_hint(command, recipe=source, training_args=training_args)
        raise ValueError(f"{hint}: {message}") from exc
