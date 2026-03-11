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
from maxent_grpo.objectives import normalize_maxent_objective_variant, resolve_objective_routing

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
    "train-baseline": "grpo",
    "train-maxent": "maxent_entropy",
}

_GRPO_SAFE_MAXENT_KNOBS = {
    "maxent_allow_empty_weight_fallback",
    "maxent_allow_stale_reference_logprobs",
    "maxent_length_normalize_ref",
    "maxent_logprob_chunk_size",
    "maxent_policy_entropy",
    "maxent_policy_entropy_mode",
    "maxent_prompt_cache_size",
    "maxent_reference_logprobs_source",
    "maxent_score_tail_tokens",
    "maxent_trl_reference_scoring",
}

_REMOVED_TRAINING_KEYS = {
    "maxent_reward_signal_gate",
    "maxent_reward_signal_min_max",
    "maxent_reward_signal_std_threshold",
    "maxent_bonus_positive_only",
    "maxent_bonus_min_reward",
    "maxent_cusp_gate",
    "maxent_cusp_reward_threshold",
    "controller_meta_objective",
    "controller_meta_analytic_steps",
    "controller_meta_optimizer",
    "controller_meta_truncation_steps",
    "controller_meta_use_hessian",
}


class _TrainingSchema(BaseModel):
    """Minimal schema capturing the knobs that need cross-field validation."""

    model_config = ConfigDict(extra="forbid")

    objective: str | None = None
    train_grpo_objective: bool | None = None
    maxent_objective_variant: str | None = None
    policy_entropy_bonus_coef: float | None = None
    default_objective: str | None = Field(default=None)
    maxent_overrides: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_maxent_conflicts(self) -> "_TrainingSchema":
        effective = resolve_objective_routing(
            objective=self.objective,
            train_grpo_objective=self.train_grpo_objective,
            maxent_objective_variant=self.maxent_objective_variant,
            policy_entropy_bonus_coef=self.policy_entropy_bonus_coef,
            default_objective=self.default_objective or "maxent_entropy",
        )
        if effective.objective in {"grpo", "grpo_entropy_bonus"} and self.maxent_overrides:
            knobs = ", ".join(sorted(self.maxent_overrides))
            raise ValueError(
                "MaxEnt overrides (%s) require objective=maxent_entropy "
                "or objective=maxent_listwise" % knobs
            )
        return self


def _training_values(payload: Any) -> MutableMapping[str, Any]:
    """Return a mapping containing the knobs relevant to validation."""

    if isinstance(payload, Mapping):
        return {key: payload[key] for key in payload}
    values: MutableMapping[str, Any] = {}
    attr_names = set(_MAXENT_DEFAULTS)
    attr_names |= {
        "objective",
        "train_grpo_objective",
        "maxent_objective_variant",
        "policy_entropy_bonus_coef",
    }
    for name in attr_names:
        if hasattr(payload, name):
            values[name] = getattr(payload, name)
    return values


def _numeric_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _integer_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_entropy_mode(value: Any) -> str:
    """Return the canonical entropy-mode label used by config validation."""

    candidate = str(value or "exact").strip().lower()
    if candidate in {"", "none", "exact", "full", "distribution"}:
        return "exact"
    if candidate in {
        "sample",
        "estimate",
        "estimated",
        "approx",
        "approximate",
        "token",
        "token_logp",
        "nll",
        "logp",
    }:
        return "sample"
    raise ValueError("maxent_policy_entropy_mode must be one of: exact, sample")


def _is_safe_grpo_maxent_override(name: str, value: Any) -> bool:
    if name in _GRPO_SAFE_MAXENT_KNOBS:
        return True
    if name == "maxent_alpha":
        numeric = _numeric_or_none(value)
        return numeric is not None and numeric <= 0.0
    if name == "policy_entropy_bonus_coef":
        numeric = _numeric_or_none(value)
        return numeric is not None and numeric > 0.0
    if name == "maxent_objective_variant":
        return normalize_maxent_objective_variant(value, default="entropy") == "entropy"
    return False


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
        if _is_safe_grpo_maxent_override(name, value):
            continue
        overrides[name] = value
    return overrides


def _validate_listwise_microbatch_shape(values: Mapping[str, Any]) -> None:
    routing = resolve_objective_routing(
        objective=values.get("objective"),
        train_grpo_objective=values.get("train_grpo_objective"),
        maxent_objective_variant=values.get("maxent_objective_variant"),
        maxent_alpha=values.get("maxent_alpha"),
        policy_entropy_bonus_coef=values.get("policy_entropy_bonus_coef"),
    )
    if routing.train_grpo_objective:
        return
    if not routing.uses_listwise_loss:
        return
    tau = _numeric_or_none(values.get("maxent_tau"))
    if tau is None or tau <= 0.0:
        raise ValueError("listwise MaxEnt requires maxent_tau > 0")
    if routing.maxent_alpha > 0.0:
        raise ValueError("listwise MaxEnt does not use maxent_alpha; set it to 0")
    num_generations = _integer_or_none(values.get("num_generations"))
    if num_generations is None or num_generations <= 0:
        return
    for batch_name in ("per_device_train_batch_size", "per_device_eval_batch_size"):
        batch_size = _integer_or_none(values.get(batch_name))
        if batch_size is None or batch_size <= 0:
            continue
        if batch_size % num_generations != 0:
            raise ValueError(
                f"listwise MaxEnt requires {batch_name}={batch_size} to be divisible "
                f"by num_generations={num_generations} so each trainer microbatch "
                "contains whole prompt groups"
            )


def _validate_entropy_objective_settings(values: Mapping[str, Any]) -> None:
    """Reject entropy-loss settings that do not match the implemented math."""

    routing = resolve_objective_routing(
        objective=values.get("objective"),
        train_grpo_objective=values.get("train_grpo_objective"),
        maxent_objective_variant=values.get("maxent_objective_variant"),
        maxent_alpha=values.get("maxent_alpha"),
        policy_entropy_bonus_coef=values.get("policy_entropy_bonus_coef"),
    )
    if routing.uses_entropy_regularized_loss:
        entropy_mode = _normalize_entropy_mode(
            values.get("maxent_policy_entropy_mode", "exact")
        )
        if entropy_mode != "exact":
            raise ValueError(
                "Entropy-regularized MaxEnt requires maxent_policy_entropy_mode='exact'; "
                "sample mode is only valid for logging or GRPO reward bonuses."
            )


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


def _validate_removed_training_keys(values: Mapping[str, Any]) -> None:
    removed = sorted(
        name for name in _REMOVED_TRAINING_KEYS if values.get(name) is not None
    )
    if removed:
        raise ValueError("Removed training keys are no longer supported: " + ", ".join(removed))


def validate_training_config(
    training_args: GRPOConfig | Mapping[str, Any],
    *,
    command: str,
    source: str | None = None,
) -> None:
    """Validate Hydrated training knobs before dispatching to a pipeline.

    The validator ensures that the canonical ``objective`` matches the presence of
    MaxEnt-specific options. When MaxEnt knobs are supplied while the effective
    objective stays on the native GRPO path, a :class:`ValueError` is raised so
    the job fails fast.

    :param training_args: Training dataclass or mapping derived from Hydra.
    :param command: CLI command being executed (e.g., ``train-baseline``).
    :param source: Optional user-facing hint (recipe path, override description).
    :returns: ``None``. Raises on invalid or incompatible configurations.
    :raises ValueError: If incompatible knob combinations are detected.
    """

    values = _training_values(training_args)
    effective_values = dict(values)
    if (
        effective_values.get("objective") is None
        and effective_values.get("train_grpo_objective") is None
        and effective_values.get("maxent_objective_variant") is None
        and effective_values.get("policy_entropy_bonus_coef") is None
    ):
        effective_default = _DEFAULT_OBJECTIVE_BY_COMMAND.get(command)
        if effective_default is not None:
            effective_values["objective"] = effective_default
    try:
        schema_payload = {
            "objective": values.get("objective"),
            "train_grpo_objective": values.get("train_grpo_objective"),
            "maxent_objective_variant": values.get("maxent_objective_variant"),
            "policy_entropy_bonus_coef": values.get("policy_entropy_bonus_coef"),
            "default_objective": _DEFAULT_OBJECTIVE_BY_COMMAND.get(command),
            "maxent_overrides": _maxent_overrides(values),
        }
        _TrainingSchema(**schema_payload)
        _validate_removed_training_keys(effective_values)
        _validate_listwise_microbatch_shape(effective_values)
        _validate_entropy_objective_settings(effective_values)
    except ValidationError as exc:
        message = _format_validation_errors(exc.errors())
        hint = _source_hint(command, recipe=source, training_args=training_args)
        raise ValueError(f"{hint}: {message}") from exc
    except ValueError as exc:
        message = str(exc)
        hint = _source_hint(command, recipe=source, training_args=training_args)
        raise ValueError(f"{hint}: {message}") from exc
