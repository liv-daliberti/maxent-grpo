"""Shared helpers for selecting the active GRPO/MaxEnt objective."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Final

OBJECTIVES: Final[frozenset[str]] = frozenset(
    {"grpo", "grpo_entropy_bonus", "maxent_entropy", "maxent_listwise"}
)
_OBJECTIVE_ALIASES: Final[dict[str, str]] = {
    "": "maxent_entropy",
    "baseline": "grpo",
    "grpo_native": "grpo",
    "entropy_bonus": "grpo_entropy_bonus",
    "maxent": "maxent_entropy",
    "maxent_tokenwise": "maxent_entropy",
    "tokenwise": "maxent_entropy",
    "entropy": "maxent_entropy",
    "listwise": "maxent_listwise",
}
_OBJECTIVE_ERROR = (
    "objective must be one of: grpo, grpo_entropy_bonus, maxent_entropy, "
    "maxent_listwise"
)

MAXENT_OBJECTIVE_VARIANTS: Final[frozenset[str]] = frozenset(
    {"entropy", "listwise"}
)
_MAXENT_OBJECTIVE_VARIANT_ALIASES: Final[dict[str, str]] = {
    "": "entropy",
    "default": "entropy",
    "alpha": "entropy",
    "entropy_bonus": "entropy",
    "entropy_regularized": "entropy",
    "weighting": "listwise",
    "weights": "listwise",
    "list": "listwise",
    "tau_q": "listwise",
}
_MAXENT_OBJECTIVE_VARIANT_ERROR = (
    "maxent_objective_variant must be one of: entropy, listwise"
)


def _coerce_bool(value: Any, *, default: bool) -> bool:
    """Return a predictable bool for config-style inputs."""

    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "f", "no", "n", "off", ""}:
            return False
        return default
    try:
        return bool(value)
    except Exception:
        return default


def _coerce_non_negative_float(value: Any, *, default: float = 0.0) -> float:
    """Return a finite non-negative float for config-style inputs."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric):
        return default
    return max(numeric, 0.0)


def normalize_maxent_objective_variant(
    value: Any,
    *,
    default: str = "entropy",
) -> str:
    """Return the canonical MaxEnt objective variant."""

    normalized_default = str(default or "entropy").strip().lower()
    normalized_default = _MAXENT_OBJECTIVE_VARIANT_ALIASES.get(
        normalized_default, normalized_default
    )
    if normalized_default not in MAXENT_OBJECTIVE_VARIANTS:
        raise ValueError(_MAXENT_OBJECTIVE_VARIANT_ERROR)

    if value is None:
        candidate = normalized_default
    else:
        candidate = str(value).strip().lower()
        if not candidate:
            candidate = normalized_default
    normalized = _MAXENT_OBJECTIVE_VARIANT_ALIASES.get(candidate, candidate)
    if normalized not in MAXENT_OBJECTIVE_VARIANTS:
        raise ValueError(_MAXENT_OBJECTIVE_VARIANT_ERROR)
    return normalized


def is_listwise_maxent_variant(value: Any, *, default: str = "entropy") -> bool:
    """Return ``True`` when ``value`` resolves to the listwise objective."""

    return normalize_maxent_objective_variant(value, default=default) == "listwise"


def normalize_objective(
    value: Any,
    *,
    default: str = "maxent_entropy",
) -> str:
    """Return the canonical top-level objective label."""

    normalized_default = str(default or "maxent_entropy").strip().lower()
    normalized_default = _OBJECTIVE_ALIASES.get(
        normalized_default, normalized_default
    )
    if normalized_default not in OBJECTIVES:
        raise ValueError(_OBJECTIVE_ERROR)

    if value is None:
        candidate = normalized_default
    else:
        candidate = str(value).strip().lower()
        if not candidate:
            candidate = normalized_default
    normalized = _OBJECTIVE_ALIASES.get(candidate, candidate)
    if normalized not in OBJECTIVES:
        raise ValueError(_OBJECTIVE_ERROR)
    return normalized


def _infer_objective_from_legacy_inputs(
    *,
    train_grpo_objective: Any,
    maxent_objective_variant: Any,
    policy_entropy_bonus_coef: Any,
    default_objective: str,
    default_variant: str,
) -> str:
    """Map legacy objective selectors onto the canonical objective enum."""

    normalized_default = normalize_objective(default_objective)
    default_train_grpo = normalized_default in {"grpo", "grpo_entropy_bonus"}
    train_grpo = _coerce_bool(train_grpo_objective, default=default_train_grpo)
    variant = normalize_maxent_objective_variant(
        maxent_objective_variant,
        default=default_variant,
    )
    entropy_bonus = _coerce_non_negative_float(
        policy_entropy_bonus_coef,
        default=0.0,
    )
    if train_grpo:
        return "grpo_entropy_bonus" if entropy_bonus > 0.0 else "grpo"
    if variant == "listwise":
        return "maxent_listwise"
    return "maxent_entropy"


@dataclass(frozen=True)
class ObjectiveRouting:
    """Canonicalized objective selection used by config validation and training."""

    objective: str
    maxent_alpha: float
    policy_entropy_bonus_coef: float

    @property
    def train_grpo_objective(self) -> bool:
        """Return ``True`` when the native GRPO loss stays active."""

        return self.objective in {"grpo", "grpo_entropy_bonus"}

    @property
    def maxent_objective_variant(self) -> str:
        """Return the legacy MaxEnt-variant label derived from ``objective``."""

        return "listwise" if self.objective == "maxent_listwise" else "entropy"

    @property
    def maxent_requested(self) -> bool:
        """Return ``True`` when the run requests a MaxEnt objective."""

        return self.objective in {"maxent_entropy", "maxent_listwise"}

    @property
    def uses_entropy_bonus_rewards(self) -> bool:
        """Return ``True`` when GRPO runs with reward-side entropy bonus."""

        return (
            self.objective == "grpo_entropy_bonus"
            and self.policy_entropy_bonus_coef > 0.0
        )

    @property
    def uses_listwise_loss(self) -> bool:
        """Return ``True`` when the active loss is listwise MaxEnt."""

        return self.objective == "maxent_listwise"

    @property
    def uses_entropy_regularized_loss(self) -> bool:
        """Return ``True`` when the active loss is entropy-regularized MaxEnt."""

        return self.objective == "maxent_entropy"

    @property
    def uses_native_grpo_loss(self) -> bool:
        """Return ``True`` when the active loss reduces to native GRPO."""

        return self.objective in {"grpo", "grpo_entropy_bonus"}

    @property
    def route_mode(self) -> str:
        """Return the canonical trainer routing label."""

        return self.objective


def resolve_objective_routing(
    *,
    objective: Any = None,
    train_grpo_objective: Any,
    maxent_objective_variant: Any,
    maxent_alpha: Any = 0.0,
    policy_entropy_bonus_coef: Any = 0.0,
    default_objective: str = "maxent_entropy",
    default_variant: str = "entropy",
) -> ObjectiveRouting:
    """Return the canonical objective routing for a config-like payload."""

    return ObjectiveRouting(
        objective=(
            normalize_objective(objective, default=default_objective)
            if objective is not None
            else _infer_objective_from_legacy_inputs(
                train_grpo_objective=train_grpo_objective,
                maxent_objective_variant=maxent_objective_variant,
                policy_entropy_bonus_coef=policy_entropy_bonus_coef,
                default_objective=default_objective,
                default_variant=default_variant,
            )
        ),
        maxent_alpha=_coerce_non_negative_float(maxent_alpha, default=0.0),
        policy_entropy_bonus_coef=_coerce_non_negative_float(
            policy_entropy_bonus_coef,
            default=0.0,
        ),
    )
