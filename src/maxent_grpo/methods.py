"""Canonical method identity helpers for GRPO, Dr.GRPO, MaxEnt, and SEED-GRPO.

This module keeps the two method-selection axes explicit:

- algorithm family: baseline GRPO, SEED-GRPO, entropy MaxEnt, listwise MaxEnt
- loss backend: ``grpo``, ``bnpo``, or ``dr_grpo``

The rest of the codebase can use one resolved object instead of reconstructing
the intended method from several loosely related flags.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Optional

from maxent_grpo.objectives import normalize_objective

GRPO_LOSS_BACKENDS: Final[frozenset[str]] = frozenset({"grpo", "bnpo", "dr_grpo"})
_GRPO_LOSS_BACKEND_ALIASES: Final[dict[str, str]] = {
    "": "bnpo",
    "default": "bnpo",
    "native": "grpo",
    "dr": "dr_grpo",
    "drgrpo": "dr_grpo",
    "dr-grpo": "dr_grpo",
}
_GRPO_LOSS_BACKEND_ERROR = "grpo_loss_type must be one of: grpo, bnpo, dr_grpo"

METHOD_FAMILY_LABELS: Final[dict[str, str]] = {
    "grpo": "GRPO",
    "grpo_entropy_bonus": "GRPO + entropy bonus",
    "seed_grpo": "SEED-GRPO",
    "maxent_entropy": "Entropy MaxEnt",
    "maxent_listwise": "Listwise MaxEnt",
}

LOSS_BACKEND_LABELS: Final[dict[str, str]] = {
    "grpo": "GRPO",
    "bnpo": "BNPO",
    "dr_grpo": "Dr.GRPO",
}


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
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


def normalize_grpo_loss_type(value: Any, *, default: str = "bnpo") -> str:
    """Return the canonical GRPO loss-backend label."""

    normalized_default = str(default or "bnpo").strip().lower()
    normalized_default = _GRPO_LOSS_BACKEND_ALIASES.get(
        normalized_default, normalized_default
    )
    if normalized_default not in GRPO_LOSS_BACKENDS:
        raise ValueError(_GRPO_LOSS_BACKEND_ERROR)

    if value is None:
        candidate = normalized_default
    else:
        candidate = str(value).strip().lower()
        if not candidate:
            candidate = normalized_default
    normalized = _GRPO_LOSS_BACKEND_ALIASES.get(candidate, candidate)
    if normalized not in GRPO_LOSS_BACKENDS:
        raise ValueError(_GRPO_LOSS_BACKEND_ERROR)
    return normalized


def infer_method_family(*, objective: Any, seed_grpo_enabled: Any = False) -> str:
    """Return the canonical algorithm-family key for a config-like payload."""

    normalized_objective = normalize_objective(objective, default="maxent_entropy")
    if normalized_objective == "grpo" and _coerce_bool(seed_grpo_enabled, default=False):
        return "seed_grpo"
    return normalized_objective


def _canonical_method_name(family: str, loss_backend: str) -> str:
    """Return the human-readable method label combining both selection axes."""

    family_label = METHOD_FAMILY_LABELS[family]
    backend_label = LOSS_BACKEND_LABELS[loss_backend]
    if family == "grpo":
        if loss_backend == "grpo":
            return "GRPO"
        if loss_backend == "dr_grpo":
            return "Dr.GRPO"
        return "GRPO (BNPO loss)"
    if family == "grpo_entropy_bonus":
        return f"{family_label} ({backend_label} loss)"
    return f"{family_label} ({backend_label} loss)"


@dataclass(frozen=True)
class MethodSpec:
    """Resolved training-method identity."""

    family: str
    family_label: str
    objective: str
    loss_backend: str
    loss_backend_label: str
    seed_grpo_enabled: bool
    canonical_name: str
    slug: str

    @property
    def uses_dr_grpo_backend(self) -> bool:
        """Return ``True`` when the resolved loss backend is Dr.GRPO."""

        return self.loss_backend == "dr_grpo"


def resolve_method_spec(
    *,
    objective: Any,
    grpo_loss_type: Any = None,
    seed_grpo_enabled: Any = False,
    default_objective: str = "maxent_entropy",
    default_loss_backend: str = "bnpo",
) -> MethodSpec:
    """Resolve a config-like payload into one canonical method identity."""

    normalized_objective = normalize_objective(objective, default=default_objective)
    normalized_loss_backend = normalize_grpo_loss_type(
        grpo_loss_type, default=default_loss_backend
    )
    family = infer_method_family(
        objective=normalized_objective,
        seed_grpo_enabled=seed_grpo_enabled,
    )
    family_label = METHOD_FAMILY_LABELS[family]
    seed_enabled = family == "seed_grpo"
    canonical_name = _canonical_method_name(family, normalized_loss_backend)
    return MethodSpec(
        family=family,
        family_label=family_label,
        objective=normalized_objective,
        loss_backend=normalized_loss_backend,
        loss_backend_label=LOSS_BACKEND_LABELS[normalized_loss_backend],
        seed_grpo_enabled=seed_enabled,
        canonical_name=canonical_name,
        slug=f"{family}__{normalized_loss_backend}",
    )


def resolve_method_spec_from_args(args: Any) -> Optional[MethodSpec]:
    """Return a resolved method spec from a training-args-like object."""

    if args is None:
        return None
    objective = getattr(args, "objective", None)
    if objective in (None, ""):
        return None
    loss_backend = getattr(args, "grpo_loss_type", None)
    if loss_backend in (None, ""):
        loss_backend = getattr(args, "loss_type", None)
    return resolve_method_spec(
        objective=objective,
        grpo_loss_type=loss_backend,
        seed_grpo_enabled=getattr(args, "seed_grpo_enabled", False),
    )


__all__ = [
    "GRPO_LOSS_BACKENDS",
    "LOSS_BACKEND_LABELS",
    "METHOD_FAMILY_LABELS",
    "MethodSpec",
    "infer_method_family",
    "normalize_grpo_loss_type",
    "resolve_method_spec",
    "resolve_method_spec_from_args",
]
