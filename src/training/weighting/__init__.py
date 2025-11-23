"""Weighting helpers and dataclasses for the MaxEnt-GRPO trainer."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .types import (  # noqa: F401  re-export types
    KlControllerSettings,
    QDistributionSettings,
    TauSchedule,
    WeightLoggingView,
    WeightNormalizationSettings,
    WeightStats,
    WeightingSettings,
)

_LOGIC_EXPORTS = {
    "CONTROLLER_STATE_FILENAME",
    "collect_weight_entropy",
    "compute_weight_stats",
    "controller_state_dict",
    "load_controller_state",
    "maybe_update_beta",
    "maybe_update_tau",
    "save_controller_state",
    "split_reference_logprobs",
    "split_reference_token_counts",
    "weight_vector_from_q",
}

_TYPE_EXPORTS = {
    "KlControllerSettings",
    "QDistributionSettings",
    "TauSchedule",
    "WeightLoggingView",
    "WeightNormalizationSettings",
    "WeightStats",
    "WeightingSettings",
}

__all__ = sorted(_LOGIC_EXPORTS | _TYPE_EXPORTS)


def __getattr__(name: str) -> Any:
    """Lazily import weighting logic helpers on demand."""
    if name in _LOGIC_EXPORTS:
        module = import_module(".logic", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the sorted public attributes exposed by this package."""
    return __all__
