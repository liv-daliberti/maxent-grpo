"""Legacy loop compatibility shim.

The custom training loop was removed. Active training now runs through the
TRL/HF trainer path. This module only re-exports helper hooks for
backward-compatible imports.
"""

from __future__ import annotations

from .trainer_hooks import (
    _apply_weighting_overrides_from_config,
    _cache_meta_stats,
    _log_prompt_objective,
    _maybe_overwrite_controller_state_from_config,
    _maybe_save_seed_heatmap,
)
from .types import TrainingLoopContext


def run_training_loop(ctx: TrainingLoopContext) -> None:
    """Legacy entrypoint retained only to fail fast with a clear message."""

    del ctx
    raise RuntimeError(
        "Custom training loop removed. Use the TRL/HF Trainer loop via "
        "pipelines.training.baseline or pipelines.training.maxent."
    )


__all__ = [
    "_apply_weighting_overrides_from_config",
    "_cache_meta_stats",
    "_log_prompt_objective",
    "_maybe_overwrite_controller_state_from_config",
    "_maybe_save_seed_heatmap",
    "run_training_loop",
]
