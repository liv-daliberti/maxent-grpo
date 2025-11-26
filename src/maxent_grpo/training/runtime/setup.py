"""Setup utilities for loading runtime dependencies and accelerator plugins."""

from __future__ import annotations

from typing import Any, Optional, Tuple

from maxent_grpo.utils.imports import (
    cached_import as _import_module,
    optional_import as _optional_dependency,
    require_dependency as _require_dependency,
)

from . import deps as _deps
from .config import (
    GenerationSamplingConfig,
    MaxEntOptions,
    SeedAugmentationConfig,
    VLLMClientConfig,
)
from .torch_stub import _build_torch_stub

# Re-export accelerator/deepspeed types for callers.
Accelerator = _deps.Accelerator
DeepSpeedPlugin = _deps.DeepSpeedPlugin


def _sync_dep_imports() -> None:
    """Ensure monkeypatches against this module propagate to the deps helpers."""

    if hasattr(_deps, "sync_import_helpers"):
        _deps.sync_import_helpers(
            _import_module, _optional_dependency, _require_dependency
        )


def require_torch(context: str) -> Any:
    """Return the torch module or a stub for test environments."""

    _sync_dep_imports()
    return _deps.require_torch(context)


def require_dataloader(context: str) -> Any:
    """Return torch.utils.data.DataLoader with a descriptive error on failure."""

    _sync_dep_imports()
    return _deps.require_dataloader(context)


def require_accelerator(context: str) -> Any:
    """Return accelerate.Accelerator or raise a helpful RuntimeError."""

    _sync_dep_imports()
    return _deps.require_accelerator(context)


def require_transformer_base_classes(context: str) -> Tuple[Any, Any]:
    """Return (PreTrainedModel, PreTrainedTokenizer) with clear failure messages."""

    _sync_dep_imports()
    return _deps.require_transformer_base_classes(context)


def require_deepspeed(context: str, module: str = "deepspeed") -> Any:
    """Return a DeepSpeed module import or raise a contextual RuntimeError."""

    _sync_dep_imports()
    return _deps.require_deepspeed(context, module)


def get_trl_prepare_deepspeed() -> Optional[Any]:
    """Return TRL's prepare_deepspeed helper when available."""

    _sync_dep_imports()
    return _deps.get_trl_prepare_deepspeed()


def _maybe_create_deepspeed_plugin() -> Optional[Any]:
    """Construct a DeepSpeedPlugin from Accelerate env/config when available."""

    _sync_dep_imports()
    return _deps.maybe_create_deepspeed_plugin()


__all__ = [
    "Accelerator",
    "DeepSpeedPlugin",
    "GenerationSamplingConfig",
    "MaxEntOptions",
    "SeedAugmentationConfig",
    "VLLMClientConfig",
    "_build_torch_stub",
    "_import_module",
    "_optional_dependency",
    "_require_dependency",
    "_maybe_create_deepspeed_plugin",
    "get_trl_prepare_deepspeed",
    "require_accelerator",
    "require_dataloader",
    "require_deepspeed",
    "require_torch",
    "require_transformer_base_classes",
]
