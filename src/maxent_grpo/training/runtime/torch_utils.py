"""Compatibility layer exposing torch-related helpers for tests/importers."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any

from maxent_grpo.utils.imports import cached_import as _import_module

from .torch_stub import _build_torch_stub


def require_torch(context: str) -> Any:
    """Return the torch module or a stub for test environments."""

    del context  # context is provided for parity with public helpers.
    existing = sys.modules.get("torch")
    if existing is not None:
        return existing
    try:
        torch_mod = _import_module("torch")
    except (ModuleNotFoundError, RuntimeError):  # pragma: no cover - import guard
        torch_mod = None
        try:
            _bootstrap = importlib.import_module("ops.sitecustomize")
            installer = getattr(_bootstrap, "_install_torch_stub", None)
            if callable(installer):
                installer()
                _import_module.cache_clear()
                torch_mod = _import_module("torch")
        except (ImportError, AttributeError, RuntimeError):
            torch_mod = None
        if torch_mod is None:
            torch_mod = _build_torch_stub()
    required_attrs = ("tensor", "full", "ones_like", "zeros")

    def _missing_required(mod: Any) -> bool:
        return mod is None or any(not hasattr(mod, attr) for attr in required_attrs)

    if _missing_required(torch_mod):
        try:
            _bootstrap = importlib.import_module("ops.sitecustomize")
            installer = getattr(_bootstrap, "_install_torch_stub", None)
            if callable(installer):
                installer()
                _import_module.cache_clear()
                torch_mod = _import_module("torch")
        except (ImportError, AttributeError, RuntimeError):
            torch_mod = None

    if _missing_required(torch_mod):
        torch_mod = _build_torch_stub()
    return torch_mod


def require_dataloader(context: str) -> Any:
    """Return torch.utils.data.DataLoader with a descriptive error on failure."""

    hint = f"Torch's DataLoader is required for MaxEnt-GRPO {context}. Install torch first."
    try:
        torch_data = _import_module("torch.utils.data")
    except (ModuleNotFoundError, RuntimeError):  # pragma: no cover - import guard
        torch_data = None
        try:
            _bootstrap = importlib.import_module("ops.sitecustomize")
            installer = getattr(_bootstrap, "_install_torch_stub", None)
            if callable(installer):
                installer()
                _import_module.cache_clear()
                torch_data = _import_module("torch.utils.data")
        except (ImportError, AttributeError, RuntimeError):
            torch_data = None

        if torch_data is None:
            torch_mod = sys.modules.get("torch")
            if torch_mod is None:
                torch_mod = _build_torch_stub()
                sys.modules["torch"] = torch_mod
            utils_mod = getattr(torch_mod, "utils", None)
            if utils_mod is None:
                utils_mod = ModuleType("torch.utils")
                sys.modules["torch.utils"] = utils_mod
                torch_mod.utils = utils_mod
            data_mod = ModuleType("torch.utils.data")
            data_mod.DataLoader = type("DataLoader", (), {})
            data_mod.Sampler = type("Sampler", (), {})
            sys.modules["torch.utils.data"] = data_mod
            utils_mod.data = data_mod
            torch_data = data_mod
    try:
        return getattr(torch_data, "DataLoader")
    except AttributeError as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            f"torch.utils.data.DataLoader is missing; update your torch installation. {hint}"
        ) from exc


__all__ = ["_build_torch_stub", "_import_module", "require_dataloader", "require_torch"]
