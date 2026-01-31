"""Shared helpers for optional and required dependency imports."""

from __future__ import annotations

import importlib
import sys
from functools import lru_cache
from types import ModuleType
from typing import Optional


@lru_cache(maxsize=None)
def cached_import(module_name: str) -> ModuleType:
    """Import a module with caching to avoid repeated lookups.

    :param module_name: Fully qualified module path to import.
    :returns: Imported module object, cached for subsequent calls.
    :raises ImportError: Propagated if the module cannot be imported.
    """
    return importlib.import_module(module_name)


def optional_import(module_name: str) -> Optional[ModuleType]:
    """Import a module if available without bubbling up ImportError.

    This deliberately skips the cached importer so tests and call sites that
    monkeypatch ``sys.modules`` see their changes reflected immediately.

    :param module_name: Fully qualified module path to import.
    :returns: Imported module, or ``None`` when the module is missing.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def require_dependency(module_name: str, context_hint: str) -> ModuleType:
    """Import a dependency or raise a helpful error when it is missing.

    :param module_name: Fully qualified module path to import.
    :param context_hint: Human-friendly error message describing the caller context.
    :returns: Imported module, if available.
    :raises ImportError: Wrapped with ``context_hint`` when the dependency is absent.
    """
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    try:
        return cached_import(module_name)
    except ModuleNotFoundError as exc:
        raise ImportError(context_hint) from exc


__all__ = ["cached_import", "optional_import", "require_dependency"]
