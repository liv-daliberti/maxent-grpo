"""Runtime guards against stubbed or missing dependencies."""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any, Iterable

_ALLOW_STUBS_ENV = "ALLOW_STUBS"
_ALLOW_STUB_VALUES = {"1", "true", "yes", "y", "on"}

# If pytest is already imported, default to allowing stubs.
if "pytest" in sys.modules and not os.environ.get(_ALLOW_STUBS_ENV):
    os.environ[_ALLOW_STUBS_ENV] = "1"


def _allow_stubs() -> bool:
    """Return True when the environment explicitly allows stubbed deps."""

    raw = os.environ.get(_ALLOW_STUBS_ENV, "")
    if raw.strip().lower() in _ALLOW_STUB_VALUES:
        return True
    # Pytest sets PYTEST_CURRENT_TEST for each test item; allow stubs in that case.
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return True
    if os.environ.get("PYTEST_XDIST_WORKER") or os.environ.get("PYTEST_WORKER"):
        return True
    if os.environ.get("PYTEST_RUNNING"):
        return True
    if "pytest" in sys.modules:
        return True
    if os.environ.get("MAXENT_CLI_SMOKE_MARKER") or os.environ.get("MAXENT_CLI_SMOKE_KIND"):
        return True
    return False


def _module_is_stub(module: Any) -> bool:
    """Return True when a module looks like a lightweight stub."""

    if module is None:
        return False
    if getattr(module, "__maxent_stub__", False):
        return True
    if getattr(module, "__spec__", None) is None and getattr(module, "__file__", None) is None:
        return True
    return False


def _object_is_stub(obj: Any) -> bool:
    """Return True when an instance or class advertises stub status."""

    if obj is None:
        return False
    if getattr(obj, "__maxent_stub__", False):
        return True
    cls = getattr(obj, "__class__", None)
    if cls is not None and getattr(cls, "__maxent_stub__", False):
        return True
    return False


def _check_modules(names: Iterable[str]) -> list[str]:
    """Return human-friendly errors for missing or stubbed modules."""

    failures: list[str] = []
    for name in names:
        try:
            module = importlib.import_module(name)
        except (ImportError, OSError, RuntimeError) as exc:  # pragma: no cover - environment dependent
            failures.append(f"{name} (import failed: {exc})")
            continue
        if _module_is_stub(module):
            failures.append(f"{name} (stub)")
    return failures


def ensure_real_dependencies(
    *,
    context: str,
    require_torch: bool = True,
    require_transformers: bool = True,
    require_trl: bool = True,
    require_datasets: bool = True,
    model: Any = None,
    tokenizer: Any = None,
) -> None:
    """Raise if required deps are missing or replaced with stubs.

    Set ``ALLOW_STUBS=1`` to bypass the guard (intended for tests/CI stubs).

    :param context: Human-readable context included in error messages.
    :param require_torch: Whether to enforce a real ``torch`` import.
    :param require_transformers: Whether to enforce ``transformers`` availability.
    :param require_trl: Whether to enforce ``trl`` availability.
    :param require_datasets: Whether to enforce ``datasets`` availability.
    :param model: Optional model instance to check for stub markers.
    :param tokenizer: Optional tokenizer instance to check for stub markers.
    :returns: ``None``. Raises when required dependencies are missing.
    :raises RuntimeError: If required modules or objects appear stubbed or missing.
    """

    if _allow_stubs():
        return
    modules: list[str] = []
    if require_torch:
        modules.append("torch")
    if require_transformers:
        modules.append("transformers")
    if require_trl:
        modules.append("trl")
    if require_datasets:
        modules.append("datasets")
    failures = _check_modules(modules)
    if _object_is_stub(model):
        failures.append("model (stub)")
    if _object_is_stub(tokenizer):
        failures.append(
            "tokenizer (stub; AutoTokenizer fallback likely due to offline model fetch or missing transformers)"
        )
    if failures:
        details = ", ".join(failures)
        raise RuntimeError(
            f"{context} requires real dependencies, but found: {details}. "
            f"Set {_ALLOW_STUBS_ENV}=1 to bypass."
        )


__all__ = ["ensure_real_dependencies"]
