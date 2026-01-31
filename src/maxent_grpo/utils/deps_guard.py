"""Runtime guards against missing dependencies."""

from __future__ import annotations

import importlib
from typing import Iterable


def _check_modules(names: Iterable[str]) -> list[str]:
    """Return human-friendly errors for missing modules."""

    failures: list[str] = []
    for name in names:
        try:
            importlib.import_module(name)
        except (ImportError, OSError, RuntimeError) as exc:  # pragma: no cover - environment dependent
            failures.append(f"{name} (import failed: {exc})")
    return failures


def ensure_real_dependencies(
    *,
    context: str,
    require_torch: bool = True,
    require_transformers: bool = True,
    require_trl: bool = True,
    require_datasets: bool = True,
    model: object | None = None,
    tokenizer: object | None = None,
) -> None:
    """Raise if required deps are missing.

    :param context: Human-readable context included in error messages.
    :param require_torch: Whether to enforce a real ``torch`` import.
    :param require_transformers: Whether to enforce ``transformers`` availability.
    :param require_trl: Whether to enforce ``trl`` availability.
    :param require_datasets: Whether to enforce ``datasets`` availability.
    :param model: Optional model instance (unused; retained for signature compatibility).
    :param tokenizer: Optional tokenizer instance (unused; retained for signature compatibility).
    :returns: ``None``. Raises when required dependencies are missing.
    :raises RuntimeError: If required modules are missing.
    """
    _ = (model, tokenizer)
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
    if failures:
        details = ", ".join(failures)
        raise RuntimeError(
            f"{context} requires real dependencies, but found: {details}."
        )


__all__ = ["ensure_real_dependencies"]
