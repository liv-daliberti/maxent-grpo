"""
Ensure hub module surfaces a clear error when transformers is absent.
"""

from __future__ import annotations

import builtins
import importlib
import sys

import pytest


def test_hub_import_re_raises_missing_transformers(monkeypatch):
    """Import should re-raise ModuleNotFoundError if transformers is missing."""

    # Drop cached modules so the import block executes.
    monkeypatch.delitem(sys.modules, "maxent_grpo.core.hub", raising=False)
    monkeypatch.delitem(sys.modules, "transformers", raising=False)
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "transformers":
            raise ModuleNotFoundError("transformers missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("maxent_grpo.core.hub")
