"""Unit tests for training package initialization shims."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace


def test_training_import_seeds_wandb_stub(monkeypatch):
    monkeypatch.delitem(sys.modules, "training", raising=False)
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    import training

    assert "wandb" in sys.modules
    stub = sys.modules["wandb"]
    assert getattr(getattr(stub, "errors", SimpleNamespace()), "Error", None) is RuntimeError

    # Ensure __dir__ returns a sorted list of exported names
    names = training.__dir__()
    assert isinstance(names, list)
    assert names == sorted(names)
