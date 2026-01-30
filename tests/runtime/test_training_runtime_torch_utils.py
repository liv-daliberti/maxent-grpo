"""Branch coverage for training.runtime.torch_utils helpers."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace


def test_require_torch_bootstrap_retry_falls_back(monkeypatch):
    from maxent_grpo.training.runtime import torch_utils

    torch_utils._import_module.cache_clear()
    for mod in ("torch", "sitecustomize"):
        monkeypatch.delitem(sys.modules, mod, raising=False)

    torch_attempts = [
        SimpleNamespace(tensor=lambda *_a, **_k: None),  # missing required attrs
        SimpleNamespace(tensor=lambda *_a, **_k: None),
    ]

    def fake_import(name: str):
        if name == "torch":
            return torch_attempts.pop(0)
        return importlib.import_module(name)

    fake_import.cache_clear = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setattr(torch_utils, "_import_module", fake_import)

    bootstrap = ModuleType("sitecustomize")
    install_calls = []

    def _install_torch_stub():
        install_calls.append(True)

    bootstrap._install_torch_stub = _install_torch_stub
    ops_pkg = ModuleType("ops")
    ops_pkg.sitecustomize = bootstrap
    monkeypatch.setitem(sys.modules, "ops", ops_pkg)
    monkeypatch.setitem(sys.modules, "sitecustomize", bootstrap)

    torch_mod = torch_utils.require_torch("test-bootstrap-missing")
    assert install_calls == [True]
    assert hasattr(torch_mod, "zeros")
    assert torch_mod.zeros((2,)).tolist() == [0, 0]


def test_require_torch_returns_existing(monkeypatch):
    from maxent_grpo.training.runtime import torch_utils

    sentinel = object()
    monkeypatch.setitem(sys.modules, "torch", sentinel)
    assert torch_utils.require_torch("existing") is sentinel
    monkeypatch.delitem(sys.modules, "torch", raising=False)


def test_require_torch_missing_attrs_and_no_bootstrap(monkeypatch):
    from maxent_grpo.training.runtime import torch_utils

    torch_utils._import_module.cache_clear()
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    # First call returns a module missing required attrs; sitecustomize import will fail.
    monkeypatch.setattr(
        torch_utils,
        "_import_module",
        lambda name: (
            SimpleNamespace(tensor=lambda *_a, **_k: None)
            if name == "torch"
            else importlib.import_module(name)
        ),
    )
    torch_mod = torch_utils.require_torch("no-bootstrap")
    assert hasattr(torch_mod, "tensor")
    assert hasattr(torch_mod, "zeros")


def test_require_dataloader_builds_stub(monkeypatch):
    from maxent_grpo.training.runtime import torch_utils

    torch_utils._import_module.cache_clear()
    for mod in ("torch.utils.data", "torch.utils", "torch"):
        monkeypatch.delitem(sys.modules, mod, raising=False)

    def _failing_import(name: str):
        raise ModuleNotFoundError("missing")

    _failing_import.cache_clear = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setattr(torch_utils, "_import_module", _failing_import)
    loader_cls = torch_utils.require_dataloader("ctx")
    assert loader_cls.__name__ == "DataLoader"
    assert "torch.utils.data" in sys.modules
