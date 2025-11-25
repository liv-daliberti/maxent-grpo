"""
Additional coverage-focused tests for :mod:`maxent_grpo.training.run_helpers`.
"""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import maxent_grpo.training.run_helpers as run_helpers
import maxent_grpo.training.runtime.torch_utils as torch_utils


def test_tensor_getitem_and_setitem_simple():
    stub = run_helpers._build_torch_stub()
    tensor = stub.tensor([[1, 2], [3, 4]])
    second = tensor[1]
    assert second.tolist() == [3, 4]
    tensor[0] = [9, 8]
    assert tensor.tolist()[0] == [9, 8]


def test_tensor_numeric_helpers_and_binary_scalar():
    stub = run_helpers._build_torch_stub()
    tensor = stub.tensor([[1, 2], [3, 4]])
    assert tensor.float() is tensor
    assert tensor.numel() == 2
    summed = tensor.sum(dim=1)
    assert summed.tolist() == [3, 7]
    empty = stub.tensor([])
    assert empty.item() == []
    vector = stub.tensor([2, 4])
    assert (vector / 2).tolist() == [1.0, 2.0]
    assert (vector - 1).tolist() == [1, 3]
    assert vector.cpu() is vector
    cat = stub.cat([stub.tensor([5]), [6]])
    assert cat.tolist() == [5, 6]


def test_require_torch_recovers_when_required_attrs_missing(monkeypatch):
    torch_utils._import_module.cache_clear()
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    def _fake_import(name: str):
        if name == "torch":
            return SimpleNamespace()
        if name == "ops.sitecustomize":
            raise ImportError("no bootstrap")
        return importlib.import_module(name)

    _fake_import.cache_clear = lambda: None
    monkeypatch.setattr(torch_utils, "_import_module", _fake_import)
    torch_mod = run_helpers.require_torch("missing-required")
    for attr in ("tensor", "full", "ones_like", "zeros"):
        assert hasattr(torch_mod, attr)
