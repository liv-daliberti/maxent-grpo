"""Tests for utils.fallbacks and utils.imports helpers."""

from __future__ import annotations

import importlib
from types import ModuleType, SimpleNamespace

import pytest

from maxent_grpo.utils import fallbacks
from maxent_grpo.utils import imports


def test_optional_import_and_cache(monkeypatch):
    sentinel = ModuleType("sentinel_mod")
    seen = {}

    def _import(name):
        seen[name] = seen.get(name, 0) + 1
        if name == "sentinel":
            return sentinel
        raise ImportError

    monkeypatch.setattr(importlib, "import_module", _import)
    imports.cached_import.cache_clear()
    assert imports.optional_import("sentinel") is sentinel
    assert imports.optional_import("missing") is None
    # cached_import should have memoized after first call
    assert imports.optional_import("sentinel") is sentinel
    assert seen["sentinel"] >= 1


def test_require_dependency_raises(monkeypatch):
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ModuleNotFoundError()),
    )
    imports.cached_import.cache_clear()
    with pytest.raises(ImportError):
        imports.require_dependency("missing", "ctx")


def test_is_peft_model_safe_and_dist_fallback(monkeypatch):
    # No accelerate.utils available
    monkeypatch.setattr(fallbacks, "optional_import", lambda name: None)
    assert fallbacks.is_peft_model_safe(object()) is False

    # accelerate.utils present but missing callable
    accel_utils = SimpleNamespace(is_peft_model="not-callable")
    monkeypatch.setattr(fallbacks, "optional_import", lambda name: accel_utils)
    assert fallbacks.is_peft_model_safe(object()) is False

    # accelerate.utils present but raises
    accel_utils.is_peft_model = lambda *_a, **_k: (_ for _ in ()).throw(ValueError())
    monkeypatch.setattr(fallbacks, "optional_import", lambda name: accel_utils)
    assert fallbacks.is_peft_model_safe(object()) is False

    # accelerate.utils present and returns True
    accel_utils.is_peft_model = lambda *_a, **_k: True
    assert fallbacks.is_peft_model_safe(object()) is True

    class _Dist:
        pass

    dist = _Dist()
    wrapped = fallbacks.dist_with_fallback(dist)
    # Should attach missing attrs
    assert callable(wrapped.is_available)
    assert wrapped.get_world_size() == 1
    assert wrapped.is_initialized() is False
    out = [None]
    wrapped.all_gather_object(out, "wrapped")
    assert out[0] == "wrapped"
    assert wrapped.broadcast_object_list(["wrapped"], 0) is None
    # None input yields stub
    stub = fallbacks.dist_with_fallback(None)
    assert stub.is_available() is False and stub.get_world_size() == 1
    assert stub.is_initialized() is False
    out_stub = [None, "keep"]
    stub.all_gather_object(out_stub, "stub")
    assert out_stub == ["stub", "keep"]
    assert stub.broadcast_object_list(["stub"], 0) is None
