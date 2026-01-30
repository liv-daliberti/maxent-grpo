"""Additional coverage for training.runtime.setup edge paths."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

import maxent_grpo.training.runtime.setup as setup_mod
import maxent_grpo.training.runtime.torch_utils as torch_utils


def test_require_torch_builds_stub_when_missing_attrs(monkeypatch):
    """Cover the secondary fallback path when torch is missing required attrs."""

    calls = {}

    def _fake_import(name: str):
        calls.setdefault("names", []).append(name)
        if name == "torch":
            # Present but missing zeros/full/ones_like
            return SimpleNamespace(tensor=lambda *_a, **_k: "tensor-only")
        if name == "sitecustomize":
            raise ImportError("no stub installer")
        return __import__(name)

    _fake_import.cache_clear = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setattr(torch_utils, "_import_module", _fake_import)
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    torch_mod = setup_mod.require_torch("ctx")
    # Stub should be built with the required helpers.
    assert callable(torch_mod.zeros)
    assert callable(torch_mod.full)
    sym_cls = getattr(torch_mod, "SymBool", None)
    if sym_cls is not None:
        assert bool(sym_cls(True)) is True
    # Clean up stubbed torch modules so other tests can install their own stubs.
    for name in ("torch", "torch.utils", "torch.utils.data"):
        monkeypatch.delitem(sys.modules, name, raising=False)


def test_require_deepspeed_wraps_import_error(monkeypatch):
    monkeypatch.setattr(
        setup_mod,
        "_require_dependency",
        lambda *_a, **_k: (_ for _ in ()).throw(ImportError("missing")),
    )
    with pytest.raises(RuntimeError):
        setup_mod.require_deepspeed("ctx")


def test_maybe_create_deepspeed_plugin_reads_config(tmp_path, monkeypatch):
    cfg_path = tmp_path / "acc.yaml"
    cfg_path.write_text(
        "deepspeed_config:\n"
        "  zero_stage: 2\n"
        "  offload_param_device: cpu\n"
        "  offload_optimizer_device: cpu\n"
        "  zero3_init_flag: true\n"
        "  zero3_save_16bit_model: false\n",
        encoding="utf-8",
    )

    captured = {}

    class _DSP:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    ds_utils = ModuleType("accelerate.utils")
    ds_utils.DeepSpeedPlugin = _DSP
    monkeypatch.setattr(setup_mod, "_require_dependency", lambda *_a, **_k: ds_utils)
    monkeypatch.setenv("ACCELERATE_USE_DEEPSPEED", "true")
    monkeypatch.setenv("ACCELERATE_CONFIG_FILE", str(cfg_path))

    plugin = setup_mod._maybe_create_deepspeed_plugin()
    assert isinstance(plugin, _DSP)
    assert captured["zero_stage"] == 2
    assert captured["offload_param_device"] == "cpu"
    assert captured["offload_optimizer_device"] == "cpu"
    assert captured["zero3_init_flag"] is True
    assert captured["zero3_save_16bit_model"] is False
