"""Unit tests for training.runtime.setup helpers."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from maxent_grpo.training.runtime import setup as rt_setup


def test_require_dataloader_installs_stub(monkeypatch):
    # Remove torch.utils.data to force stub path
    torch_mod = ModuleType("torch")
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.utils", ModuleType("torch.utils"))
    monkeypatch.setitem(sys.modules, "torch.utils.data", None)
    dl = rt_setup.require_dataloader("test")
    assert getattr(dl, "__name__", "") == "DataLoader"


def test_require_transformer_base_classes_missing(monkeypatch):
    rt_setup._import_module.cache_clear()
    monkeypatch.setattr(
        rt_setup,
        "_import_module",
        lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name)),
    )
    with pytest.raises(RuntimeError):
        rt_setup.require_transformer_base_classes("ctx")


def test_maybe_create_deepspeed_plugin_env_disabled(monkeypatch):
    monkeypatch.delenv("ACCELERATE_USE_DEEPSPEED", raising=False)
    assert rt_setup._maybe_create_deepspeed_plugin() is None


def test_maybe_create_deepspeed_plugin_returns_none_on_empty_cfg(monkeypatch):
    monkeypatch.setenv("ACCELERATE_USE_DEEPSPEED", "true")
    # Provide minimal accelerate.utils stub without DeepSpeedPlugin to hit guard.
    accel_utils = ModuleType("accelerate.utils")
    monkeypatch.setitem(sys.modules, "accelerate.utils", accel_utils)
    with pytest.raises(ImportError):
        rt_setup._maybe_create_deepspeed_plugin()


def test_get_trl_prepare_deepspeed(monkeypatch):
    rt_setup._import_module.cache_clear()
    utils_mod = SimpleNamespace(prepare_deepspeed=lambda: "prep")
    monkeypatch.setitem(sys.modules, "trl.trainer.utils", utils_mod)
    assert rt_setup.get_trl_prepare_deepspeed()() == "prep"
    # replace with non-callable to hit None branch
    rt_setup._import_module.cache_clear()
    monkeypatch.setitem(sys.modules, "trl.trainer.utils", SimpleNamespace())
    assert rt_setup.get_trl_prepare_deepspeed() is None
