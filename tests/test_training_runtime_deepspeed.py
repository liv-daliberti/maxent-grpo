"""Unit coverage for training.runtime.deepspeed helpers."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace

import pytest


def test_require_deepspeed_wraps_importerror(monkeypatch):
    from maxent_grpo.training.runtime import deepspeed as ds

    monkeypatch.setattr(
        ds,
        "_require_dependency",
        lambda *_a, **_k: (_ for _ in ()).throw(ImportError()),
    )
    with pytest.raises(RuntimeError):
        ds.require_deepspeed("ctx")

    ds_module = ModuleType("deepspeed")
    monkeypatch.setattr(ds, "_require_dependency", lambda *_a, **_k: ds_module)
    assert ds.require_deepspeed("ctx") is ds_module


def test_get_trl_prepare_deepspeed(monkeypatch):
    from maxent_grpo.training.runtime import deepspeed as ds

    monkeypatch.setattr(ds, "_optional_dependency", lambda name: None)
    assert ds.get_trl_prepare_deepspeed() is None

    utils_mod = ModuleType("trl.trainer.utils")
    utils_mod.prepare_deepspeed = "not-callable"
    monkeypatch.setattr(ds, "_optional_dependency", lambda name: utils_mod)
    assert ds.get_trl_prepare_deepspeed() is None

    utils_mod.prepare_deepspeed = lambda *_a, **_k: "ok"
    assert ds.get_trl_prepare_deepspeed()() == "ok"


def test_maybe_create_deepspeed_plugin_guard(monkeypatch):
    from maxent_grpo.training.runtime import deepspeed as ds

    monkeypatch.delenv("ACCELERATE_USE_DEEPSPEED", raising=False)
    assert ds._maybe_create_deepspeed_plugin() is None


def test_maybe_create_deepspeed_plugin_parses_config(monkeypatch, tmp_path):
    from maxent_grpo.training.runtime import deepspeed as ds

    if ds.yaml is None:
        pytest.skip("PyYAML is unavailable")

    monkeypatch.setenv("ACCELERATE_USE_DEEPSPEED", "true")
    cfg_path = tmp_path / "accel.yaml"
    cfg_path.write_text(
        ds.yaml.safe_dump(
            {"deepspeed_config": {"zero_stage": None, "offload_param_device": "nvme"}}
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ACCELERATE_CONFIG_FILE", str(cfg_path))
    captured_kwargs = {}

    class _Plugin:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    fake_utils = SimpleNamespace(DeepSpeedPlugin=_Plugin)
    monkeypatch.setattr(ds, "_require_dependency", lambda *_a, **_k: fake_utils)
    plugin = ds._maybe_create_deepspeed_plugin()
    assert isinstance(plugin, _Plugin)
    assert captured_kwargs == {"offload_param_device": "nvme"}


def test_maybe_create_deepspeed_plugin_handles_yaml_error(monkeypatch, tmp_path):
    from maxent_grpo.training.runtime import deepspeed as ds

    if ds.yaml is None:
        pytest.skip("PyYAML is unavailable")

    monkeypatch.setenv("ACCELERATE_USE_DEEPSPEED", "true")
    cfg_path = tmp_path / "broken.yaml"
    cfg_path.write_text(":\n- invalid", encoding="utf-8")
    monkeypatch.setenv("ACCELERATE_CONFIG_FILE", str(cfg_path))
    captured_kwargs = {}

    class _Plugin:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    fake_utils = SimpleNamespace(DeepSpeedPlugin=_Plugin)
    monkeypatch.setattr(ds, "_require_dependency", lambda *_a, **_k: fake_utils)
    plugin = ds._maybe_create_deepspeed_plugin()
    assert isinstance(plugin, _Plugin)
    assert captured_kwargs == {"zero_stage": 3}


def test_maybe_create_deepspeed_plugin_empty_kwargs(monkeypatch, tmp_path):
    from maxent_grpo.training.runtime import deepspeed as ds

    if ds.yaml is None:
        pytest.skip("PyYAML is unavailable")

    monkeypatch.setenv("ACCELERATE_USE_DEEPSPEED", "true")
    cfg_path = tmp_path / "empty.yaml"
    cfg_path.write_text(
        ds.yaml.safe_dump({"deepspeed_config": {"zero_stage": None}}), encoding="utf-8"
    )
    monkeypatch.setenv("ACCELERATE_CONFIG_FILE", str(cfg_path))
    fake_utils = SimpleNamespace(DeepSpeedPlugin=lambda **kwargs: kwargs)
    monkeypatch.setattr(ds, "_require_dependency", lambda *_a, **_k: fake_utils)
    assert ds._maybe_create_deepspeed_plugin() is None
