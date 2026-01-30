"""Unit coverage for training.runtime.deps dependency helpers."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


def test_require_torch_installs_stub_via_bootstrap(monkeypatch):
    from maxent_grpo.training.runtime import deps

    deps._import_module.cache_clear()
    for mod in ("torch", "sitecustomize"):
        monkeypatch.delitem(sys.modules, mod, raising=False)

    installed = SimpleNamespace(
        tensor=lambda *_a, **_k: "t",
        zeros=lambda *_a, **_k: "z",
        full=lambda *_a, **_k: "f",
        ones_like=lambda *_a, **_k: "o",
    )

    def _install_stub():
        sys.modules["torch"] = installed

    bootstrap = ModuleType("sitecustomize")
    bootstrap._install_torch_stub = _install_stub
    ops_pkg = ModuleType("ops")
    ops_pkg.sitecustomize = bootstrap
    monkeypatch.setitem(sys.modules, "ops", ops_pkg)
    monkeypatch.setitem(sys.modules, "sitecustomize", bootstrap)

    def fake_import(name: str):
        if name == "torch":
            if name in sys.modules:
                return sys.modules[name]
            raise ModuleNotFoundError("missing")
        return importlib.import_module(name)

    fake_import.cache_clear = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setattr(deps, "_import_module", fake_import)

    torch_mod = deps.require_torch("deps-test")
    assert torch_mod is installed
    assert torch_mod.full((1,), 5) == "f"


def test_require_torch_missing_attrs_builds_stub(monkeypatch):
    from maxent_grpo.training.runtime import deps

    deps._import_module.cache_clear()
    for mod in ("torch", "sitecustomize"):
        monkeypatch.delitem(sys.modules, mod, raising=False)

    torch_attempts = [
        SimpleNamespace(tensor=lambda *_a, **_k: None),
        SimpleNamespace(tensor=lambda *_a, **_k: None),
    ]

    def fake_import(name: str):
        if name == "torch":
            return torch_attempts.pop(0)
        return importlib.import_module(name)

    fake_import.cache_clear = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setattr(deps, "_import_module", fake_import)
    bootstrap = ModuleType("sitecustomize")
    bootstrap._install_torch_stub = lambda: None
    ops_pkg = ModuleType("ops")
    ops_pkg.sitecustomize = bootstrap
    monkeypatch.setitem(sys.modules, "ops", ops_pkg)
    monkeypatch.setitem(sys.modules, "sitecustomize", bootstrap)

    torch_mod = deps.require_torch("deps-missing")
    assert hasattr(torch_mod, "Tensor")
    assert torch_mod.zeros((1,)).tolist() == [0]


def test_require_dataloader_builds_minimal_stub(monkeypatch):
    from maxent_grpo.training.runtime import deps

    deps._import_module.cache_clear()
    for mod in ("torch.utils.data", "torch.utils", "torch"):
        monkeypatch.delitem(sys.modules, mod, raising=False)

    def failing_import(name: str):
        raise ModuleNotFoundError("missing")

    failing_import.cache_clear = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setattr(deps, "_import_module", failing_import)
    loader_cls = deps.require_dataloader("deps")
    assert loader_cls.__name__ == "DataLoader"
    assert "torch.utils.data" in sys.modules


def test_require_dataloader_raises_when_import_returns_none(monkeypatch):
    from maxent_grpo.training.runtime import deps

    deps._import_module.cache_clear()
    for mod in ("torch.utils.data", "torch.utils", "torch"):
        monkeypatch.delitem(sys.modules, mod, raising=False)

    def import_none(name: str):
        return None

    import_none.cache_clear = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setattr(deps, "_import_module", import_none)

    with pytest.raises(RuntimeError) as excinfo:
        deps.require_dataloader("deps-none")

    assert "DataLoader is required" in str(excinfo.value)


def test_require_accelerator_paths(monkeypatch):
    from maxent_grpo.training.runtime import deps

    deps._import_module.cache_clear()
    monkeypatch.setattr(
        deps,
        "_import_module",
        lambda name: (_ for _ in ()).throw(ModuleNotFoundError()),
    )
    with pytest.raises(RuntimeError):
        deps.require_accelerator("ctx")

    accel_mod = SimpleNamespace()
    monkeypatch.setattr(deps, "_import_module", lambda name: accel_mod)
    with pytest.raises(RuntimeError):
        deps.require_accelerator("ctx")

    accel_mod.Accelerator = object
    assert deps.require_accelerator("ctx") is accel_mod.Accelerator


def test_require_transformer_base_classes(monkeypatch):
    from maxent_grpo.training.runtime import deps

    deps._import_module.cache_clear()
    monkeypatch.setattr(
        deps,
        "_import_module",
        lambda name: (_ for _ in ()).throw(ModuleNotFoundError()),
    )
    with pytest.raises(RuntimeError):
        deps.require_transformer_base_classes("ctx")

    transformers_mod = SimpleNamespace()
    monkeypatch.setattr(deps, "_import_module", lambda name: transformers_mod)
    with pytest.raises(RuntimeError):
        deps.require_transformer_base_classes("ctx")

    transformers_mod.PreTrainedModel = type("Model", (), {})
    transformers_mod.PreTrainedTokenizer = type("Tokenizer", (), {})
    assert deps.require_transformer_base_classes("ctx") == (
        transformers_mod.PreTrainedModel,
        transformers_mod.PreTrainedTokenizer,
    )


def test_require_deepspeed_wraps_importerror(monkeypatch):
    from maxent_grpo.training.runtime import deps

    monkeypatch.setattr(
        deps,
        "_require_dependency",
        lambda *_a, **_k: (_ for _ in ()).throw(ImportError()),
    )
    with pytest.raises(RuntimeError):
        deps.require_deepspeed("ctx")

    ds_module = ModuleType("deepspeed")
    monkeypatch.setattr(deps, "_require_dependency", lambda *_a, **_k: ds_module)
    assert deps.require_deepspeed("ctx") is ds_module


def test_get_trl_prepare_deepspeed(monkeypatch):
    from maxent_grpo.training.runtime import deps

    monkeypatch.setattr(deps, "_optional_dependency", lambda name: None)
    assert deps.get_trl_prepare_deepspeed() is None

    utils_mod = ModuleType("trl.trainer.utils")
    utils_mod.prepare_deepspeed = "not-callable"
    monkeypatch.setattr(deps, "_optional_dependency", lambda name: utils_mod)
    assert deps.get_trl_prepare_deepspeed() is None

    utils_mod.prepare_deepspeed = lambda *_a, **_k: "ok"
    assert deps.get_trl_prepare_deepspeed()() == "ok"


def test_maybe_create_deepspeed_plugin_guard(monkeypatch):
    from maxent_grpo.training.runtime import deps

    monkeypatch.delenv("ACCELERATE_USE_DEEPSPEED", raising=False)
    assert deps._maybe_create_deepspeed_plugin() is None

    monkeypatch.setenv("ACCELERATE_USE_DEEPSPEED", "true")
    fake_utils = SimpleNamespace(
        DeepSpeedPlugin=lambda **kwargs: SimpleNamespace(kwargs=kwargs)
    )
    monkeypatch.setattr(deps, "_require_dependency", lambda *_a, **_k: fake_utils)
    monkeypatch.delenv("ACCELERATE_CONFIG_FILE", raising=False)
    plugin = deps._maybe_create_deepspeed_plugin()
    assert plugin.kwargs.get("zero_stage") == 3


def test_maybe_create_deepspeed_plugin_parses_config(monkeypatch, tmp_path):
    from maxent_grpo.training.runtime import deps

    if deps.yaml is None:
        pytest.skip("PyYAML is unavailable")

    monkeypatch.setenv("ACCELERATE_USE_DEEPSPEED", "true")
    cfg_path = tmp_path / "accel.yaml"
    cfg_path.write_text(
        deps.yaml.safe_dump(
            {
                "deepspeed_config": {
                    "zero_stage": None,
                    "offload_optimizer_device": "cpu",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ACCELERATE_CONFIG_FILE", str(cfg_path))
    captured_kwargs = {}

    class _Plugin:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    fake_utils = SimpleNamespace(DeepSpeedPlugin=_Plugin)
    monkeypatch.setattr(deps, "_require_dependency", lambda *_a, **_k: fake_utils)
    plugin = deps._maybe_create_deepspeed_plugin()
    assert isinstance(plugin, _Plugin)
    assert captured_kwargs == {"offload_optimizer_device": "cpu"}


def test_require_torch_missing_bootstrap_falls_back(monkeypatch):
    from maxent_grpo.training.runtime import deps

    deps._import_module.cache_clear()
    for mod in ("torch", "ops", "sitecustomize"):
        monkeypatch.delitem(sys.modules, mod, raising=False)

    incomplete = SimpleNamespace(tensor=lambda *_a, **_k: None)

    def fake_import(name: str):
        if name == "torch":
            return incomplete
        return importlib.import_module(name)

    fake_import.cache_clear = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setattr(deps, "_import_module", fake_import)
    monkeypatch.setitem(sys.modules, "ops", ModuleType("ops"))
    monkeypatch.delitem(sys.modules, "sitecustomize", raising=False)

    torch_mod = deps.require_torch("missing-bootstrap")
    assert torch_mod is not incomplete
    assert hasattr(torch_mod, "zeros")


def test_maybe_create_deepspeed_plugin_handles_parse_error(monkeypatch, tmp_path):
    from maxent_grpo.training.runtime import deps

    if deps.yaml is None:
        pytest.skip("PyYAML is unavailable")

    monkeypatch.setenv("ACCELERATE_USE_DEEPSPEED", "true")
    cfg_path = tmp_path / "broken.yaml"
    cfg_path.write_text(":\n- invalid", encoding="utf-8")
    monkeypatch.setenv("ACCELERATE_CONFIG_FILE", str(cfg_path))
    captured_kwargs: dict = {}

    class _Plugin:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    fake_utils = SimpleNamespace(DeepSpeedPlugin=_Plugin)
    monkeypatch.setattr(deps, "_require_dependency", lambda *_a, **_k: fake_utils)
    plugin = deps._maybe_create_deepspeed_plugin()
    assert isinstance(plugin, _Plugin)
    assert captured_kwargs == {"zero_stage": 3}


def test_maybe_create_deepspeed_plugin_empty_kwargs_returns_none(monkeypatch, tmp_path):
    from maxent_grpo.training.runtime import deps

    if deps.yaml is None:
        pytest.skip("PyYAML is unavailable")

    monkeypatch.setenv("ACCELERATE_USE_DEEPSPEED", "true")
    cfg_path = tmp_path / "empty.yaml"
    cfg_path.write_text(
        deps.yaml.safe_dump({"deepspeed_config": {"zero_stage": None}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ACCELERATE_CONFIG_FILE", str(cfg_path))
    fake_utils = SimpleNamespace(DeepSpeedPlugin=lambda **kwargs: kwargs)
    monkeypatch.setattr(deps, "_require_dependency", lambda *_a, **_k: fake_utils)
    assert deps._maybe_create_deepspeed_plugin() is None
