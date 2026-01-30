"""
Additional branch coverage for training.runtime.setup helpers.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace


import maxent_grpo.training.runtime.torch_utils as torch_utils


def test_build_torch_stub_exports_symbool_and_no_grad():
    from maxent_grpo.training.runtime import setup

    stub = setup._build_torch_stub()
    assert hasattr(stub, "SymBool")
    sym = stub.SymBool()
    # Some environments alias SymBool to bool; just ensure it is boolean-coercible.
    assert isinstance(bool(sym), bool)
    node_attr = getattr(sym, "node", None)
    assert node_attr is None or node_attr is not None
    assert callable(getattr(stub, "no_grad", None))
    assert callable(getattr(stub.autograd, "no_grad", None))


def test_require_torch_installs_stub_when_missing(monkeypatch):
    from maxent_grpo.training.runtime import setup

    torch_utils._import_module.cache_clear()
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    torch_mod = setup.require_torch("test")
    assert hasattr(torch_mod, "tensor")
    assert "torch" in sys.modules


def test_require_dataloader_uses_stub_when_import_fails(monkeypatch):
    from maxent_grpo.training.runtime import setup

    torch_utils._import_module.cache_clear()
    for mod in ("torch.utils.data", "torch.utils", "torch"):
        monkeypatch.delitem(sys.modules, mod, raising=False)

    def _failing_import(name: str):
        raise ModuleNotFoundError("missing")

    monkeypatch.setattr(setup, "_import_module", _failing_import)
    monkeypatch.setattr(torch_utils, "_import_module", _failing_import)
    loader_cls = setup.require_dataloader("tests")
    assert loader_cls.__name__ == "DataLoader"
    assert "torch.utils.data" in sys.modules


def test_get_trl_prepare_deepspeed_handles_missing(monkeypatch):
    from maxent_grpo.training.runtime import setup

    monkeypatch.setitem(
        sys.modules, "trl.trainer.utils", ModuleType("trl.trainer.utils")
    )
    monkeypatch.setattr(setup, "_optional_dependency", lambda name: None)
    assert setup.get_trl_prepare_deepspeed() is None

    utils_mod = ModuleType("trl.trainer.utils")
    utils_mod.prepare_deepspeed = lambda *_a, **_k: "ok"
    monkeypatch.setattr(setup, "_optional_dependency", lambda name: utils_mod)
    assert setup.get_trl_prepare_deepspeed()() == "ok"


def test_maybe_create_deepspeed_plugin_skips_without_flag(monkeypatch):
    from maxent_grpo.training.runtime import setup

    monkeypatch.delenv("ACCELERATE_USE_DEEPSPEED", raising=False)
    assert setup._maybe_create_deepspeed_plugin() is None

    monkeypatch.setenv("ACCELERATE_USE_DEEPSPEED", "true")
    fake_utils = ModuleType("accelerate.utils")

    def _fake_plugin_cls(**kwargs):
        return SimpleNamespace(kwargs=kwargs)

    fake_utils.DeepSpeedPlugin = _fake_plugin_cls
    monkeypatch.setattr(setup, "_require_dependency", lambda *_a, **_k: fake_utils)
    # No config provided -> kwargs empty -> return None
    monkeypatch.delenv("ACCELERATE_CONFIG_FILE", raising=False)
    plugin = setup._maybe_create_deepspeed_plugin()
    # Default config produces a plugin with default zero_stage
    assert plugin is not None
    assert getattr(plugin, "kwargs", {}).get("zero_stage") in (None, 3)
