"""
Copyright 2025 Liv d'Aliberti

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Additional coverage for training.optim edge cases.
"""

from __future__ import annotations

from types import SimpleNamespace, ModuleType

import pytest

# Ensure src/ is importable when tests are run directly.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maxent_grpo.training import optim as opt


def test_clip_grad_norm_local_handles_zero_norm_returns_none():
    model = SimpleNamespace(parameters=lambda: [])
    accel = SimpleNamespace()
    assert opt.clip_grad_norm_local(model, accel, max_grad_norm=0.0) is None


def test_sync_gradients_enabled_logs_and_returns_flag(caplog):
    caplog.set_level("DEBUG")
    accel = SimpleNamespace(sync_gradients=False)
    assert opt.sync_gradients_enabled(accel, global_step=5) is False
    assert "sync_gradients=False" in caplog.text


def test_epoch_progress_with_steps_per_epoch():
    schedule = SimpleNamespace(steps_per_epoch=4)
    assert opt.epoch_progress(schedule, epoch=1, step_in_epoch=1) == pytest.approx(1.5)


def test_apply_learning_rate_skips_when_param_groups_missing():
    class _Opt:
        def __init__(self):
            self.param_groups = None

    handles = SimpleNamespace(optimizer=_Opt(), base_optimizer=_Opt())
    opt.apply_learning_rate(handles, learning_rate=0.1)  # should not raise


def test_configure_accumulation_steps_handles_missing_attributes():
    class _Accel:
        def __init__(self):
            self.gradient_state = None

    accel = _Accel()
    opt.configure_accumulation_steps(accel, grad_accum_steps=3)
    assert (
        not hasattr(accel, "gradient_accumulation_steps")
        or accel.gradient_accumulation_steps == 3
    )


def test_build_optimization_handles_applies_weight_decay_param_groups(monkeypatch):
    class _Param:
        def __init__(self, name, requires_grad=True):
            self.name = name
            self.requires_grad = requires_grad

    class _Model:
        def __init__(self):
            self._params = [
                _Param("layer1.weight"),
                _Param("layer1.bias"),
                _Param("encoder.LayerNorm.weight"),
                _Param("frozen.weight", requires_grad=False),
            ]

        def named_parameters(self):
            return [(p.name, p) for p in self._params]

        def parameters(self):
            return list(self._params)

    captured = {}

    def _adamw(param_groups, **kwargs):
        captured["param_groups"] = param_groups
        captured["kwargs"] = kwargs
        return SimpleNamespace(param_groups=param_groups, kwargs=kwargs)

    stub_torch = SimpleNamespace(optim=SimpleNamespace(AdamW=_adamw))
    monkeypatch.setattr(opt, "torch", stub_torch, raising=False)

    cfg = SimpleNamespace(
        learning_rate=0.01,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-6,
        optim="adamw_torch",
    )
    model = _Model()

    handles = opt.build_optimization_handles(model, cfg)
    param_groups = captured["param_groups"]

    assert handles.optimizer.param_groups is param_groups
    assert len(param_groups) == 2

    decay_group = next(pg for pg in param_groups if pg["weight_decay"] == pytest.approx(0.1))
    no_decay_group = next(pg for pg in param_groups if pg["weight_decay"] == pytest.approx(0.0))

    # layer1.weight should be in the decay group
    assert any(p.name == "layer1.weight" for p in decay_group["params"])
    # bias and LayerNorm.weight should be in the no‑decay group
    assert any(p.name == "layer1.bias" for p in no_decay_group["params"])
    assert any(p.name == "encoder.LayerNorm.weight" for p in no_decay_group["params"])
    # frozen parameters should be skipped entirely
    assert all(p.name != "frozen.weight" for pg in param_groups for p in pg["params"])

    kwargs = captured["kwargs"]
    assert kwargs["lr"] == pytest.approx(0.01)
    assert kwargs["betas"] == (pytest.approx(0.9), pytest.approx(0.95))
    assert kwargs["eps"] == pytest.approx(1e-6)
    assert "fused" not in kwargs


def test_build_optimization_handles_respects_fused_flag_fallback(monkeypatch):
    class _Param:
        def __init__(self, name):
            self.name = name
            self.requires_grad = True

    class _Model:
        def named_parameters(self):
            params = [_Param("w")]
            return [(p.name, p) for p in params]

        def parameters(self):
            return []

    calls = []

    def _adamw(param_groups, **kwargs):
        calls.append(kwargs.copy())
        if "fused" in kwargs:
            raise TypeError("fused not supported")
        return SimpleNamespace(param_groups=param_groups, kwargs=kwargs)

    stub_torch = SimpleNamespace(optim=SimpleNamespace(AdamW=_adamw))
    monkeypatch.setattr(opt, "torch", stub_torch, raising=False)

    cfg = SimpleNamespace(
        learning_rate=0.02,
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        optim="adamw_torch_fused",
    )
    model = _Model()

    handles = opt.build_optimization_handles(model, cfg)
    assert handles.learning_rate == pytest.approx(0.02)
    # First call should include fused=True, second should not.
    assert len(calls) == 2
    assert calls[0].get("fused") is True
    assert "fused" not in calls[1]


def test_build_optimization_handles_uses_bnb_adamw8bit_when_available(monkeypatch):
    class _Param:
        def __init__(self, name):
            self.name = name
            self.requires_grad = True

    class _Model:
        def named_parameters(self):
            params = [_Param("w1"), _Param("LayerNorm.weight")]
            return [(p.name, p) for p in params]

        def parameters(self):
            return []

    calls = []

    def _adamw8(param_groups, **kwargs):
        calls.append((param_groups, kwargs.copy()))
        return SimpleNamespace(param_groups=param_groups, kwargs=kwargs)

    bnb_mod = ModuleType("bitsandbytes")
    bnb_mod.optim = SimpleNamespace(AdamW8bit=_adamw8, PagedAdamW8bit=None)
    monkeypatch.setitem(sys.modules, "bitsandbytes", bnb_mod)

    stub_torch = SimpleNamespace(optim=SimpleNamespace(AdamW=None))
    monkeypatch.setattr(opt, "torch", stub_torch, raising=False)

    cfg = SimpleNamespace(
        learning_rate=0.01,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-6,
        optim="adamw_bnb_8bit",
    )
    model = _Model()

    handles = opt.build_optimization_handles(model, cfg)
    assert handles.learning_rate == pytest.approx(0.01)
    assert calls
    param_groups, kwargs = calls[0]
    assert kwargs["lr"] == pytest.approx(0.01)
    assert kwargs["betas"] == (pytest.approx(0.9), pytest.approx(0.95))
    assert kwargs["eps"] == pytest.approx(1e-6)
    assert kwargs["weight_decay"] == pytest.approx(0.1)
    assert len(param_groups) == 2


def test_build_optimization_handles_falls_back_when_bitsandbytes_missing(monkeypatch):
    class _Param:
        def __init__(self, name):
            self.name = name
            self.requires_grad = True

    class _Model:
        def named_parameters(self):
            params = [_Param("w")]
            return [(p.name, p) for p in params]

        def parameters(self):
            return []

    calls = []

    def _adamw(param_groups, **kwargs):
        calls.append(kwargs.copy())
        return SimpleNamespace(param_groups=param_groups, kwargs=kwargs)

    monkeypatch.delitem(sys.modules, "bitsandbytes", raising=False)
    stub_torch = SimpleNamespace(optim=SimpleNamespace(AdamW=_adamw))
    monkeypatch.setattr(opt, "torch", stub_torch, raising=False)

    cfg = SimpleNamespace(
        learning_rate=0.03,
        weight_decay=0.2,
        adam_beta1=0.9,
        adam_beta2=0.99,
        adam_epsilon=1e-8,
        optim="adamw_bnb_8bit",
    )
    model = _Model()

    handles = opt.build_optimization_handles(model, cfg)
    assert handles.learning_rate == pytest.approx(0.03)
    assert calls
    kwargs = calls[0]
    assert kwargs["lr"] == pytest.approx(0.03)
    assert kwargs["betas"] == (pytest.approx(0.9), pytest.approx(0.99))
    assert kwargs["eps"] == pytest.approx(1e-8)
