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

Unit tests for training.optim utilities using lightweight stubs.
"""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from types import SimpleNamespace

torch_stub = types.ModuleType("torch")
torch_stub.Tensor = type("Tensor", (), {})
torch_stub.nn = SimpleNamespace(
    utils=SimpleNamespace(clip_grad_norm_=lambda *a, **k: 0.0)
)
sys.modules.setdefault("torch", torch_stub)
transformers_stub = types.ModuleType("transformers")
transformers_stub.PreTrainedModel = type("PreTrainedModel", (), {})
transformers_stub.PreTrainedTokenizer = type("PreTrainedTokenizer", (), {})
sys.modules.setdefault("transformers", transformers_stub)

import pytest  # noqa: E402

from maxent_grpo.training import optim as opt  # noqa: E402


class _Param:
    def __init__(self, grad=1.0):
        self.grad = grad


class _Model:
    def __init__(self, grads):
        self._params = [_Param(g) for g in grads]

    def parameters(self):
        return self._params


class _Optimizer:
    def __init__(self, lrs):
        self.param_groups = [{"lr": lr} for lr in lrs]
        self.step_called = 0
        self.zero_called = []

    def step(self):
        self.step_called += 1

    def zero_grad(self, set_to_none=False):
        self.zero_called.append(set_to_none)


def test_clip_grad_norm_local_uses_accelerator_clip(monkeypatch):
    model = _Model([1.0, None])

    class _Accel:
        def clip_grad_norm_(self, params, max_norm, norm_type=None):
            return 3.14

    grad = opt.clip_grad_norm_local(model, _Accel(), max_grad_norm=1.0)
    assert grad == pytest.approx(3.14)


def test_clip_grad_norm_local_returns_none_without_grads():
    model = _Model([None])
    grad = opt.clip_grad_norm_local(model, SimpleNamespace(), max_grad_norm=1.0)
    assert grad is None


def test_clip_grad_norm_local_skips_when_max_norm_non_positive():
    grad = opt.clip_grad_norm_local(_Model([1.0]), SimpleNamespace(), max_grad_norm=0)
    assert grad is None


def test_clip_grad_norm_local_retries_without_norm_type():
    model = _Model([1.0])

    class _Accel:
        def __init__(self):
            self.calls = []

        def clip_grad_norm_(self, params, max_norm, norm_type=None):
            if norm_type is not None:
                raise TypeError("legacy signature")
            self.calls.append((tuple(params), max_norm))
            return 5.0

    accel = _Accel()
    grad = opt.clip_grad_norm_local(model, accel, max_grad_norm=1.0)
    assert grad == pytest.approx(5.0)
    assert accel.calls == [(tuple(model.parameters()), 1.0)]


def test_clip_grad_norm_local_uses_torch_fallback_and_tensor_conversion(monkeypatch):
    model = _Model([1.0])

    class _Tensor:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def item(self):
            return self.value

    monkeypatch.setattr(opt.torch, "Tensor", _Tensor)
    monkeypatch.setattr(
        opt.torch, "nn", SimpleNamespace(utils=SimpleNamespace()), raising=False
    )
    monkeypatch.setattr(
        opt.torch.nn.utils,
        "clip_grad_norm_",
        lambda params, max_norm, norm_type=None: _Tensor(7.0),
        raising=False,
    )
    grad = opt.clip_grad_norm_local(model, SimpleNamespace(), max_grad_norm=1.0)
    assert grad == pytest.approx(7.0)


def test_clip_grad_norm_local_returns_none_on_unconvertible_value():
    class _Accel:
        def clip_grad_norm_(self, params, max_norm, norm_type=None):
            class _Bad:
                def __float__(self):
                    raise ValueError("nope")

            return _Bad()

    grad = opt.clip_grad_norm_local(_Model([1.0]), _Accel(), max_grad_norm=1.0)
    assert grad is None


def test_scheduled_learning_rate_handles_warmup_and_decay():
    handles = SimpleNamespace(learning_rate=1.0)
    schedule = SimpleNamespace(total_training_steps=10, warmup_steps=2)
    assert opt.scheduled_learning_rate(schedule, handles, step=1) == 0.5
    lr_mid = opt.scheduled_learning_rate(schedule, handles, step=5)
    assert 0.0 < lr_mid < 1.0


def test_apply_learning_rate_updates_all_parameter_groups():
    handles = SimpleNamespace(
        optimizer=_Optimizer([0.0, 0.1]),
        base_optimizer=_Optimizer([0.2]),
    )
    opt.apply_learning_rate(handles, learning_rate=0.5)
    for group in handles.optimizer.param_groups + handles.base_optimizer.param_groups:
        assert group["lr"] == 0.5


def test_optimizer_step_uses_accelerator_hook(monkeypatch):
    model = _Model([1.0])
    optimizer = _Optimizer([0.1])
    handles = SimpleNamespace(optimizer=optimizer, base_optimizer=optimizer, learning_rate=0.01)
    schedule = SimpleNamespace(max_grad_norm=1.5)

    class _Accel:
        def __init__(self):
            self.clip_calls = 0
            self.step_calls = 0

        def clip_grad_norm_(self, params, max_norm, norm_type=None):
            self.clip_calls += 1
            return 2.5

        def optimizer_step(self, opt_instance):
            self.step_calls += 1

    accel = _Accel()
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(accelerator=accel, model=model),
        optimization=SimpleNamespace(schedule=schedule, handles=handles),
    )
    state = SimpleNamespace(global_step=10)

    grad_norm = opt.optimizer_step(ctx, state, current_lr=0.25)
    assert grad_norm == pytest.approx(2.5)
    assert accel.step_calls == 1
    assert optimizer.zero_called == [True]
    assert state.global_step == 11
    assert handles.optimizer.param_groups[0]["lr"] == 0.25


def test_optimizer_step_falls_back_to_optimizer_step_method():
    model = _Model([1.0])
    optimizer = _Optimizer([0.3])

    class _Accel:
        def clip_grad_norm_(self, params, max_norm, norm_type=None):
            return 1.1

    accel = _Accel()
    handles = SimpleNamespace(optimizer=optimizer, base_optimizer=optimizer, learning_rate=0.5)
    ctx = SimpleNamespace(
        runtime=SimpleNamespace(accelerator=accel, model=model),
        optimization=SimpleNamespace(
            schedule=SimpleNamespace(max_grad_norm=2.0), handles=handles
        ),
    )
    state = SimpleNamespace(global_step=0)

    grad_norm = opt.optimizer_step(ctx, state, current_lr=0.75)
    assert grad_norm == pytest.approx(1.1)
    assert optimizer.step_called == 1
    assert optimizer.zero_called == [True]
    assert state.global_step == 1
    assert handles.optimizer.param_groups[0]["lr"] == 0.75


def test_epoch_progress_without_defined_steps_per_epoch():
    schedule = SimpleNamespace(steps_per_epoch=0)
    assert opt.epoch_progress(schedule, epoch=3, step_in_epoch=5) == pytest.approx(4.0)


def test_configure_accumulation_steps_prefers_setter(monkeypatch):
    called = {}

    class _GradState:
        def set_gradient_accumulation_steps(self, steps):
            called["state"] = steps

    class _Accel:
        def __init__(self):
            self.gradient_state = _GradState()

        def set_gradient_accumulation_steps(self, steps):
            called["accel"] = steps

    opt.configure_accumulation_steps(_Accel(), grad_accum_steps=4)
    assert called["accel"] == 4 or called["state"] == 4


def test_configure_accumulation_steps_sets_attribute_on_fallback():
    class _Accel:
        def __init__(self):
            self.gradient_state = None
            self.gradient_accumulation_steps = 1

        def set_gradient_accumulation_steps(self, steps):
            raise TypeError("signature mismatch")

    accel = _Accel()
    opt.configure_accumulation_steps(accel, grad_accum_steps=3)
    assert accel.gradient_accumulation_steps == 3


def test_detect_deepspeed_state_reads_plugin():
    accel = SimpleNamespace(
        state=SimpleNamespace(
            deepspeed_plugin=SimpleNamespace(zero_stage="2"),
            distributed_type="deepspeed",
        )
    )
    ds = opt.detect_deepspeed_state(accel)
    assert ds.use_deepspeed is True
    assert ds.zero_stage == 2


def test_require_accumulation_context_handles_deepspeed(monkeypatch):
    monkeypatch.setattr(
        opt, "detect_deepspeed_state", lambda accel: opt.DeepspeedState(True, 2)
    )
    ctx = opt.require_accumulation_context(SimpleNamespace(), model=None)
    assert hasattr(ctx, "__enter__")


def test_require_accumulation_context_calls_accumulate(monkeypatch):
    class _Accel:
        def accumulate(self, model):
            called["model"] = model

            @contextmanager
            def _ctx():
                yield

            return _ctx()

    called = {}
    monkeypatch.setattr(
        opt, "detect_deepspeed_state", lambda accel: opt.DeepspeedState(False, 0)
    )
    ctx = opt.require_accumulation_context(_Accel(), model="m")
    with ctx:
        pass
    assert called["model"] == "m"


def test_require_accumulation_context_raises_when_accumulate_missing():
    with pytest.raises(RuntimeError):
        opt.require_accumulation_context(SimpleNamespace(), model=None)