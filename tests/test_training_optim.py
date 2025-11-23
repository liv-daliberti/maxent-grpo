"""
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

from training import optim as opt  # noqa: E402


class _Param:
    def __init__(self, grad=1.0):
        self.grad = grad


class _Model:
    def __init__(self, grads):
        self._params = [_Param(g) for g in grads]

    def parameters(self):
        return self._params


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


def test_scheduled_learning_rate_handles_warmup_and_decay():
    handles = SimpleNamespace(learning_rate=1.0)
    schedule = SimpleNamespace(total_training_steps=10, warmup_steps=2)
    assert opt.scheduled_learning_rate(schedule, handles, step=1) == 0.5
    lr_mid = opt.scheduled_learning_rate(schedule, handles, step=5)
    assert 0.0 < lr_mid < 1.0


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
