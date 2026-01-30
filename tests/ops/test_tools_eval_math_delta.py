"""
Unit tests for tools/eval_math_delta.py.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace, ModuleType
import sys

import pytest


from tools import eval_math_delta
from maxent_grpo.pipelines.inference.inference import (
    InferenceModelSpec,
    MathEvalConfig,
)


def _install_torch_stub(monkeypatch):
    torch_stub = ModuleType("torch")
    torch_stub.cuda = SimpleNamespace(is_available=lambda: False)
    torch_stub.backends = SimpleNamespace(
        mps=SimpleNamespace(is_available=lambda: False)
    )
    torch_stub.device = lambda name=None: f"dev:{name or 'cpu'}"

    @contextmanager
    def _no_grad():
        yield

    torch_stub.no_grad = _no_grad
    torch_stub.float16 = "float16"
    torch_stub.bfloat16 = "bfloat16"
    torch_stub.float32 = "float32"
    monkeypatch.setitem(sys.modules, "torch", torch_stub)


def test_evaluate_delta_stub_runner_matches_baseline(monkeypatch):
    _install_torch_stub(monkeypatch)
    dataset = [
        {"problem": "1+1", "answer": "2"},
        {"problem": "2+2", "answer": "4"},
    ]
    cfg = MathEvalConfig(limit=2)
    base = InferenceModelSpec(model_name_or_path="base", label="baseline")
    cand = InferenceModelSpec(model_name_or_path="cand", label="candidate")
    delta = eval_math_delta.evaluate_delta(
        base, cand, dataset=dataset, eval_cfg=cfg, runner="stub"
    )
    assert delta.baseline_acc == delta.candidate_acc
    assert delta.delta == pytest.approx(0.0)
