"""Unit tests for the math_500 inference pipeline utilities."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from pipelines.inference import math500 as math_mod


def test_prepare_examples_respects_limit():
    dataset = [{"problem": "1+1", "answer": "2"}, {"problem": "2+2", "answer": "4"}]
    cfg = math_mod.Math500EvalConfig(limit=None)
    examples = math_mod._prepare_examples(dataset, cfg, limit=1)
    assert examples == [("1+1", "2")]


def test_prepare_examples_raises_on_empty():
    with pytest.raises(ValueError):
        math_mod._prepare_examples([], math_mod.Math500EvalConfig(), limit=None)


def test_inference_runner_factory_and_collect(monkeypatch):
    calls: Dict[str, Any] = {}

    class _Runner(math_mod.PromptRunner):
        def __init__(self, spec):
            calls["spec"] = spec

        def generate(self, problems: List[str]) -> List[str]:
            return ["yes" for _ in problems]

        def close(self) -> None:
            calls["closed"] = True

    dataset = [{"problem": "p", "answer": "yes"}]
    specs = [
        math_mod.InferenceModelSpec(model_name_or_path="m1", label="L", style="grpo")
    ]
    results = math_mod.run_math500_inference(
        specs,
        eval_cfg=math_mod.Math500EvalConfig(
            prompt_column="problem", solution_column="answer"
        ),
        dataset=dataset,
        collect_generations=True,
        runner_factory=lambda spec: _Runner(spec),
    )
    assert results[0].total == 1
    assert results[0].generations == ["yes"]
    assert calls["closed"] is True


def test_inference_validates_runner_lengths():
    class _BadRunner(math_mod.PromptRunner):
        def __init__(self, _spec):
            pass

        def generate(self, problems: List[str]) -> List[str]:
            return []

        def close(self) -> None:
            return None

    dataset = [{"problem": "p", "answer": "yes"}]
    specs = [math_mod.InferenceModelSpec(model_name_or_path="m1")]
    with pytest.raises(RuntimeError):
        math_mod.run_math500_inference(
            specs,
            eval_cfg=math_mod.Math500EvalConfig(
                prompt_column="problem", solution_column="answer"
            ),
            dataset=dataset,
            runner_factory=lambda spec: _BadRunner(spec),
        )
