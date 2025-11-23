"""Unit tests for the math_500 inference helpers."""

from __future__ import annotations

import sys
import types
from typing import Sequence

# Provide a tiny stub for `transformers` so importing `rewards` succeeds even in
# CI environments without the dependency installed.
if "transformers" not in sys.modules:  # pragma: no cover - import-time shim
    fake_tf = types.SimpleNamespace(
        PreTrainedModel=object,
        PreTrainedTokenizer=object,
        PreTrainedTokenizerBase=object,
    )
    sys.modules["transformers"] = fake_tf  # type: ignore[assignment]

from inference.math500 import (  # noqa: E402 - after stub injection
    InferenceModelSpec,
    Math500EvalConfig,
    Math500InferenceResult,
    run_math500_inference,
)


class DummyRunner:
    """Minimal prompt runner that derives answers from the prompt suffix."""

    def __init__(self, _spec: InferenceModelSpec):
        self.closed = False
        self.calls: list[Sequence[str]] = []

    def generate(self, problems: Sequence[str]) -> list[str]:
        """Return `<answer>` tags seeded from the problem string."""

        self.calls.append(tuple(problems))
        outputs: list[str] = []
        for text in problems:
            parsed = text.split("=")[-1]
            outputs.append(f"<think></think><answer>{parsed}</answer>")
        return outputs

    def close(self) -> None:
        self.closed = True


def test_run_math500_inference_accumulates_accuracy() -> None:
    """Verify accuracy/correct counts are derived from reward hits."""

    dataset = [
        {"problem": "ans=42", "answer": "42"},
        {"problem": "ans=7", "answer": "13"},
    ]
    spec = InferenceModelSpec(
        model_name_or_path="stub/model",
        batch_size=1,
        style="maxent",
    )
    config = Math500EvalConfig(prompt_column="problem", solution_column="answer")
    results = run_math500_inference(
        [spec],
        eval_cfg=config,
        dataset=dataset,
        runner_factory=lambda _spec: DummyRunner(_spec),
    )
    assert len(results) == 1
    res: Math500InferenceResult = results[0]
    assert res.total == 2
    assert res.correct == 1
    assert abs(res.accuracy - 0.5) < 1e-6


def test_collect_generations_captures_raw_outputs() -> None:
    """Ensure the `collect_generations` flag stores completions in results."""

    dataset = [
        {"problem": "ans=101", "answer": "101"},
        {"problem": "ans=202", "answer": "202"},
    ]
    spec = InferenceModelSpec(model_name_or_path="stub/model", batch_size=2)
    config = Math500EvalConfig(
        prompt_column="problem", solution_column="answer", limit=5
    )
    results = run_math500_inference(
        [spec],
        eval_cfg=config,
        dataset=dataset,
        collect_generations=True,
        limit=1,
        runner_factory=lambda _spec: DummyRunner(_spec),
    )
    generated = results[0].generations
    assert generated is not None
    assert generated == ["<think></think><answer>101</answer>"]
