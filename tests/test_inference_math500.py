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

Unit tests for the math_500 inference helpers.
"""

from __future__ import annotations

import sys
import types
from typing import Sequence

import pytest

# Provide a tiny stub for `transformers` so importing `rewards` succeeds even in
# CI environments without the dependency installed.
if "transformers" not in sys.modules:  # pragma: no cover - import-time shim
    fake_tf = types.SimpleNamespace(
        PreTrainedModel=object,
        PreTrainedTokenizer=object,
        PreTrainedTokenizerBase=object,
    )
    sys.modules["transformers"] = fake_tf  # type: ignore[assignment]

from maxent_grpo.pipelines.inference.math500 import (  # noqa: E402 - after stub injection
    InferenceModelSpec,
    Math500EvalConfig,
    Math500InferenceResult,
    TransformersPromptRunner,
    _normalize_dtype,
    _resolve_default_device,
    load_math500_dataset,
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


def test_inference_requires_model_specs() -> None:
    with pytest.raises(ValueError):
        run_math500_inference([], dataset=[{"problem": "1", "answer": "1"}])


def test_runner_mismatch_counts_raise(monkeypatch):
    dataset = [{"problem": "p", "answer": "a"}]
    spec = InferenceModelSpec(model_name_or_path="stub/model", batch_size=1)

    class _Runner:
        def __init__(self, _spec):
            pass

        def generate(self, probs):
            return []  # wrong length

        def close(self):
            pass

    with pytest.raises(RuntimeError):
        run_math500_inference(
            [spec], dataset=dataset, runner_factory=lambda _s: _Runner(_s)
        )


def test_default_device_prefers_cuda_then_mps(monkeypatch):
    class _Torch:
        class _Backends:
            class _MPS:
                def is_available(self):
                    return True

        def __init__(self):
            self.cuda = types.SimpleNamespace(is_available=lambda: False)
            self.backends = types.SimpleNamespace(mps=self._Backends._MPS())

        class device:
            def __init__(self, name):
                self.type = name

    torch_stub = _Torch()
    torch_stub.backends = types.SimpleNamespace(mps=_Torch._Backends._MPS())
    monkeypatch.setattr("maxent_grpo.pipelines.inference.math500.torch", torch_stub)
    dev = _resolve_default_device(None)
    assert isinstance(dev, _Torch.device)
    assert dev.type == "mps"


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("auto", "auto"),
        (" float16 ", "float16"),
        ("unknown", "unknown"),
    ],
)
def test_normalize_dtype_handles_strings(value, expected, monkeypatch):
    class _Torch:
        class dtype:  # sentinel for isinstance checks
            pass

        float16 = "float16"
        backends = types.SimpleNamespace(mps=None)
        cuda = types.SimpleNamespace(is_available=lambda: False)

        def device(self, name):
            return types.SimpleNamespace(type=name)

    _Torch.dtype = _Torch.dtype  # ensure attribute exists for isinstance checks
    monkeypatch.setattr("maxent_grpo.pipelines.inference.math500.torch", _Torch)
    assert _normalize_dtype(value) == expected


def test_transformers_runner_builds_prompt_with_template(monkeypatch):
    class _Tok:
        def __init__(self):
            self.pad_token_id = 1
            self.eos_token_id = 2
            self.pad_token = "<pad>"
            self.chat_template = None

        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=True
        ):
            return f"TEMPLATE:{messages[0]['content']}"

        def __call__(self, prompts, **_kwargs):
            return {
                "input_ids": types.SimpleNamespace(to=lambda *_: [0]),
                "attention_mask": [2, 2],
            }

        def decode(self, _ids, **_kwargs):
            return "decoded"

    class _Model:
        def to(self, device):
            self.device = device

        def eval(self):
            pass

        def generate(self, **_kwargs):
            return [[0, 1, 2]]

    monkeypatch.setattr(
        "maxent_grpo.pipelines.inference.math500.AutoTokenizer",
        types.SimpleNamespace(from_pretrained=lambda *_a, **_k: _Tok()),
    )
    monkeypatch.setattr(
        "maxent_grpo.pipelines.inference.math500.AutoModelForCausalLM",
        types.SimpleNamespace(from_pretrained=lambda *_a, **_k: _Model()),
    )
    torch_stub = types.SimpleNamespace(
        no_grad=lambda: types.SimpleNamespace(
            __enter__=lambda *_: None, __exit__=lambda *_: None
        ),
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False)
        ),
        device=lambda name: types.SimpleNamespace(type=name),
        dtype=type("dtype", (), {}),
    )
    # Provide transformers stubs so the runner does not try to hit HF Hub
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace())
    monkeypatch.setattr("maxent_grpo.pipelines.inference.math500.torch", torch_stub)

    spec = InferenceModelSpec(model_name_or_path="stub", chat_template="<<template>>")
    runner = TransformersPromptRunner(spec)
    prompt = runner._build_prompt("abc")
    assert prompt.startswith("TEMPLATE:")


def test_load_math500_dataset_requires_datasets(monkeypatch):
    monkeypatch.setattr("maxent_grpo.pipelines.inference.math500.load_dataset", None)
    with pytest.raises(ImportError):
        load_math500_dataset(Math500EvalConfig())


def test_missing_columns_raise_friendly_error() -> None:
    """Surface readable errors when dataset columns are absent."""

    dataset = [{"prompt": "x=1"}]  # missing problem/answer
    spec = InferenceModelSpec(model_name_or_path="stub/model", batch_size=1)
    cfg = Math500EvalConfig(prompt_column="problem", solution_column="answer")
    with pytest.raises(ValueError) as excinfo:
        run_math500_inference(
            [spec],
            eval_cfg=cfg,
            dataset=dataset,
            runner_factory=lambda _spec: DummyRunner(_spec),
        )
    msg = str(excinfo.value)
    assert "missing required column 'problem'" in msg
    assert "available: ['prompt']" in msg


def test_empty_dataset_raises_friendly_error() -> None:
    """Ensure empty datasets raise a descriptive failure."""

    dataset: list[dict[str, str]] = []
    spec = InferenceModelSpec(model_name_or_path="stub/model", batch_size=1)
    cfg = Math500EvalConfig(prompt_column="problem", solution_column="answer")
    with pytest.raises(ValueError) as excinfo:
        run_math500_inference(
            [spec],
            eval_cfg=cfg,
            dataset=dataset,
            runner_factory=lambda _spec: DummyRunner(_spec),
        )
    assert "contained no usable rows" in str(excinfo.value)
