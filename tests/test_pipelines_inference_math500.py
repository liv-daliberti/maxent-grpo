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

import types
from types import SimpleNamespace

import pytest

from maxent_grpo.pipelines.inference import math500


def test_inference_model_spec_resolve_label_defaults():
    spec = math500.InferenceModelSpec(model_name_or_path="org/model", style="")
    assert spec.resolve_label() == "model"
    labeled = math500.InferenceModelSpec(model_name_or_path="org/model", label="custom")
    assert labeled.resolve_label() == "custom"


def test_prepare_examples_validates_columns_and_nonempty():
    cfg = math500.Math500EvalConfig(prompt_column="p", solution_column="a")
    with pytest.raises(ValueError):
        math500._prepare_examples([], cfg, limit=None)
    bad_row = [{"p": "x"}]
    with pytest.raises(ValueError):
        math500._prepare_examples(bad_row, cfg, limit=None)
    cfg_missing_solution = math500.Math500EvalConfig(
        prompt_column="p", solution_column="missing"
    )
    with pytest.raises(ValueError):
        math500._prepare_examples([{"p": "x"}], cfg_missing_solution, limit=None)


def test_prepare_examples_respects_limit():
    cfg = math500.Math500EvalConfig(prompt_column="p", solution_column="a")
    dataset = [{"p": f"q{i}", "a": f"ans{i}"} for i in range(5)]
    examples = math500._prepare_examples(dataset, cfg, limit=2)
    assert len(examples) == 2
    assert examples[0] == ("q0", "ans0")


def test_resolve_default_device_prefers_mps(monkeypatch):
    class _MPS:
        @staticmethod
        def is_available():
            return True

    class _Cuda:
        @staticmethod
        def is_available():
            return False

    class _Torch:
        cuda = _Cuda()

        class backends:
            mps = _MPS()

        @staticmethod
        def device(name):
            return f"dev:{name}"

    monkeypatch.setattr(math500, "torch", _Torch)
    assert math500._resolve_default_device(None) == "dev:mps"


def test_transformers_prompt_runner_sets_chat_template_and_pad(monkeypatch):
    class _Tokenizer:
        def __init__(self):
            self.chat_template = None
            self.pad_token_id = None
            self.eos_token_id = [7]
            self.eos_token = "<eos>"
            self.pad_token = None
            self.calls = {}

        def add_special_tokens(self, mapping):
            self.pad_token = mapping.get("pad_token")
            self.pad_token_id = 0

        def __call__(self, prompts, **_kwargs):
            self.calls["prompts"] = prompts
            return {"attention_mask": _Mask([[1, 1, 1]])}

        def decode(self, ids, **_kwargs):
            return " ".join(str(tok) for tok in ids)

    class _Mask:
        def __init__(self, arr):
            self.arr = arr

        def sum(self, dim=1):
            return [len(row) for row in self.arr]

    class _Model:
        def __init__(self):
            self.moved = None
            self.evaluated = False
            self.resized = False

        def to(self, device):
            self.moved = device
            return self

        def eval(self):
            self.evaluated = True

        def resize_token_embeddings(self, *_args, **_kwargs):
            self.resized = True

        def generate(self, **_kwargs):
            return [[1, 2, 3, 4]]

    class _Torch:
        class cuda:
            @staticmethod
            def is_available():
                return False

        class backends:
            class mps:
                @staticmethod
                def is_available():
                    return False

        dtype = type("dtype", (), {})

        @staticmethod
        def device(name):
            return f"dev:{name}"

        @staticmethod
        def no_grad():
            class _Ctx:
                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, exc_type, exc, tb):
                    return False

            return _Ctx()

    tok = _Tokenizer()
    model = _Model()
    monkeypatch.setattr(math500, "torch", _Torch)
    monkeypatch.setattr(math500, "Tensor", type("Tensor", (), {}))
    monkeypatch.setattr(
        math500, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *_a, **_k: tok)
    )
    monkeypatch.setattr(
        math500,
        "AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=lambda *_a, **_k: model),
    )

    spec = math500.InferenceModelSpec(
        model_name_or_path="m",
        chat_template="tmpl",
        torch_dtype="auto",
    )
    runner = math500.TransformersPromptRunner(spec)
    assert tok.chat_template == "tmpl"
    assert tok.pad_token_id == 7
    assert runner.device == "dev:cpu"


def test_resolve_default_device_handles_missing_torch(monkeypatch):
    monkeypatch.setattr(math500, "torch", None)
    with pytest.raises(ImportError):
        math500._resolve_default_device(None)


def test_resolve_default_device_prefers_override(monkeypatch):
    class _Cuda:
        @staticmethod
        def is_available():
            return True

    class _Mps:
        @staticmethod
        def is_available():
            return False

    class _Backends:
        mps = _Mps()

    class _Torch:
        cuda = _Cuda()
        backends = _Backends()

        @staticmethod
        def device(name):
            return f"dev:{name}"

        @staticmethod
        def dtype():
            return None

    monkeypatch.setattr(math500, "torch", _Torch)
    assert math500._resolve_default_device("cpu") == "dev:cpu"
    assert math500._resolve_default_device(None) == "dev:cuda"


def test_normalize_dtype_converts_strings(monkeypatch):
    class _Torch:
        float16 = object()
        dtype = type("dtype", (), {})

    monkeypatch.setattr(math500, "torch", _Torch)
    assert math500._normalize_dtype("float16") is _Torch.float16
    assert math500._normalize_dtype(" auto ") == "auto"


def test_transformers_prompt_runner_import_guards(monkeypatch):
    monkeypatch.setattr(math500, "torch", None)
    with pytest.raises(ImportError):
        math500.TransformersPromptRunner(
            math500.InferenceModelSpec(model_name_or_path="x")
        )
    monkeypatch.setattr(math500, "torch", types.SimpleNamespace())
    monkeypatch.setattr(math500, "AutoTokenizer", None)
    monkeypatch.setattr(math500, "AutoModelForCausalLM", None)
    with pytest.raises(ImportError):
        math500.TransformersPromptRunner(
            math500.InferenceModelSpec(model_name_or_path="x")
        )


class _StubTensor:
    def __init__(self, data):
        self.data = data

    def to(self, *_args, **_kwargs):
        return self

    def sum(self, dim=None):
        if dim is None:
            return sum(self.data)
        return [sum(row) for row in self.data]

    def __iter__(self):
        return iter(self.data)


class _StubTorch:
    float32 = object()
    float16 = object()
    dtype = type("dtype", (), {})

    class cuda:
        @staticmethod
        def is_available():
            return False

    class backends:
        class mps:
            @staticmethod
            def is_available():
                return False

    @staticmethod
    def device(name):
        return f"dev:{name}"

    @staticmethod
    def no_grad():
        class _Ctx:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, exc_type, exc, tb):
                return False

        return _Ctx()


def _stub_tokenizer():
    class _Tokenizer:
        def __init__(self):
            self.pad_token_id = None
            self.eos_token_id = [99]
            self.eos_token = "<eos>"
            self.pad_token = None
            self.chat_template = None

        def add_special_tokens(self, mapping):
            self.pad_token_id = 0
            return mapping

        def resize_token_embeddings(self, *_args, **_kwargs):
            return None

        def __call__(self, prompts, **_kwargs):
            batch = len(prompts)
            return {
                "input_ids": [[1, 2, 3]] * batch,
                "attention_mask": _StubTensor([[1, 1, 1]] * batch),
            }

        def decode(
            self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        ):
            return " ".join(str(tok) for tok in ids)

    return _Tokenizer()


def _stub_model():
    class _Model:
        def __init__(self):
            self.moved = None
            self.evaluated = False
            self.resize_called = False

        def to(self, device):
            self.moved = device
            return self

        def eval(self):
            self.evaluated = True

        def resize_token_embeddings(self, *_args, **_kwargs):
            self.resize_called = True

        def generate(self, **_kwargs):
            batch = len(_kwargs.get("input_ids", []))
            return [[1, 2, 3, 42 + i] for i in range(batch)]

    return _Model()


def test_transformers_prompt_runner_sets_padding_and_generates(monkeypatch):
    monkeypatch.setattr(math500, "torch", _StubTorch)
    monkeypatch.setattr(math500, "Tensor", _StubTensor)
    tokenizer = _stub_tokenizer()
    model = _stub_model()
    monkeypatch.setattr(
        math500,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *_a, **_k: tokenizer),
    )
    monkeypatch.setattr(
        math500,
        "AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=lambda *_a, **_k: model),
    )
    spec = math500.InferenceModelSpec(
        model_name_or_path="demo/model",
        system_prompt="system",
        temperature=0.1,
        chat_template=None,
    )
    runner = math500.TransformersPromptRunner(spec)
    prompt = runner._build_prompt("prob")
    assert "SYSTEM" in prompt and "prob" in prompt
    outputs = runner.generate(["p1", "p2"])
    assert outputs == ["42", "43"]
    assert runner.tokenizer.pad_token_id == 99
    assert runner.tokenizer.pad_token == "<eos>"
    runner.close()


def test_transformers_prompt_runner_infers_missing_eos_and_padding(monkeypatch):
    monkeypatch.setattr(math500, "torch", _StubTorch)
    monkeypatch.setattr(math500, "Tensor", _StubTensor)

    class _TokenizerMissingEOS:
        def __init__(self):
            self.pad_token_id = None
            self.eos_token_id = None
            self.eos_token = "<eos>"
            self.pad_token = None
            self.chat_template = None
            self.special_added = False

        def add_special_tokens(self, mapping):
            self.special_added = True
            self.pad_token_id = 7
            return mapping

        def resize_token_embeddings(self, *_args, **_kwargs):
            self.resize_called = True

        def __len__(self):
            return 5

        def __call__(self, prompts, **_kwargs):
            batch = len(prompts)
            return {
                "input_ids": [[1]] * batch,
                "attention_mask": _StubTensor([[1]] * batch),
            }

        def decode(
            self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        ):
            return " ".join(str(tok) for tok in ids)

    tokenizer = _TokenizerMissingEOS()
    model = _stub_model()
    monkeypatch.setattr(
        math500,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *_a, **_k: tokenizer),
    )
    monkeypatch.setattr(
        math500,
        "AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=lambda *_a, **_k: model),
    )
    spec = math500.InferenceModelSpec(
        model_name_or_path="demo/model", chat_template=None
    )
    runner = math500.TransformersPromptRunner(spec)
    assert tokenizer.special_added is True
    assert tokenizer.pad_token_id == 7
    assert getattr(model, "resize_called", False) is True
    assert runner.tokenizer.pad_token == "<eos>"
    runner.close()


def test_run_math500_inference_collects_and_validates(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        math500,
        "pure_accuracy_reward_math",
        lambda gens, answers: [1.0 if g == a else 0.0 for g, a in zip(gens, answers)],
    )

    class _Runner(math500.PromptRunner):
        def __init__(self, spec):
            calls.setdefault("specs", []).append(spec.model_name_or_path)

        def generate(self, problems):
            calls.setdefault("prompts", []).extend(problems)
            return [f"{p}-ans" for p in problems]

        def close(self):
            calls["closed"] = True

    specs = [
        math500.InferenceModelSpec(model_name_or_path="m1", batch_size=2),
        math500.InferenceModelSpec(model_name_or_path="m2", batch_size=1),
    ]
    dataset = [{"problem": f"q{i}", "answer": f"q{i}-ans"} for i in range(3)]
    results = math500.run_math500_inference(
        specs,
        math500.Math500EvalConfig(),
        dataset=dataset,
        collect_generations=True,
        runner_factory=lambda spec: _Runner(spec),
    )
    assert len(results) == 2
    assert results[0].correct == 3 and results[0].accuracy == 1.0
    assert results[0].generations == ["q0-ans", "q1-ans", "q2-ans"]
    assert calls["closed"] is True


def test_run_math500_inference_raises_on_length_mismatch(monkeypatch):
    monkeypatch.setattr(
        math500,
        "pure_accuracy_reward_math",
        lambda gens, answers: [0.0 for _ in answers],
    )

    class _BadRunner(math500.PromptRunner):
        def generate(self, problems):
            return []

        def close(self):
            pass

    with pytest.raises(RuntimeError):
        math500.run_math500_inference(
            [math500.InferenceModelSpec(model_name_or_path="m")],
            math500.Math500EvalConfig(),
            dataset=[{"problem": "p", "answer": "a"}],
            runner_factory=lambda spec: _BadRunner(),
        )
