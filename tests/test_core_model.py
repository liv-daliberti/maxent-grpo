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
"""

from __future__ import annotations

from types import SimpleNamespace

import importlib
import sys


def test_get_tokenizer_fallback_sets_chat_template(monkeypatch):
    module = importlib.reload(importlib.import_module("maxent_grpo.core.model"))

    # Force fallback path by raising when attempting to load a pretrained tokenizer.
    def _raise(*_args, **_kwargs):
        raise OSError("fail")

    monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", _raise)

    model_args = SimpleNamespace(
        model_name_or_path="demo", model_revision=None, trust_remote_code=False
    )
    training_args = SimpleNamespace(chat_template="<<chat>>")
    tok = module.get_tokenizer(model_args, training_args)

    assert tok.chat_template == "<<chat>>"
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tokenize=True,
        add_generation_prompt=True,
    )
    assert isinstance(rendered, list)
    assert all(isinstance(b, int) for b in rendered)


def test_get_model_quantization_and_gradient_checkpointing(monkeypatch):
    module = importlib.reload(importlib.import_module("maxent_grpo.core.model"))

    captured = {}

    class _StubModel:
        def __init__(self):
            self.gc_calls = []

        @classmethod
        def from_pretrained(cls, name, **kwargs):
            captured["kwargs"] = kwargs
            captured["name"] = name
            return cls()

        def gradient_checkpointing_enable(self, **kwargs):
            self.gc_calls.append(kwargs)
            if kwargs:
                raise TypeError("trigger fallback")

    monkeypatch.setattr(module, "AutoModelForCausalLM", _StubModel)
    monkeypatch.setattr(
        module, "get_quantization_config", lambda _args: {"quant": True}
    )
    monkeypatch.setattr(module, "get_kbit_device_map", lambda: {"model": 0})

    model_args = SimpleNamespace(
        model_name_or_path="demo-model",
        model_revision="rev",
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype="float16",
    )
    training_args = SimpleNamespace(
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"foo": "bar"}
    )

    model = module.get_model(model_args, training_args)

    assert captured["name"] == "demo-model"
    kwargs = captured["kwargs"]
    assert kwargs["quantization_config"] == {"quant": True}
    assert kwargs["device_map"] == {"model": 0}
    assert kwargs["torch_dtype"] == getattr(module.torch, "float16", "float16")
    assert kwargs["use_cache"] is False  # disabled when grad checkpointing is on
    assert model.gc_calls == [{"foo": "bar"}, {}]  # fallback retried without kwargs


def test_get_model_auto_dtype_and_gc_without_kwargs(monkeypatch):
    module = importlib.reload(importlib.import_module("maxent_grpo.core.model"))
    captured = {}

    class _StubModel:
        def __init__(self):
            self.gc_calls = []

        @classmethod
        def from_pretrained(cls, name, **kwargs):
            captured["name"] = name
            captured["kwargs"] = kwargs
            return cls()

        def gradient_checkpointing_enable(self):
            self.gc_calls.append({})

    monkeypatch.setattr(module, "AutoModelForCausalLM", _StubModel)
    monkeypatch.setattr(module, "get_quantization_config", lambda _args: None)
    monkeypatch.setattr(
        module, "get_kbit_device_map", lambda: {"should_not_be_used": True}
    )

    model_args = SimpleNamespace(
        model_name_or_path="demo-auto",
        model_revision=None,
        trust_remote_code=False,
        attn_implementation=None,
        torch_dtype="auto",
    )
    training_args = SimpleNamespace(
        gradient_checkpointing=True, gradient_checkpointing_kwargs=None
    )

    model = module.get_model(model_args, training_args)
    kwargs = captured["kwargs"]
    assert captured["name"] == "demo-auto"
    assert kwargs["torch_dtype"] == "auto"
    assert kwargs["device_map"] is None  # not computed when quantization_config is None
    assert kwargs["quantization_config"] is None
    assert kwargs["use_cache"] is False
    assert model.gc_calls == [{}]


def test_get_model_preserves_nonstring_dtype(monkeypatch):
    module = importlib.reload(importlib.import_module("maxent_grpo.core.model"))
    dtype_obj = SimpleNamespace(label="dtype")
    captured = {}

    class _StubModel:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            captured["name"] = name
            captured["kwargs"] = kwargs
            return cls()

        def gradient_checkpointing_enable(self, *_a, **_k):
            raise AssertionError("should not be called")

    monkeypatch.setattr(module, "AutoModelForCausalLM", _StubModel)
    monkeypatch.setattr(module, "get_quantization_config", lambda _args: None)
    monkeypatch.setattr(module, "get_kbit_device_map", lambda: None)

    model_args = SimpleNamespace(
        model_name_or_path="demo-dtype",
        model_revision="v1",
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=dtype_obj,
    )
    training_args = SimpleNamespace(gradient_checkpointing=False)

    module.get_model(model_args, training_args)
    kwargs = captured["kwargs"]
    assert kwargs["torch_dtype"] is dtype_obj
    assert kwargs["use_cache"] is True
    assert kwargs["device_map"] is None


def test_trl_stub_helpers_return_none(monkeypatch):
    module = importlib.reload(importlib.import_module("maxent_grpo.core.model"))
    monkeypatch.setitem(sys.modules, "trl", None)
    module = importlib.reload(importlib.import_module("maxent_grpo.core.model"))
    assert module.get_quantization_config(SimpleNamespace()) is None
    assert module.get_kbit_device_map() is None
