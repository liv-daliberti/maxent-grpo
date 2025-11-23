from __future__ import annotations

from types import SimpleNamespace

import importlib
import builtins
import sys


def test_get_tokenizer_fallback_sets_chat_template(monkeypatch):
    module = importlib.reload(importlib.import_module("src.core.model"))

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
    module = importlib.reload(importlib.import_module("src.core.model"))

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


def test_transformers_import_failure_uses_fallback_stubs(monkeypatch):
    monkeypatch.delitem(sys.modules, "src.core.model", raising=False)
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name.startswith("transformers"):
            raise ModuleNotFoundError("transformers missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    module = importlib.import_module("src.core.model")

    tok = module.PreTrainedTokenizer()
    messages = [{"role": "user", "content": "hi"}]
    tokens = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    assert tokens == list("user: hi\nassistant:".encode("utf-8"))
    rendered = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    assert rendered == "user: hi"

    model = module.AutoModelForCausalLM.from_pretrained("demo")
    assert hasattr(model, "config")
    assert model.gradient_checkpointing_enable() is None


def test_get_model_auto_dtype_and_gc_without_kwargs(monkeypatch):
    module = importlib.reload(importlib.import_module("src.core.model"))
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
    module = importlib.reload(importlib.import_module("src.core.model"))
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


def test_trl_import_failure_uses_stubbed_helpers(monkeypatch):
    monkeypatch.delitem(sys.modules, "src.core.model", raising=False)
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name.startswith("trl"):
            raise ModuleNotFoundError("trl missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    module = importlib.import_module("src.core.model")

    assert module.get_quantization_config(SimpleNamespace()) is None
    assert module.get_kbit_device_map() is None
