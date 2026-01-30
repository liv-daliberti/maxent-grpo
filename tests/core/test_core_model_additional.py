"""
Additional unit tests for core.model helpers.
"""

from __future__ import annotations

from types import SimpleNamespace


from maxent_grpo.core import model as core_model


class _ModelConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _TrainingArgs(SimpleNamespace):
    pass


def test_get_tokenizer_falls_back_and_sets_chat_template(monkeypatch):
    """AutoTokenizer failure should fall back to the stub and apply chat_template."""

    monkeypatch.setattr(
        core_model,
        "AutoTokenizer",
        SimpleNamespace(
            from_pretrained=lambda *_a, **_k: (_ for _ in ()).throw(OSError())
        ),
    )
    training_args = _TrainingArgs(chat_template="chat-template")
    tok = core_model.get_tokenizer(
        _ModelConfig(
            model_name_or_path="stub", model_revision=None, trust_remote_code=False
        ),
        training_args,
    )
    assert isinstance(tok, core_model.PreTrainedTokenizerStub)
    assert getattr(tok, "chat_template") == "chat-template"


def test_get_model_resolves_dtype_and_enables_gradient_checkpointing(monkeypatch):
    """Ensure dtype conversion and gradient checkpointing kwargs are applied."""

    captured = {}

    class _DummyModel:
        def __init__(self):
            self.gc_calls = []

        def gradient_checkpointing_enable(self, **kwargs):
            self.gc_calls.append(kwargs)

    def _from_pretrained(name, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
        return _DummyModel()

    monkeypatch.setattr(core_model, "get_quantization_config", lambda *_a, **_k: None)
    monkeypatch.setattr(
        core_model, "get_kbit_device_map", lambda *_a, **_k: {"0": "cpu"}
    )
    monkeypatch.setattr(
        core_model,
        "AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=_from_pretrained),
    )

    model = core_model.get_model(
        _ModelConfig(
            model_name_or_path="stub/model",
            model_revision="rev",
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype="float16",
        ),
        _TrainingArgs(
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        ),
    )

    assert captured["name"] == "stub/model"
    kwargs = captured["kwargs"]
    assert kwargs["revision"] == "rev"
    assert kwargs["trust_remote_code"] is True
    assert kwargs["attn_implementation"] == "sdpa"
    assert kwargs["use_cache"] is False  # disabled when gradient checkpointing on
    # dtype should resolve from string to torch attribute when available
    assert str(kwargs["torch_dtype"]) in ("torch.float16", "float16")
    assert isinstance(model, _DummyModel)
    assert model.gc_calls == [{"use_reentrant": False}]


def test_get_model_gradient_checkpointing_fallback_on_type_error(monkeypatch):
    """If gradient_checkpointing_enable raises TypeError with kwargs, retry without."""

    captured = {}

    class _ModelWithGC:
        def __init__(self):
            self.calls = []
            self._called_once = False

        def gradient_checkpointing_enable(self, **kwargs):
            self.calls.append(kwargs)
            if not self._called_once:
                self._called_once = True
                raise TypeError("unsupported kwargs")

    def _from_pretrained(name, **kwargs):
        captured["kwargs"] = kwargs
        return _ModelWithGC()

    monkeypatch.setattr(core_model, "get_quantization_config", lambda *_a, **_k: None)
    monkeypatch.setattr(core_model, "get_kbit_device_map", lambda *_a, **_k: None)
    monkeypatch.setattr(
        core_model,
        "AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=_from_pretrained),
    )

    model = core_model.get_model(
        _ModelConfig(
            model_name_or_path="stub",
            model_revision=None,
            trust_remote_code=False,
            attn_implementation=None,
            torch_dtype=None,
        ),
        _TrainingArgs(
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"foo": "bar"}
        ),
    )

    assert isinstance(model, _ModelWithGC)
    # First call attempted with kwargs; second call without kwargs after TypeError
    assert model.calls[0] == {"foo": "bar"}
    assert model.calls[1] == {}


def test_get_model_applies_torch_compile_when_enabled(monkeypatch):
    """When torch_compile is true and torch.compile is available, wrap the model."""

    captured = {}

    class _BaseModel:
        pass

    def _from_pretrained(name, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
        return _BaseModel()

    monkeypatch.setattr(
        core_model,
        "AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=_from_pretrained),
    )
    monkeypatch.setattr(core_model, "get_quantization_config", lambda *_a, **_k: None)
    monkeypatch.setattr(core_model, "get_kbit_device_map", lambda *_a, **_k: None)

    compile_calls = {}

    def _compile(model, **kwargs):
        compile_calls["model"] = model
        compile_calls["kwargs"] = kwargs
        return SimpleNamespace(compiled_from=model, compile_kwargs=kwargs)

    monkeypatch.setattr(core_model.torch, "compile", _compile, raising=False)

    model_args = _ModelConfig(
        model_name_or_path="stub-compile",
        model_revision=None,
        trust_remote_code=False,
        attn_implementation=None,
        torch_dtype=None,
    )
    training_args = _TrainingArgs(
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs=None,
        torch_compile=True,
    )

    compiled_model = core_model.get_model(model_args, training_args)
    assert isinstance(compiled_model, SimpleNamespace)
    assert isinstance(compiled_model.compiled_from, _BaseModel)
    # Ensure compile was invoked with the model and a mode kwarg when supported.
    assert compile_calls["model"].__class__ is _BaseModel
    assert "mode" in compile_calls["kwargs"]
