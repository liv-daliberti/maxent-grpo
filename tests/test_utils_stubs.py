"""
Coverage for utils.stubs helpers.
"""

from __future__ import annotations

from types import SimpleNamespace

from maxent_grpo.utils import stubs


def test_assign_module_best_effort():
    obj = SimpleNamespace()
    stubs._assign_module(obj, "demo.module")
    # SimpleNamespace allows setting __module__
    assert getattr(obj, "__module__", None) == "demo.module"

    class _NoModule:
        __slots__ = ()

    inst = _NoModule()
    stubs._assign_module(inst, "cannot.set")  # should not raise
    # __slots__ objects may still get __module__ injected; just ensure no exception
    assert getattr(inst, "__module__", None) in ("cannot.set", _NoModule.__module__)


def test_fallback_tokenizer_chat_template_and_encode():
    tok = stubs.FallbackTokenizer.from_pretrained("ignored")
    assert isinstance(tok, stubs.FallbackTokenizer)
    msg = [{"role": "user", "content": "hi"}]
    rendered = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    assert "assistant:" in rendered.lower()
    encoded = tok.apply_chat_template(msg, tokenize=True, add_generation_prompt=False)
    assert isinstance(encoded, list)
    assert all(isinstance(b, int) for b in encoded)


def test_auto_config_stub_and_model_stub():
    cfg = stubs.AutoConfigStub.from_pretrained("ignored")
    assert isinstance(cfg, stubs.AutoConfigStub)
    model = stubs.AutoModelForCausalLMStub.from_pretrained("ignored")
    assert isinstance(model, stubs.AutoModelForCausalLMStub)
    assert hasattr(model, "config")
    # gradient checkpointing stub returns None and does not raise
    assert model.gradient_checkpointing_enable() is None


def test_stub_aliases_point_to_fallback_tokenizer():
    assert stubs.AutoTokenizerStub is stubs.FallbackTokenizer
    assert stubs.PreTrainedTokenizerStub is stubs.FallbackTokenizer
