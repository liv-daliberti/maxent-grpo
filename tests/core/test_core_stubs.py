"""Unit tests for lightweight core stubs used when transformers is absent."""

from __future__ import annotations

from maxent_grpo.core import stubs


def test_tokenizer_stub_apply_and_decode():
    tokenizer = stubs.PreTrainedTokenizerStub.from_pretrained("ignored")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=True,
        add_generation_prompt=True,
    )
    assert isinstance(rendered, list) and rendered
    decoded = tokenizer.decode([104, 105])
    assert decoded == "hi"
    # skip_special_tokens=False still decodes and exercises the no-op branch.
    assert tokenizer.decode([65], skip_special_tokens=False) == "A"
    assert tokenizer.decode([-1]) == "-1"  # triggers fallback path
    # Also exercise non-tokenizing path for coverage.
    plain = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        add_generation_prompt=False,
    )
    assert "USER: hello" in plain


def test_auto_model_stub_surfaces_config_and_gc():
    model = stubs.AutoModelForCausalLMStub.from_pretrained("model")
    assert hasattr(model, "config")
    # Should be callable without error and return None
    assert model.gradient_checkpointing_enable() is None


def test_auto_config_stub_from_pretrained():
    cfg = stubs.AutoConfigStub.from_pretrained("model")
    assert getattr(cfg, "num_attention_heads", None) == 0
