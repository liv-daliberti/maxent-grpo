"""Coverage for VLLMRequestMixin._prompt_char_limit edge cases."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace

import maxent_grpo.generation.vllm_requests as vr


class _PromptLimitHelper(vr.VLLMRequestMixin):
    def __init__(self, ctx):
        self.ctx = ctx


def test_prompt_char_limit_prefers_ctx_override():
    helper = _PromptLimitHelper(
        SimpleNamespace(prompt_char_limit=123, max_prompt_len=10)
    )
    assert helper._prompt_char_limit() == 123


def test_prompt_char_limit_handles_negative_constant(monkeypatch):
    dummy_vllm = ModuleType("maxent_grpo.generation.vllm")
    dummy_vllm.PROMPT_CHAR_LIMIT = -1
    modules = __import__("sys").modules
    monkeypatch.setitem(modules, dummy_vllm.__name__, dummy_vllm)
    monkeypatch.setattr(
        modules["maxent_grpo.generation"], "vllm", dummy_vllm, raising=False
    )
    helper = _PromptLimitHelper(
        SimpleNamespace(prompt_char_limit=None, max_prompt_len=6)
    )

    # Negative constant falls back to approx chars computed from max_len.
    assert helper._prompt_char_limit() == 24


def test_prompt_char_limit_uses_constant_when_no_max_len(monkeypatch):
    dummy_vllm = ModuleType("maxent_grpo.generation.vllm")
    dummy_vllm.PROMPT_CHAR_LIMIT = 42
    modules = __import__("sys").modules
    monkeypatch.setitem(modules, dummy_vllm.__name__, dummy_vllm)
    monkeypatch.setattr(
        modules["maxent_grpo.generation"], "vllm", dummy_vllm, raising=False
    )
    helper = _PromptLimitHelper(
        SimpleNamespace(prompt_char_limit=None, max_prompt_len=None)
    )

    assert helper._prompt_char_limit() == 42
