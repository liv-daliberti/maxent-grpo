"""Unit tests for prompt helpers in training.runtime.prompts."""

from __future__ import annotations

import logging


from maxent_grpo.training.runtime import prompts


def test_truncate_prompt_warns_once(caplog, monkeypatch):
    caplog.set_level(logging.WARNING)
    monkeypatch.setenv("MAX_PROMPT_CHARS", "5")
    # Reload to pick up env override
    from importlib import reload

    mod = reload(prompts)
    long_prompt = "123456789"
    first = mod.truncate_prompt(long_prompt)
    assert first == "12345"
    second = mod.truncate_prompt(long_prompt)
    assert second == "12345"
    # Only one warning emitted
    warns = [rec for rec in caplog.records if rec.levelno == logging.WARNING]
    assert len(warns) == 1


def test_prompt_char_limit_from_tokens_respects_floor(monkeypatch):
    monkeypatch.setenv("MAX_PROMPT_CHARS", "100")
    from importlib import reload

    mod = reload(prompts)
    assert mod._prompt_char_limit_from_tokens(0) == 100
    assert mod._prompt_char_limit_from_tokens(10) >= 100


def test_to_prompt_chat_template(monkeypatch):
    class _Tok:
        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=True
        ):
            return f"TEMPLATE:{messages[-1]['content']}"

    example = {"prompt": "hi", "answer": "42"}
    out = prompts._to_prompt(
        example, _Tok(), "prompt", system_prompt="SYS", char_limit=10
    )
    assert out["prompt"].startswith("TEMPLATE:")
    assert out["answer"] == "42"


def test_to_prompt_fallback_text(monkeypatch):
    class _Tok:
        def apply_chat_template(self, *args, **kwargs):
            raise AttributeError

    example = {"instruction": "hi"}
    out = prompts._to_prompt(
        example, _Tok(), "instruction", system_prompt=None, char_limit=50
    )
    assert out["prompt"].startswith("USER:")
    # Truncated prompt should still include assistant prefix
    assert out["prompt"].endswith("ASSISTANT:")
