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


def test_to_prompt_seed_no_template_ignores_chat_template():
    class _Tok:
        def apply_chat_template(self, *args, **kwargs):
            raise AssertionError("chat template should not be used")

    example = {"prompt": "hi", "answer": "42"}
    out = prompts._to_prompt(
        example,
        _Tok(),
        "prompt",
        system_prompt="SYS",
        char_limit=50,
        prompt_template="no",
    )
    assert out["prompt"] == "hi"
    assert out["answer"] == "42"


def test_to_prompt_seed_qwen_math_template_matches_official_string():
    class _Tok:
        def apply_chat_template(self, *args, **kwargs):
            raise AssertionError("chat template should not be used")

    example = {"prompt": "Solve x+1=2", "answer": "1"}
    out = prompts._to_prompt(
        example,
        _Tok(),
        "prompt",
        system_prompt=None,
        char_limit=500,
        prompt_template="qwen_math",
    )
    assert out["prompt"] == (
        "<|im_start|>system\nPlease reason step by step, and put your final answer "
        "within \\boxed{}.<|im_end|>\n<|im_start|>user\nSolve x+1=2<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_to_prompt_seed_r1_template_matches_official_string():
    class _Tok:
        def apply_chat_template(self, *args, **kwargs):
            raise AssertionError("chat template should not be used")

    example = {"prompt": "Solve x+1=2", "answer": "1"}
    out = prompts._to_prompt(
        example,
        _Tok(),
        "prompt",
        system_prompt=None,
        char_limit=500,
        prompt_template="r1",
    )
    assert out["prompt"] == (
        "A conversation between User and Assistant. The User asks a question, and "
        "the Assistant solves it. The Assistant first thinks about the reasoning "
        "process in the mind and then provides the User with the answer. The "
        "reasoning process is enclosed within <think> </think> and answer is "
        "enclosed within <answer> </answer> tags, respectively, i.e., <think> "
        "reasoning process here </think> <answer> answer here </answer>.\nUser: "
        "Solve x+1=2\nAssistant: <think>"
    )


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
