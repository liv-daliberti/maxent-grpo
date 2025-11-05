import types
from importlib import reload

import grpo as grpo


class DummyTok:
    def __init__(self, template_ok=True):
        self.template_ok = template_ok
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        if not self.template_ok:
            raise RuntimeError("no template")
        return "|".join(m["content"] for m in messages) + ("<GEN>" if add_generation_prompt else "")


def test_to_prompt_with_system_and_template():
    ex = {"problem": "sum 1+1", "answer": "2"}
    tok = DummyTok(template_ok=True)
    out = grpo._to_prompt(ex, tok, prompt_column="problem", system_prompt="SYS")
    assert out["prompt"].startswith("SYS|")
    assert out["answer"] == "2"


def test_to_prompt_fallback_string_format():
    ex = {"problem": "q", "solution": "a"}
    tok = DummyTok(template_ok=False)
    out = grpo._to_prompt(ex, tok, prompt_column="problem", system_prompt=None)
    # Fallback prefixes USER:/ASSISTANT:
    assert "USER: q\nASSISTANT:" in out["prompt"]
    assert out["answer"] == "a"
