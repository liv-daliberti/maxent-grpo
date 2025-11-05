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
