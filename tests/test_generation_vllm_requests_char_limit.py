"""Extra coverage for vLLM request prompt length handling."""

from __future__ import annotations

import builtins
import sys

from maxent_grpo.generation import vllm_requests
from tests.helpers.vllm import make_vllm_context


class _Req(vllm_requests.VLLMRequestMixin):
    def __init__(self):
        self.ctx = make_vllm_context(prompt_char_limit=None, max_prompt_len=None)
        self._safe_generate = None
        self._time = None
        self._fallback_generate = None


def test_prompt_char_limit_import_failure(monkeypatch):
    req = _Req()
    # Ensure import raises to hit the fallback branch.
    real_import = builtins.__import__

    def _failing_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "maxent_grpo.generation.vllm":
            raise ImportError("boom")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "maxent_grpo.generation.vllm", raising=False)
    monkeypatch.setattr(builtins, "__import__", _failing_import)
    assert req._prompt_char_limit() == vllm_requests._DEFAULT_PROMPT_CHAR_LIMIT


def test_prompt_char_limit_prefers_override(monkeypatch):
    req = _Req()
    req.ctx.prompt_char_limit = 5
    assert req._prompt_char_limit() == 5
    req.ctx.prompt_char_limit = -1
    req.ctx.max_prompt_len = 3  # approx_chars=12
    assert req._prompt_char_limit() == max(vllm_requests._DEFAULT_PROMPT_CHAR_LIMIT, 12)
