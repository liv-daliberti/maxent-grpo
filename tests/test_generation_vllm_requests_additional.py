"""Additional coverage for vllm_requests prompt length fallback."""

from __future__ import annotations

import builtins
from types import SimpleNamespace

import maxent_grpo.generation.vllm_requests as vr


class _ReqHelper(vr.VLLMRequestMixin):
    def __init__(self):
        self.ctx = SimpleNamespace(
            prompt_char_limit=None,
            max_prompt_len=None,
        )


def test_prompt_char_limit_import_error(monkeypatch):
    helper = _ReqHelper()
    orig_import = builtins.__import__

    def _boom(name, *args, **kwargs):
        if name == "maxent_grpo.generation.vllm":
            raise ImportError("fail")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _boom)
    assert helper._prompt_char_limit() == vr._DEFAULT_PROMPT_CHAR_LIMIT
