"""Unit coverage for vLLM request helpers."""

from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace, ModuleType

from maxent_grpo.generation.vllm_requests import (
    VLLMRequestMixin,
    _DEFAULT_PROMPT_CHAR_LIMIT,
)
from maxent_grpo.generation.vllm_state import _VLLMGenerationState


class _Dummy(VLLMRequestMixin):
    def __init__(self, ctx):
        self.ctx = ctx
        self._safe_generate = None
        self._time = None
        self._fallback_generate = None


def test_prompt_char_limit_falls_back_on_import_error(monkeypatch):
    # Ensure import of maxent_grpo.generation.vllm fails to hit the ImportError branch.
    monkeypatch.delitem(sys.modules, "maxent_grpo.generation.vllm", raising=False)
    ctx = SimpleNamespace(prompt_char_limit=None, max_prompt_len=0)
    dummy = _Dummy(ctx)
    assert dummy._prompt_char_limit() == _DEFAULT_PROMPT_CHAR_LIMIT


def test_prompt_char_limit_falls_back_on_missing_attr(monkeypatch):
    # Provide a vllm module without PROMPT_CHAR_LIMIT to trigger AttributeError path.
    vllm_mod = ModuleType("maxent_grpo.generation.vllm")
    monkeypatch.setitem(sys.modules, "maxent_grpo.generation.vllm", vllm_mod)
    ctx = SimpleNamespace(prompt_char_limit=None, max_prompt_len=1)
    dummy = _Dummy(ctx)
    # approx_chars will be 4; fallback should choose max of default and approx.
    assert dummy._prompt_char_limit() == max(_DEFAULT_PROMPT_CHAR_LIMIT, 4)


def test_prompt_char_limit_prefers_imported_constant(monkeypatch):
    """When PROMPT_CHAR_LIMIT is importable, it should be returned."""

    monkeypatch.delitem(sys.modules, "maxent_grpo.generation.vllm", raising=False)
    vllm_mod = ModuleType("maxent_grpo.generation.vllm")
    vllm_mod.PROMPT_CHAR_LIMIT = 17
    monkeypatch.setitem(sys.modules, "maxent_grpo.generation.vllm", vllm_mod)
    pkg = sys.modules.get("maxent_grpo.generation")
    if pkg is not None:
        monkeypatch.setattr(pkg, "vllm", vllm_mod, raising=False)
    ctx = SimpleNamespace(prompt_char_limit=None, max_prompt_len=0)
    dummy = _Dummy(ctx)
    assert dummy._prompt_char_limit() == 17


def test_prompt_char_limit_handles_attribute_error(monkeypatch):
    """AttributeError during import should trigger the fallback constant."""

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "maxent_grpo.generation.vllm":
            raise AttributeError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    monkeypatch.delitem(sys.modules, "maxent_grpo.generation.vllm", raising=False)
    ctx = SimpleNamespace(prompt_char_limit=None, max_prompt_len=0)
    dummy = _Dummy(ctx)
    assert dummy._prompt_char_limit() == _DEFAULT_PROMPT_CHAR_LIMIT


def test_prompt_char_limit_uses_approx_when_default_disabled(monkeypatch):
    """If default limit is disabled and import fails, fall back to approx_chars."""

    real_import = __import__
    import_calls = []

    def fake_import(name, *args, **kwargs):
        if name.startswith("maxent_grpo.generation"):
            import_calls.append(name)
            raise ImportError("no vllm")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    monkeypatch.setattr(
        "maxent_grpo.generation.vllm_requests._DEFAULT_PROMPT_CHAR_LIMIT", -1
    )
    monkeypatch.delitem(sys.modules, "maxent_grpo.generation.vllm", raising=False)
    pkg = sys.modules.get("maxent_grpo.generation")
    if pkg is not None and hasattr(pkg, "vllm"):
        monkeypatch.delattr(pkg, "vllm", raising=False)
    ctx = SimpleNamespace(prompt_char_limit=None, max_prompt_len=2)
    dummy = _Dummy(ctx)
    result = dummy._prompt_char_limit()
    assert import_calls  # verify import path was exercised
    assert result == 8


def test_prepare_vllm_targets_with_dedup(monkeypatch):
    monkeypatch.setenv("MAXENT_VLLM_DEDUP", "1")
    ctx = SimpleNamespace()
    dummy = _Dummy(ctx)
    prompts, counts, mapping = dummy._prepare_vllm_targets(
        ["a", "b", "a"], num_samples=2, per_prompt_counts=[1, 2, 3]
    )
    assert prompts == ["a", "b"]
    assert counts == [1, 2]
    assert mapping == [0, 1, 0]


def test_prompt_char_limit_prefers_override():
    ctx = SimpleNamespace(prompt_char_limit=42, max_prompt_len=0)
    dummy = _Dummy(ctx)
    assert dummy._prompt_char_limit() == 42


def test_merge_vllm_results_records_overflow():
    ctx = SimpleNamespace(generation_stats={})
    dummy = _Dummy(ctx)
    state = _VLLMGenerationState(
        prompts=["p"],
        target_counts=[1],
        requested_n=1,
        round_limit=1,
        track_logprobs=False,
    )
    dummy._merge_vllm_results(
        state, grouped=[["x", "y"]], grouped_meta=None, pending_indices=[0]
    )
    assert ctx.generation_stats["vllm_excess_prompts"] == 1
    assert ctx.generation_stats["vllm_excess_completions"] == 1


def test_backfill_missing_skips_when_no_need(monkeypatch):
    ctx = SimpleNamespace(
        vllm_backfill_local=True,
        generation_stats={"vllm_backfilled_prompts": 0},
    )
    dummy = _Dummy(ctx)
    dummy._fallback_generate = lambda prompts, n, counts: ([["fill"]], None)
    state = _VLLMGenerationState(
        prompts=["p"],
        target_counts=[1],
        requested_n=1,
        round_limit=1,
        track_logprobs=False,
    )
    state.aggregated[0].append("existing")
    dummy._backfill_missing(state, [0])
    assert state.aggregated[0] == ["existing"]
    assert ctx.generation_stats["vllm_backfilled_prompts"] == 1


def test_record_vllm_failure_updates_stats(monkeypatch):
    ctx = SimpleNamespace(
        vllm_backfill_local=True,
        generation_stats={"vllm_failed_prompts": 0},
    )
    dummy = _Dummy(ctx)
    state = _VLLMGenerationState(
        prompts=["p1", "p2"],
        target_counts=[2, 1],
        requested_n=1,
        round_limit=2,
        track_logprobs=False,
    )
    state.aggregated[0].append("done")
    dummy._record_vllm_failure(state, [0, 1])
    assert ctx.generation_stats["vllm_failed_prompts"] == 2
