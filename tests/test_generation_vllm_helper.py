"""
Focused unit tests for vLLMGenerationHelper covering rarely hit branches.
"""

from types import SimpleNamespace
from typing import List, Optional

import maxent_grpo.generation.vllm as vllm
from maxent_grpo.patches.vllm import safe_generate
import builtins
import sys
import time


class _StubParam:
    def __init__(self, name: str):
        self.data = f"data:{name}"


class _StubAccel:
    def __init__(self, world: int = 1, rank: int = 0):
        self.num_processes = world
        self.process_index = rank
        self.is_main_process = rank == 0


def _make_helper(
    *, world_size: int = 1, rank: int = 0, prompt_char_limit: Optional[int] = None
) -> vllm.VLLMGenerationHelper:
    ctx = SimpleNamespace(
        accelerator=_StubAccel(world_size, rank),
        generation_stats={},
        vllm_request_logprobs=False,
        vllm_sync_weights=False,
        vllm_url="http://localhost:8000",
        prompt_char_limit=prompt_char_limit,
        max_prompt_len=None,
    )
    return vllm.VLLMGenerationHelper(
        ctx, fallback_generate=lambda *_a, **_k: ([], None)
    )


def test_sync_model_params_handles_fsdp_like_wrappers(monkeypatch):
    """fsdp-like modules with wrapped names should be cleaned and deduped."""
    helper = _make_helper()
    helper._fsdp_cls = None  # force local fsdp detection path

    pushed: List[str] = []
    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, _p: pushed.append(name)
    )
    monkeypatch.setattr(helper, "_reset_vllm_cache", lambda: pushed.append("reset"))

    class _Child:
        def named_parameters(self):
            return [
                ("_fsdp_wrapped_module.weight", _StubParam("cw")),
                ("b", _StubParam("cb")),
            ]

    class _Model:
        def __init__(self):
            self.called = False

        def summon_full_params(self):
            return None

        def named_children(self):
            return [("child", _Child())]

        def named_parameters(self):
            return [
                ("a", _StubParam("a1")),
                ("_checkpoint_wrapped_module.a", _StubParam("a2")),
            ]

    helper._sync_model_params_to_vllm(_Model())
    # Deduped and cleaned names (parent params are skipped when children exist).
    assert pushed[:-1] == ["child.weight", "child.b"]
    assert pushed[-1] == "reset"


def test_sync_standard_params_skips_without_named_parameters(monkeypatch):
    helper = _make_helper()
    helper._gather_factory = lambda _p: vllm.nullcontext()
    called = []
    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, _p: called.append(name)
    )

    class _Model:
        def parameters(self):
            return []

        named_parameters = None

    helper._sync_standard_params(_Model())
    assert called == []


def test_init_falls_back_when_vllm_import_fails(monkeypatch):
    """Ensure helper uses local fallbacks when vllm import raises."""
    existing = sys.modules.get("maxent_grpo.generation.vllm")
    monkeypatch.delitem(sys.modules, "maxent_grpo.generation.vllm", raising=False)

    real_import = builtins.__import__

    def _failing_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "maxent_grpo.generation.vllm":
            raise ImportError("boom")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _failing_import)
    helper = _make_helper()
    assert helper._safe_generate is safe_generate
    assert helper._time is time
    if existing is not None:
        monkeypatch.setitem(sys.modules, "maxent_grpo.generation.vllm", existing)


def test_generate_respects_explicit_sync_hooks(monkeypatch):
    helper = _make_helper()
    sync_calls = []
    maybe_sync_calls = []

    monkeypatch.setattr(
        helper, "maybe_sync_weights", lambda *a, **k: maybe_sync_calls.append(a)
    )
    monkeypatch.setattr(
        helper, "_prepare_vllm_targets", lambda *a, **k: (["p"], [1], None)
    )
    monkeypatch.setattr(helper, "_resolve_vllm_round_limit", lambda *_a, **_k: 1)

    def _run(state):
        state.aggregated = [["x"]]
        state.aggregated_meta = None

    monkeypatch.setattr(helper, "_run_vllm_rounds", _run)

    grouped, meta = helper.generate(
        ["p"],
        num_samples=1,
        per_prompt_counts=None,
        ensure_client=lambda: sync_calls.append("ensure"),
        sync_model=lambda _m: sync_calls.append("sync"),
    )

    # maybe_sync_weights should receive the explicit hooks; we do not execute them here.
    assert len(maybe_sync_calls) == 1
    ensure_arg, sync_arg = maybe_sync_calls[0]
    assert callable(ensure_arg) and callable(sync_arg)
    assert (
        sync_calls == []
    )  # hooks not invoked by the helper; passed through for caller
    assert grouped == [["x"]] and meta is None


def test_generate_recovers_from_typeerror_in_sync(monkeypatch):
    helper = _make_helper()
    sync_calls = []

    def _sync_fn(*args, **_kwargs):
        if args:
            raise TypeError("bad args")
        sync_calls.append("noargs")

    helper.maybe_sync_weights = _sync_fn
    monkeypatch.setattr(
        helper, "_prepare_vllm_targets", lambda *a, **k: (["p"], [1], None)
    )
    monkeypatch.setattr(helper, "_resolve_vllm_round_limit", lambda *_a, **_k: 1)

    def _run(state):
        state.aggregated = [["ok"]]
        state.aggregated_meta = None

    monkeypatch.setattr(helper, "_run_vllm_rounds", _run)
    grouped, meta = helper.generate(
        ["p"],
        num_samples=1,
        per_prompt_counts=None,
        ensure_client=lambda: None,
        sync_model=lambda _m: None,
    )
    assert sync_calls == ["noargs"]
    assert grouped == [["ok"]] and meta is None


def test_generate_collective_with_scatter_none(monkeypatch):
    """Scatter returning None should yield empty results."""
    helper = _make_helper(world_size=2, rank=1)
    # prevent normal generate call; we only care about scatter
    monkeypatch.setattr(helper, "_scatter_object", lambda *_a, **_k: (None, None))
    grouped, meta = helper._scatter_vllm_payload(
        flat_prompts=["a", "b"],
        offsets=[0, 1, 2],
        grouped_all=[["g0"], ["g1"]],
        meta_all=None,
    )
    assert grouped == [] and meta is None


def test_prompt_char_limit_prefers_max_of_base_and_estimate():
    helper = _make_helper(prompt_char_limit=None)
    helper.ctx.max_prompt_len = 10  # approx_chars=40
    # PROMPT_CHAR_LIMIT from run_helpers is > 40, so max should be PROMPT_CHAR_LIMIT
    assert helper._prompt_char_limit() == vllm.PROMPT_CHAR_LIMIT


def test_prompt_char_limit_falls_back_on_import_failure(monkeypatch):
    helper = _make_helper(prompt_char_limit=None)
    helper.ctx.max_prompt_len = None
    existing = sys.modules.get("maxent_grpo.generation.vllm")
    monkeypatch.delitem(sys.modules, "maxent_grpo.generation.vllm", raising=False)

    real_import = builtins.__import__

    def _failing_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "maxent_grpo.generation.vllm":
            raise ImportError("boom")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _failing_import)
    from maxent_grpo.generation import vllm_requests

    assert helper._prompt_char_limit() == vllm_requests._DEFAULT_PROMPT_CHAR_LIMIT
    if existing is not None:
        monkeypatch.setitem(sys.modules, "maxent_grpo.generation.vllm", existing)


def test_prompt_char_limit_uses_default_when_import_raises(monkeypatch):
    helper = _make_helper(prompt_char_limit=None)
    helper.ctx.max_prompt_len = 5  # approx_chars=20
    real_import = builtins.__import__

    def _raising_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "maxent_grpo.generation.vllm":
            raise RuntimeError("bad import")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _raising_import)
    from maxent_grpo.generation import vllm_requests

    # Even with approx_chars present, the default constant wins when import fails.
    assert helper._prompt_char_limit() == vllm_requests._DEFAULT_PROMPT_CHAR_LIMIT
