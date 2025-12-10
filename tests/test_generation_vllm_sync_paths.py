"""
Additional tests for vLLMGenerationHelper covering sync/wrapper branches.
"""

from types import SimpleNamespace
from typing import List

import maxent_grpo.generation.vllm as vllm
from tests.helpers.vllm import make_vllm_context


class _StubAccel:
    def __init__(self):
        self.is_main_process = True
        self.num_processes = 1
        self.process_index = 0
        self.wait_for_everyone_called = False

    def wait_for_everyone(self):
        self.wait_for_everyone_called = True


def _helper_with_client():
    ctx = make_vllm_context(
        accelerator=_StubAccel(),
        generation_stats={},
        vllm_sync_weights=True,
        vllm_request_logprobs=False,
        vllm_url="http://localhost:8000",
        vllm_backfill_local=False,
        model=SimpleNamespace(),
    )
    helper = vllm.VLLMGenerationHelper(
        ctx, fallback_generate=lambda *_a, **_k: ([], None)
    )
    return helper


def test_sync_model_params_prefers_child_params(monkeypatch):
    helper = _helper_with_client()
    helper._fsdp_cls = type("FSDPStub", (), {})
    pushed: List[str] = []
    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, _p: pushed.append(name)
    )
    monkeypatch.setattr(helper, "_reset_vllm_cache", lambda: pushed.append("reset"))

    class _Child:
        def named_parameters(self):
            return [("_fsdp_wrapped_module.w", object())]

    class _Model(helper._fsdp_cls):  # type: ignore[misc]
        def named_children(self):
            return [("child", _Child())]

        def named_parameters(self):
            return [("top", object())]

    helper._sync_model_params_to_vllm(_Model())
    # At least child params pushed with cleaned names, then reset.
    assert pushed and pushed[-1] == "reset"
    assert any(name == "child.w" for name in pushed)


def test_sync_peft_params_filters_names(monkeypatch):
    helper = _helper_with_client()
    pushed: List[str] = []
    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, _p: pushed.append(name)
    )

    class _Model:
        prefix = "skipme"

        def parameters(self):
            return [object()]

        def merge_adapter(self):
            pushed.append("merge")

        def unmerge_adapter(self):
            pushed.append("unmerge")

        def named_parameters(self):
            return [
                ("base_model.model.keep", object()),
                ("modules_to_save.default.x", object()),
                ("original_module.y", object()),
                ("skipme.z", object()),
            ]

    helper._sync_peft_params(_Model())
    assert "merge" in pushed and "unmerge" in pushed
    # Filters applied: skip prefix/original_module; strip modules_to_save.
    assert any("keep" in name for name in pushed)
    assert "modules_to_save.default.x" not in pushed
    assert "skipme.z" not in pushed


def test_generate_calls_custom_sync(monkeypatch):
    helper = _helper_with_client()
    sync_calls = []
    ensure_calls = []

    monkeypatch.setattr(helper, "maybe_sync_weights", lambda *a, **k: None)
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
        ensure_client=lambda: ensure_calls.append("ensure"),
        sync_model=lambda _m: sync_calls.append("sync"),
    )
    assert grouped == [["ok"]] and meta is None
    assert ensure_calls == ["ensure"]
    assert sync_calls == ["sync"]


def test_client_callable_wraps_updates(monkeypatch):
    helper = _helper_with_client()

    class _Client:
        def update_named_param(self, name, data):
            return f"{name}:{data}"

    helper._vllm_client = _Client()
    wrapped = helper._client_callable("update_named_param")
    assert callable(wrapped)
    assert wrapped("x", "y") == "x:y"


def test_run_vllm_rounds_records_failures(monkeypatch):
    helper = _helper_with_client()
    helper.ctx.generation_stats = {
        "vllm_failed_prompts": 0,
        "vllm_retry_rounds": 0,
        "vllm_backfilled_prompts": 0,
        "vllm_weight_syncs": 0,
    }
    state = vllm._VLLMGenerationState(
        prompts=["a", "b"],
        target_counts=[1, 1],
        requested_n=1,
        round_limit=1,
        track_logprobs=False,
    )
    # Force pending -> failure path
    monkeypatch.setattr(state, "pending_indices", lambda: [0, 1])
    monkeypatch.setattr(helper, "_execute_vllm_request", lambda *_a, **_k: False)
    helper._run_vllm_rounds(state)
    assert helper.ctx.generation_stats["vllm_failed_prompts"] == 2


def test_sync_model_params_walks_summon_full_params(monkeypatch):
    helper = _helper_with_client()
    helper._fsdp_cls = None
    pushes: List[str] = []
    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, _p: pushes.append(name)
    )
    monkeypatch.setattr(helper, "_reset_vllm_cache", lambda: pushes.append("reset"))

    class _Leaf:
        def named_children(self):
            return []

        def named_parameters(self):
            return [
                ("_fsdp_wrapped_module.dup", object()),
                ("dup", object()),
            ]

    class _Root:
        def summon_full_params(self):
            return None

        def named_children(self):
            return [("leaf", _Leaf())]

    helper._sync_model_params_to_vllm(_Root())
    assert pushes[-1] == "reset"
    assert pushes.count("leaf.dup") == 1  # deduped despite wrapper variant


def test_sync_standard_params_child_traversal(monkeypatch):
    helper = _helper_with_client()
    pushes: List[str] = []
    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, _p: pushes.append(name)
    )

    class _ChildSkip:
        named_parameters = None

    class _ChildDup:
        def named_parameters(self):
            return [("_checkpoint_wrapped_module.same", object()), ("same", object())]

    class _Model:
        def named_children(self):
            return [("skip", _ChildSkip()), ("c", _ChildDup())]

    helper._sync_standard_params(_Model())
    assert pushes == ["c.same"]


def test_sync_fsdp_params_dedups_child_and_root(monkeypatch):
    helper = _helper_with_client()
    pushes: List[str] = []
    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, _p: pushes.append(name)
    )

    class _Param:
        def __init__(self, name):
            self.name = name

    class _FSDP:
        @staticmethod
        def summon_full_params(_module, recurse=False, writeback=False):
            from contextlib import contextmanager

            @contextmanager
            def _ctx():
                yield

            return _ctx()

    class _Child(_FSDP):
        def named_children(self):
            return []

        def named_parameters(self):
            return [
                ("_fsdp_wrapped_module.dup", _Param("x")),
                ("dup", _Param("y")),
            ]

    class _Root(_FSDP):
        def __init__(self):
            self.child = _Child()

        def named_children(self):
            return [("child", self.child)]

        def named_parameters(self):
            return [
                ("_fsdp_wrapped_module.rdup", _Param("a")),
                ("rdup", _Param("b")),
            ]

    helper._sync_fsdp_params(_Root(), fsdp_cls=_FSDP)
    # Deduped: only one for child and one for root despite duplicates.
    assert pushes.count("child.dup") == 1
    assert pushes.count("rdup") == 1


def test_generate_updates_stats_when_custom_sync_used(monkeypatch):
    helper = _helper_with_client()
    helper.ctx.generation_stats = {}
    monkeypatch.setattr(
        helper, "_prepare_vllm_targets", lambda *_a, **_k: (["p"], [1], None)
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
    assert grouped == [["ok"]] and meta is None
    assert helper.ctx.generation_stats["vllm_weight_syncs"] == 1
