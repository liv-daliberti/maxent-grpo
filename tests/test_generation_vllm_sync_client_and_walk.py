"""
Tests for vLLMGenerationHelper client/init and summon_full_params walk branches.
"""

from types import SimpleNamespace
from typing import List

import maxent_grpo.generation.vllm as vllm


class _StubAccel:
    def __init__(self):
        self.is_main_process = True
        self.num_processes = 1
        self.process_index = 0

    def unwrap_model(self, model):
        return model


def _ctx():
    return SimpleNamespace(
        accelerator=_StubAccel(),
        generation_stats={},
        vllm_sync_weights=True,
        vllm_request_logprobs=False,
        vllm_url="http://localhost:8000",
        vllm_backfill_local=False,
        model=SimpleNamespace(),
    )


def test_ensure_vllm_client_handles_non_callable_cls(monkeypatch):
    helper = vllm.VLLMGenerationHelper(_ctx(), lambda *_a, **_k: ([], None))
    helper.ctx.vllm_sync_weights = True

    def _import_bad():
        return "not-a-class"

    ready = helper._ensure_vllm_client(import_vllm_client_cls=_import_bad)
    assert ready is False
    assert helper._vllm_client is None
    assert helper._vllm_sync_ready is False


def test_ensure_vllm_client_init_missing(monkeypatch):
    helper = vllm.VLLMGenerationHelper(_ctx(), lambda *_a, **_k: ([], None))

    class _Client:
        def __init__(self, **_kwargs):
            pass

    ready = helper._ensure_vllm_client(import_vllm_client_cls=lambda: _Client)
    assert ready is False
    assert helper._vllm_client is None
    assert helper._vllm_sync_ready is False


def test_maybe_sync_weights_short_circuit(monkeypatch):
    helper = vllm.VLLMGenerationHelper(_ctx(), lambda *_a, **_k: ([], None))
    helper.ctx.generation_stats = {}
    # ensure_client returns False -> nothing happens
    helper.maybe_sync_weights(ensure_client=lambda: False)
    assert helper.ctx.generation_stats == {}


def test_sync_model_params_walks_summon_full_params(monkeypatch):
    helper = vllm.VLLMGenerationHelper(_ctx(), lambda *_a, **_k: ([], None))
    helper._fsdp_cls = None
    pushed: List[str] = []
    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, _p: pushed.append(name)
    )
    monkeypatch.setattr(helper, "_reset_vllm_cache", lambda: pushed.append("reset"))

    class _Leaf:
        def named_parameters(self):
            return [("a", object())]

        def named_children(self):
            return []

    class _Root:
        def summon_full_params(self):
            return None

        def named_children(self):
            return [("leaf", _Leaf())]

        def named_parameters(self):
            return [("_checkpoint_wrapped_module.r", object())]

    helper._sync_model_params_to_vllm(_Root())
    assert pushed[-1] == "reset"
    assert any("r" in name for name in pushed)
    assert any("leaf.a" in name for name in pushed)
