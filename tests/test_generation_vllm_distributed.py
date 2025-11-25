"""Tests for vLLM distributed scatter/gather helpers."""

from __future__ import annotations

import sys
from types import SimpleNamespace

from maxent_grpo.generation import vllm_distributed as dist


def test_current_torch_prefers_vllm_module(monkeypatch):
    sentinel = SimpleNamespace(torch="stub")
    monkeypatch.setitem(sys.modules, "maxent_grpo.generation.vllm", sentinel)
    assert dist._current_torch() == "stub"
    monkeypatch.delitem(sys.modules, "maxent_grpo.generation.vllm", raising=False)


def test_current_torch_falls_back_to_module(monkeypatch):
    monkeypatch.delitem(sys.modules, "maxent_grpo.generation.vllm", raising=False)
    # Should return the module-level torch when no shim is present.
    assert dist._current_torch() is dist.torch


def test_scatter_vllm_payload_handles_none_result(monkeypatch):
    class _Accel:
        num_processes = 2
        process_index = 1
        is_main_process = False

    class _Mixin(dist.VLLMDistributedMixin):
        def __init__(self):
            self.ctx = SimpleNamespace(accelerator=_Accel())
            # Return None to exercise the early exit branch.
            self._scatter_object = lambda accel, payload, src=0: None

        def _pluck_rank_outputs(self, *args, **kwargs):
            raise AssertionError("should not be called")

    helper = _Mixin()
    grouped, meta = helper._scatter_vllm_payload(
        flat_prompts=["p0", "p1"],
        offsets=[0, 1],
        grouped_all=[["g0"], ["g1"]],
        meta_all=[["m0"], ["m1"]],
    )
    assert grouped == [] and meta is None
