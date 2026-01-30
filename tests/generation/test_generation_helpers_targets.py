"""Focused tests for vLLM-related helpers in training.generation.helpers."""

from __future__ import annotations

import os
from types import SimpleNamespace


import maxent_grpo.training.generation.helpers as helpers
from maxent_grpo.training.run_helpers import VLLMClientConfig


def _make_ctx(**overrides):
    vllm_cfg = VLLMClientConfig(
        url=overrides.pop("vllm_url", "http://host:8000/generate"),
        rounds_cfg=overrides.pop("rounds_cfg", 0),
        retry_sleep=0.0,
        backfill_local=overrides.pop("backfill_local", False),
        request_logprobs=False,
        sync_weights=overrides.pop("sync_weights", False),
    )
    ctx = helpers.GenerationContext(
        max_prompt_len=4,
        max_completion_len=3,
        gen_temperature=1.0,
        gen_top_p=0.9,
        use_vllm=True,
        vllm=vllm_cfg,
        accelerator=SimpleNamespace(is_main_process=True),
        model=None,
        tokenizer=None,
        generation_stats={"vllm_failed_prompts": 0},
        device=helpers.torch.device("cpu"),
    )
    for key, val in overrides.items():
        setattr(ctx, key, val)
    return ctx


def test_vllm_base_url_and_round_limit(monkeypatch):
    ctx = _make_ctx(rounds_cfg=5)
    gen = helpers.CompletionGenerator(ctx)
    assert (
        gen._vllm_base_url("http://example.com:8000/generate")
        == "http://example.com:8000"
    )
    assert gen._vllm_base_url("localhost:8000/generate") == "localhost:8000"
    assert gen._resolve_vllm_round_limit(2) == 5  # rounds_cfg overrides requested_n
    ctx.vllm.rounds_cfg = 0
    assert gen._resolve_vllm_round_limit(3) == 3


def test_ensure_vllm_client_handles_missing_and_present(monkeypatch):
    ctx = _make_ctx(sync_weights=True)
    gen = helpers.CompletionGenerator(ctx)
    monkeypatch.setattr(helpers, "_import_vllm_client_cls", lambda: None)
    assert gen._ensure_vllm_client() is False
    assert gen._vllm_sync_ready is False

    class _Client:
        def __init__(self, base_url=None):
            self.base_url = base_url
            self.initted = False

        def init_communicator(self):
            self.initted = True

    monkeypatch.setattr(helpers, "_import_vllm_client_cls", lambda: _Client)
    assert gen._ensure_vllm_client() is True
    assert isinstance(gen._vllm_client, _Client)
    assert gen._vllm_client.initted is True
    # cached path
    assert gen._ensure_vllm_client() is True


def test_merge_results_tracks_overflow(monkeypatch):
    ctx = _make_ctx()
    ctx.generation_stats["vllm_excess_prompts"] = 0
    ctx.generation_stats["vllm_excess_completions"] = 0
    gen = helpers.CompletionGenerator(ctx)
    state = helpers._VLLMGenerationState(
        prompts=["p1"],
        target_counts=[1],
        requested_n=2,
        round_limit=1,
        track_logprobs=True,
    )
    gen._merge_vllm_results(
        state,
        grouped=[["a", "b"]],
        grouped_meta=[[None, None]],
        pending_indices=[0],
    )
    assert state.aggregated[0][0] == "a"
    assert ctx.generation_stats["vllm_excess_prompts"] == 1
    assert ctx.generation_stats["vllm_excess_completions"] == 1


def test_record_vllm_failure_and_coalesce(monkeypatch, caplog):
    ctx = _make_ctx(backfill_local=False)
    gen = helpers.CompletionGenerator(ctx)
    state = helpers._VLLMGenerationState(
        prompts=["p1", "p2"],
        target_counts=[1, 1],
        requested_n=1,
        round_limit=2,
        track_logprobs=False,
    )
    caplog.set_level("WARNING")
    gen._record_vllm_failure(state, [0, 1])
    assert ctx.generation_stats["vllm_failed_prompts"] == 2
    assert "remaining completions" in caplog.text

    groups = [["a"], ["b"]]
    merged, meta = gen._coalesce_grouped_outputs(groups, prompt_count=1, requested_n=2)
    assert merged == [["a", "b"]]
    assert meta is None  # meta dropped when regrouping mismatched shapes


def test_prepare_vllm_targets_and_dedup(monkeypatch):
    ctx = _make_ctx()
    gen = helpers.CompletionGenerator(ctx)
    prompts = ["p1", "p1", "p2"]
    counts = [2, 3, 1]
    os.environ["MAXENT_VLLM_DEDUP"] = "1"
    unique_prompts, target_counts, mapping = gen._prepare_vllm_targets(
        prompts, 2, counts
    )
    assert unique_prompts == ["p1", "p2"]
    assert target_counts == [2, 1]
    assert mapping == [0, 0, 1]
    os.environ.pop("MAXENT_VLLM_DEDUP", None)
