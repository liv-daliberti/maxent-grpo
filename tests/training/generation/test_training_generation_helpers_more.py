"""Additional unit tests for training.generation.helpers."""

from __future__ import annotations

from importlib import reload, import_module
from types import SimpleNamespace
import pytest
from maxent_grpo.training.runtime.prompts import GenerationPenaltyConfig
from maxent_grpo.training.runtime.setup import VLLMClientConfig


def _make_ctx(helpers=None, use_vllm: bool = False):
    if helpers is None:
        from maxent_grpo.training.generation import helpers as _helpers

        helpers = _helpers
    vllm_cfg = VLLMClientConfig(
        url="http://localhost",
        rounds_cfg=1,
        retry_sleep=0.0,
        backfill_local=use_vllm,
        request_logprobs=False,
    )
    return helpers.GenerationContext(
        max_prompt_len=4,
        max_completion_len=4,
        gen_temperature=0.1,
        gen_top_p=0.9,
        use_vllm=use_vllm,
        vllm=vllm_cfg,
        accelerator=SimpleNamespace(
            num_processes=1,
            is_main_process=True,
            unwrap_model=lambda m: m,
            autocast=None,
            process_index=0,
        ),
        model=SimpleNamespace(),
        tokenizer=SimpleNamespace(
            decode=lambda ids, skip_special_tokens=True: "".join(str(x) for x in ids)
        ),
        generation_stats={},
        device=SimpleNamespace(type="cpu"),
        penalty=GenerationPenaltyConfig(),
    )


@pytest.fixture
def helpers(monkeypatch, training_stubs):
    # Ensure torch stub is in place via training_stubs fixture.
    return reload(import_module("maxent_grpo.training.generation.helpers"))


def test_generation_context_penalty_proxies(helpers):
    gctx = _make_ctx(helpers)
    gctx.gen_top_k = 5
    assert gctx.gen_top_k == 5 and gctx.penalty.gen_top_k == 5
    gctx.gen_stop_sequences = ["</s>"]
    assert gctx.penalty.gen_stop_sequences == ["</s>"]


def test_vllm_base_url_and_summary(helpers):
    gen = helpers.CompletionGenerator(_make_ctx(helpers))
    assert gen._vllm_base_url("http://host:8000/generate") == "http://host:8000"
    summary = gen._summarize_grouped([["abc"], ["def", "ghi"]], limit=1)
    assert "len=" in summary and "sample" in summary


def test_decode_sequences_and_coalesce(helpers):
    toks = SimpleNamespace(
        decode=lambda ids, skip_special_tokens=True: "".join(map(str, ids))
    )
    sequences = [[0, 1, 2], [3, 4]]
    outs = helpers.CompletionGenerator._decode_sequences(sequences, [1, 1], toks)
    assert outs == ["12", "4"]
    groups = [["a"], ["b"], ["c"], ["d"]]
    regrouped, regrouped_meta = helpers.CompletionGenerator._coalesce_grouped_outputs(
        groups, prompt_count=2, requested_n=2, meta=None
    )
    assert len(regrouped) == 2 and regrouped[0] == ["a", "b"]
    # Mismatch leaves meta as None
    same_groups, meta = helpers.CompletionGenerator._coalesce_grouped_outputs(
        [["a"], ["b"], ["c"]], prompt_count=2, requested_n=2, meta=[[], [], []]
    )
    assert meta is None


def test_merge_vllm_results_tracks_overflow(helpers):
    ctx = _make_ctx(helpers)
    ctx.generation_stats = {}
    gen = helpers.CompletionGenerator(ctx)
    state = helpers._VLLMGenerationState(
        prompts=["p1"],
        target_counts=[1],
        requested_n=1,
        round_limit=1,
        track_logprobs=False,
    )
    state.aggregated[0] = ["x", "y"]
    gen._merge_vllm_results(
        state,
        grouped=[["a", "b"]],
        grouped_meta=[[None, None]],
        pending_indices=[0],
    )
    assert ctx.generation_stats.get("vllm_excess_completions", 0) > 0


def test_backfill_missing_invokes_local(monkeypatch, helpers):
    ctx = _make_ctx(helpers, use_vllm=True)
    ctx.generation_stats = {"vllm_backfilled_prompts": 0}
    gen = helpers.CompletionGenerator(ctx)
    state = helpers._VLLMGenerationState(
        prompts=["p1"],
        target_counts=[2],
        requested_n=2,
        round_limit=1,
        track_logprobs=False,
    )
    monkeypatch.setattr(
        gen,
        "_generate_local",
        lambda prompts, num_samples, per_prompt_counts=None: ([["loc1", "loc2"]], None),
    )
    gen._backfill_missing(state, [0])
    assert ctx.generation_stats["vllm_backfilled_prompts"] == 1
    assert state.aggregated[0]  # backfilled


def test_generate_validates_counts(helpers):
    gen = helpers.CompletionGenerator(_make_ctx(helpers))
    with pytest.raises(ValueError):
        gen.generate(["p1"], 1, per_prompt_counts=[1, 2])


def test_pluck_and_scatter_single_rank(helpers):
    ctx = _make_ctx(helpers)
    gen = helpers.CompletionGenerator(ctx)
    grouped_all = [["a"], ["b"]]
    meta_all = [["m1"], ["m2"]]
    offsets = [0, 1, 2]
    grouped, meta = gen._pluck_rank_outputs(
        grouped_all, meta_all, offsets, prompts=["p1", "p2"]
    )
    assert grouped == grouped_all and meta == meta_all
    grouped_local, meta_local = gen._scatter_vllm_payload(
        ["p1", "p2"], offsets, grouped_all, meta_all
    )
    assert grouped_local == grouped
    assert meta_local == meta


def test_request_vllm_batch_handles_mismatch(monkeypatch, caplog, helpers):
    ctx = _make_ctx(helpers, use_vllm=True)
    gen = helpers.CompletionGenerator(ctx)
    # Force mismatch between pending count and returned groups to hit warning path
    monkeypatch.setattr(
        gen,
        "_invoke_vllm_requests",
        lambda prompts, n: ([["a"]], None, 0.1),
    )
    caplog.set_level("WARNING")
    grouped, meta = gen._request_vllm_batch(["p1", "p2"], request_count=1)
    assert grouped is None and meta is None
