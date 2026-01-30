"""Extra unit tests for training.rollout.helpers (local generation paths)."""

from __future__ import annotations

import builtins
from contextlib import AbstractContextManager
from types import SimpleNamespace


from maxent_grpo.training.rollout import helpers as gen_helpers
from maxent_grpo.training.rollout import local as local_gen
from maxent_grpo.training.runtime.prompts import GenerationPenaltyConfig
from maxent_grpo.training.runtime.setup import VLLMClientConfig


def _ctx() -> SimpleNamespace:
    """Return a lightweight context with only the attributes the generator touches."""

    vllm_cfg = VLLMClientConfig(
        url="http://localhost",
        rounds_cfg=1,
        retry_sleep=0.0,
        backfill_local=False,
        request_logprobs=False,
    )
    return SimpleNamespace(
        accelerator=SimpleNamespace(
            unwrap_model=lambda m: m,
            process_index=0,
            num_processes=1,
            is_main_process=True,
        ),
        model=object(),
        tokenizer=SimpleNamespace(),
        generation_stats={},
        device="cpu",
        penalty=GenerationPenaltyConfig(),
        gen_temperature=0.2,
        gen_top_p=0.9,
        gen_top_k=None,
        max_completion_len=4,
        max_prompt_len=8,
        use_vllm=False,
        vllm=vllm_cfg,
        vllm_rounds_cfg=1,
    )


def test_generate_local_groups_outputs(monkeypatch):
    ctx = _ctx()
    gen = gen_helpers.CompletionGenerator(ctx)

    # Force deterministic request/response plumbing without hitting torch.
    monkeypatch.setattr(gen, "_resolve_local_counts", lambda prompts, num, per: [1, 2])
    monkeypatch.setattr(
        gen,
        "_build_local_prompt_requests",
        lambda prompts, counts: (["p1", "p1", "p2"], [0, 0, 1]),
    )
    monkeypatch.setattr(
        gen, "_tokenize_expanded_prompts", lambda prompts: ("enc", [1] * len(prompts))
    )
    monkeypatch.setattr(
        gen,
        "_run_local_model",
        lambda enc, lens: [f"{p}-gen" for p in ["p1", "p1", "p2"]],
    )

    grouped, meta = gen._generate_local(["prompt1", "prompt2"], num_samples=2)
    assert grouped == [["p1-gen", "p1-gen"], ["p2-gen"]]
    assert meta is None


def test_generate_local_handles_empty_prompts():
    gen = gen_helpers.CompletionGenerator(_ctx())
    grouped, meta = gen._generate_local([], num_samples=1)
    assert grouped == [] and meta is None


def test_zero3_gather_factory_no_zero(monkeypatch):
    accel = SimpleNamespace(
        state=SimpleNamespace(deepspeed_plugin=SimpleNamespace(zero_stage=0))
    )
    gather = gen_helpers._zero3_gather_factory(accel)
    ctx = gather([])
    assert isinstance(ctx, AbstractContextManager)
    with ctx:
        pass  # should be a no-op context manager


def test_summarize_grouped_truncates_preview():
    summary = gen_helpers.CompletionGenerator._summarize_grouped(
        [["long-sample-text", "ignored"], "not-a-list"], limit=1
    )
    assert "len=2" in summary
    assert "long-sample-text" in summary


def test_prompt_char_limit_falls_back_without_helpers_import(monkeypatch):
    gen = local_gen.LocalGenerationMixin(SimpleNamespace(max_prompt_len=0))
    monkeypatch.setattr(local_gen, "PROMPT_CHAR_LIMIT", 7)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "maxent_grpo.training.rollout.helpers":
            raise ImportError("helpers missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert gen._prompt_char_limit() == 7


def test_generate_local_uses_truncate_fallback_when_helpers_missing(monkeypatch):
    gen = gen_helpers.CompletionGenerator(_ctx())
    monkeypatch.setattr(gen, "_prompt_char_limit", lambda: 3)
    monkeypatch.setattr(
        gen, "_resolve_local_counts", lambda prompts, num, per: [1] * len(prompts)
    )
    monkeypatch.setattr(
        gen,
        "_build_local_prompt_requests",
        lambda prompts, counts: (prompts, list(range(len(prompts)))),
    )
    monkeypatch.setattr(
        gen,
        "_tokenize_expanded_prompts",
        lambda prompts: ({"enc": prompts}, [1] * len(prompts)),
    )
    monkeypatch.setattr(
        gen,
        "_run_local_model",
        lambda enc, lens: [f"out-{p}" for p in enc["enc"]],
    )

    trunc_calls = []
    monkeypatch.setattr(
        local_gen,
        "_truncate_prompt",
        lambda prompt, limit: (
            trunc_calls.append((prompt, limit)) or f"short-{prompt}"
        ),
    )

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "maxent_grpo.training.rollout.helpers":
            raise ImportError("helpers unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    grouped, meta = gen._generate_local(["aaaa", "bbbb"], num_samples=1)
    assert grouped == [["out-short-aaaa"], ["out-short-bbbb"]]
    assert trunc_calls == [("aaaa", 3), ("bbbb", 3)]
    assert meta is None
