"""Unit coverage for training.generation.generator.CompletionGenerator."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from maxent_grpo.training.generation import generator


pytestmark = pytest.mark.generation


class _DummyHelper:
    def __init__(self, ctx, fallback):
        self.ctx = ctx
        self.fallback = fallback
        self._vllm_client = None
        self._vllm_sync_ready = False
        self._last_vllm_synced_step = None
        self._fsdp_cls = None


def _make_ctx(use_vllm: bool = False):
    return SimpleNamespace(use_vllm=use_vllm, as_dict=lambda: {"use_vllm": use_vllm})


def _make_generator(monkeypatch, *, use_vllm: bool = False, ctx=None, local_fn=None):
    monkeypatch.setattr(generator, "VLLMGenerationHelper", _DummyHelper)
    local_fn = local_fn or (
        lambda self, prompts, num_samples, per_prompt_counts=None: (
            "local",
            prompts,
            num_samples,
            per_prompt_counts,
        )
    )
    monkeypatch.setattr(
        generator.CompletionGenerator,
        "_generate_local",
        local_fn,
        raising=False,
    )
    return generator.CompletionGenerator(ctx or _make_ctx(use_vllm))


def test_init_sets_helper_hooks(monkeypatch):
    ctx = _make_ctx()

    def _local(self, prompts, num_samples, per_prompt_counts=None):
        return "sentinel"

    gen = _make_generator(monkeypatch, ctx=ctx, local_fn=_local)
    helper = gen._vllm_helper

    assert isinstance(helper, _DummyHelper)
    assert helper.ctx is ctx
    assert getattr(helper.fallback, "__func__", helper.fallback) is _local
    assert helper._safe_generate is generator.safe_generate
    assert helper._scatter_object is generator._scatter_object
    assert helper._time is generator.time
    assert helper._is_peft_model_safe is generator._is_peft_model_safe
    assert (
        getattr(helper._fallback_generate, "__func__", helper._fallback_generate)
        is _local
    )


def test_describe_delegates_to_context(monkeypatch):
    ctx = _make_ctx()
    gen = _make_generator(monkeypatch, ctx=ctx)

    assert gen.describe() == ctx.as_dict()


def test_generate_returns_empty_for_no_prompts(monkeypatch):
    gen = _make_generator(monkeypatch)
    gen._generate_local = lambda *_a, **_k: (_ for _ in ()).throw(
        RuntimeError("unused")
    )

    assert gen.generate([], 2) == ([], None)


def test_generate_validates_per_prompt_counts(monkeypatch):
    gen = _make_generator(monkeypatch)

    with pytest.raises(ValueError):
        gen.generate(["only"], 1, per_prompt_counts=[1, 2])


def test_generate_routes_to_vllm_when_enabled(monkeypatch):
    gen = _make_generator(monkeypatch, use_vllm=True)
    calls = {}

    def _fake_vllm(prompts, num_samples, per_prompt_counts=None):
        calls["args"] = (prompts, num_samples, per_prompt_counts)
        return ["vllm"], None

    gen._generate_vllm_collective = _fake_vllm

    result = gen.generate(["a", "b"], 3, per_prompt_counts=[1, 2])
    assert result == (["vllm"], None)
    assert calls["args"] == (["a", "b"], 3, [1, 2])


def test_generate_routes_to_local_when_vllm_disabled(monkeypatch):
    gen = _make_generator(monkeypatch, use_vllm=False)
    calls = {}

    def _fake_local(prompts, num_samples, per_prompt_counts=None):
        calls["args"] = (prompts, num_samples, per_prompt_counts)
        return ["local"], None

    gen._generate_local = _fake_local

    result = gen.generate(["prompt"], 2)
    assert result == (["local"], None)
    assert calls["args"] == (["prompt"], 2, None)


def test_init_handles_vllm_mixin_failure(monkeypatch):
    ctx = SimpleNamespace(accelerator=object(), use_vllm=False, as_dict=lambda: {})

    def _raise_init(self, *_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(generator, "VLLMGenerationHelper", _DummyHelper)
    monkeypatch.setattr(
        generator, "VLLMGenerationMixin", SimpleNamespace(__init__=_raise_init)
    )
    gen = generator.CompletionGenerator(ctx)
    helper = gen._vllm_helper
    assert helper is not None
    assert helper._safe_generate is generator.safe_generate
    assert helper._scatter_object is generator._scatter_object
    assert helper._time is generator.time
    assert helper._is_peft_model_safe is generator._is_peft_model_safe
    assert getattr(
        helper._fallback_generate, "__func__", helper._fallback_generate
    ) is getattr(gen._generate_local, "__func__", gen._generate_local)


def test_property_proxies_point_at_helper(monkeypatch):
    gen = _make_generator(monkeypatch)

    gen._vllm_client = "client"
    gen._vllm_sync_ready = True
    gen._last_vllm_synced_step = 7
    gen._fsdp_cls = "fsdp"

    assert gen._vllm_client == "client"
    assert gen._vllm_sync_ready is True
    assert gen._last_vllm_synced_step == 7
    assert gen._fsdp_cls == "fsdp"
    assert gen._vllm_helper._vllm_client == "client"
    assert gen._vllm_helper._vllm_sync_ready is True
    assert gen._vllm_helper._last_vllm_synced_step == 7
    assert gen._vllm_helper._fsdp_cls == "fsdp"
