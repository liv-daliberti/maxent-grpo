"""
Copyright 2025 Liv d'Aliberti

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Additional coverage for training.generation.helpers.
"""

from __future__ import annotations

from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
import os
import sys
import pytest

from maxent_grpo.training.generation import helpers
from maxent_grpo.training.run_helpers import (
    GenerationPenaltyConfig,
    VLLMClientConfig,
)


def _ctx_base(**overrides):
    penalty = GenerationPenaltyConfig()
    vllm_cfg = VLLMClientConfig(
        url="http://host/generate",
        rounds_cfg=0,
        retry_sleep=0.0,
        backfill_local=False,
        request_logprobs=False,
    )
    ctx = helpers.GenerationContext(
        accelerator=SimpleNamespace(
            num_processes=1,
            process_index=0,
            is_main_process=True,
            gather_object=None,
            broadcast_object_list=None,
            scatter_object=None,
            unwrap_model=lambda m: m,
        ),
        model=SimpleNamespace(),
        tokenizer=SimpleNamespace(decode=lambda ids, skip_special_tokens=True: f"dec:{list(ids)}"),
        generation_stats={"current_step": 0, "vllm_backfilled_prompts": 0, "vllm_failed_prompts": 0},
        device="cpu",
        max_prompt_len=4,
        max_completion_len=3,
        gen_temperature=1.0,
        gen_top_p=0.9,
        use_vllm=True,
        vllm=vllm_cfg,
        penalty=penalty,
    )
    for key, val in overrides.items():
        setattr(ctx, key, val)
    return ctx


def test_zero3_gather_factory_uses_deepspeed(monkeypatch):
    gather_called = {}

    class _Gather:
        def __init__(self, params):
            gather_called["params"] = params

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    ds_mod = ModuleType("deepspeed")
    ds_mod.zero = SimpleNamespace(GatheredParameters=_Gather)
    monkeypatch.setitem(sys.modules, "deepspeed", ds_mod)
    accel = SimpleNamespace(state=SimpleNamespace(deepspeed_plugin=SimpleNamespace(zero_stage=3)))
    factory = helpers._zero3_gather_factory(accel)  # noqa: SLF001
    with factory(["p1"]) as ctx:
        assert isinstance(ctx, _Gather)
    assert gather_called["params"] == ["p1"]


def test_zero3_gather_factory_returns_nullcontext_when_disabled():
    accel = SimpleNamespace(state=SimpleNamespace(deepspeed_plugin=SimpleNamespace(zero_stage=1)))
    factory = helpers._zero3_gather_factory(accel)  # noqa: SLF001
    assert isinstance(factory([]), nullcontext)


def test_is_peft_model_safe_handles_missing_and_errors(monkeypatch):
    monkeypatch.delitem(sys.modules, "accelerate.utils", raising=False)
    assert helpers._is_peft_model_safe(object()) is False  # noqa: SLF001

    accel_utils = ModuleType("accelerate.utils")
    accel_utils.is_peft_model = lambda target: target == "ok"
    monkeypatch.setitem(sys.modules, "accelerate.utils", accel_utils)
    assert helpers._is_peft_model_safe("ok") is True  # noqa: SLF001
    accel_utils.is_peft_model = lambda _t: (_ for _ in ()).throw(TypeError())
    assert helpers._is_peft_model_safe("bad") is False  # noqa: SLF001


def test_import_vllm_client_cls(monkeypatch):
    monkeypatch.delitem(sys.modules, "trl.extras.vllm_client", raising=False)
    assert helpers._import_vllm_client_cls() is None  # noqa: SLF001
    mod = ModuleType("trl.extras.vllm_client")
    sentinel = object()
    mod.VLLMClient = sentinel
    monkeypatch.setitem(sys.modules, "trl.extras.vllm_client", mod)
    assert helpers._import_vllm_client_cls() is sentinel  # noqa: SLF001


def test_generation_context_accessors_and_as_dict():
    ctx = _ctx_base()
    ctx.gen_top_k = 7
    ctx.gen_best_of = 3
    ctx.gen_frequency_penalty = 0.4
    ctx.gen_presence_penalty = 0.1
    ctx.gen_stop_sequences = ["</s>"]
    desc = ctx.as_dict()
    assert desc["top_k"] == 7
    assert ctx.penalty.gen_stop_sequences == ["</s>"]


def test_vllm_generation_state_methods():
    with pytest.raises(ValueError):
        helpers._VLLMGenerationState(prompts=["a"], target_counts=[], requested_n=1, round_limit=1, track_logprobs=False)
    state = helpers._VLLMGenerationState(prompts=["a", "b"], target_counts=[1, 0], requested_n=2, round_limit=2, track_logprobs=True)
    assert state.pending_indices() == [0]
    assert state.remaining_counts([0, 1]) == [1, 0]
    grouped, meta = state.trim()
    assert grouped == [[], []]
    assert meta == [[], []]
    state.drop_meta()
    assert state.aggregated_meta is None


def test_vllm_base_url_and_round_limit():
    gen = helpers.CompletionGenerator(_ctx_base())
    assert gen._vllm_base_url("http://x:8000/generate") == "http://x:8000"
    assert gen._vllm_base_url("http://x:8000") == "http://x:8000"
    assert gen._resolve_vllm_round_limit(3) == 3
    gen.ctx.vllm.rounds_cfg = 2
    assert gen._resolve_vllm_round_limit(5) == 2


def test_seed_and_retry_incomplete_prompts(monkeypatch):
    grouped, meta = helpers.CompletionGenerator._seed_generation_groups(2, [["a"], []], [[None], []])  # noqa: SLF001
    assert grouped[0] == ["a"]
    completions = [["b"], ["c"]]
    metas = [["m1"], ["m2"]]

    def _generator(prompts, expected, remaining):
        assert remaining == [1]
        return completions, metas

    out_groups, out_meta = helpers.CompletionGenerator._retry_incomplete_prompts(  # noqa: SLF001
        None,
        ["p1", "p2"],
        _generator,
        expected_generations=1,
        aggregated_comps=grouped,
        aggregated_meta=meta,
        max_retry_rounds=1,
    )
    assert out_groups[1] == ["b"]
    assert out_meta[1] == ["m1"]


def test_build_local_prompt_requests_and_prompt_char_limit(monkeypatch):
    gen = helpers.CompletionGenerator(_ctx_base())
    prompts = ["p1", "p2"]
    expanded, idx = gen._build_local_prompt_requests(prompts, [2, 0])
    assert expanded == ["p1", "p1"]
    assert idx == [0, 0]
    monkeypatch.setattr(helpers, "PROMPT_CHAR_LIMIT", 0)
    gen.ctx.max_prompt_len = 0
    assert gen._prompt_char_limit() == 0
    monkeypatch.setattr(helpers, "PROMPT_CHAR_LIMIT", 10)
    gen.ctx.max_prompt_len = 2
    assert gen._prompt_char_limit() == 10


def test_decode_sequences_and_resolve_local_counts(monkeypatch):
    gen = helpers.CompletionGenerator(_ctx_base())
    sequences = helpers.require_torch("ctx").tensor([[1, 2, 3], [4, 5, 6]])
    out = gen._decode_sequences(sequences, [1, 2], gen.ctx.tokenizer)
    assert out[0].startswith("dec:")
    assert gen._resolve_local_counts(["a", "b"], 2, None) == [2, 2]
    with pytest.raises(ValueError):
        gen._resolve_local_counts(["a"], 1, [1, 2])


def test_coalesce_grouped_outputs_and_merge_chunk():
    merged, merged_meta = helpers.CompletionGenerator._merge_group_chunk(  # noqa: SLF001
        [["a"], ["b"]], [[["ma"]], [["mb"]]], requested_n=1
    )
    assert merged == ["a"]
    assert merged_meta == [["ma"]]
    groups, meta = helpers.CompletionGenerator._coalesce_grouped_outputs(  # noqa: SLF001
        [["a"], ["b"], ["c"], ["d"]],
        prompt_count=2,
        requested_n=2,
        meta=[[["ma"]], [["mb"]], [["mc"]], [["md"]]],
    )
    assert len(groups) == 2
    assert meta is not None and len(meta) == 2
    fallback_groups, fallback_meta = helpers.CompletionGenerator._coalesce_grouped_outputs(  # noqa: SLF001
        [["a"], ["b"], ["c"]],
        prompt_count=2,
        requested_n=1,
    )
    assert fallback_meta is None
    assert fallback_groups == [["a"], ["b"], ["c"]]


def test_prepare_vllm_targets_and_expand(monkeypatch):
    gen = helpers.CompletionGenerator(_ctx_base())
    prompts, counts, mapping = gen._prepare_vllm_targets(["p1", "p1"], 2, None)
    assert mapping is None and counts == [2, 2]
    monkeypatch.setenv("MAXENT_VLLM_DEDUP", "1")
    prompts2, counts2, mapping2 = gen._prepare_vllm_targets(["p1", "p1"], 1, [1, 2])
    assert prompts2 == ["p1"]
    assert counts2 == [1]
    assert mapping2 == [0, 0]
    grouped, meta = helpers.CompletionGenerator._expand_dedup_results(  # noqa: SLF001
        [["a"]], [["m"]], mapping2
    )
    assert grouped == [["a"], ["a"]]
    assert meta == [["m"], ["m"]]
    monkeypatch.delenv("MAXENT_VLLM_DEDUP", raising=False)


def test_gather_broadcast_and_scatter_helpers():
    accel = SimpleNamespace(
        num_processes=1,
        process_index=0,
        gather_object=lambda value: [value, value],
        broadcast_object_list=None,
        scatter_object=None,
    )
    assert helpers._gather_object_list(accel, ["a"]) == [["a"], ["a"]]
    helpers._broadcast_object_list(accel, [None, None], src=0)
    accel_multi = SimpleNamespace(
        num_processes=2,
        process_index=1,
        scatter_object=None,
    )
    result = helpers._scatter_object(accel_multi, [["x"], ["y"]], src=0)
    assert result == ["y"]