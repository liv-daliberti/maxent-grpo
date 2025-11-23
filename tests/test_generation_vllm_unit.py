"""Unit tests for lightweight helpers in ``generation.vllm``."""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest


def _load_vllm(monkeypatch):
    """Import generation.vllm with torch/accelerate stubs installed."""

    torch_stub = ModuleType("torch")
    torch_dist = SimpleNamespace(
        is_available=lambda: False,
        is_initialized=lambda: False,
    )
    torch_stub.distributed = torch_dist
    torch_stub.Tensor = type("Tensor", (), {})
    torch_utils = ModuleType("torch.utils")
    torch_utils_data = ModuleType("torch.utils.data")
    torch_utils_data.DataLoader = type("DataLoader", (), {})
    torch_utils_data.Sampler = type("Sampler", (), {})
    torch_utils.data = torch_utils_data
    monkeypatch.setitem(sys.modules, "torch.utils", torch_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.data", torch_utils_data)
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    accelerate_stub = ModuleType("accelerate")
    accelerate_stub.Accelerator = type("Accelerator", (), {})
    monkeypatch.setitem(sys.modules, "accelerate", accelerate_stub)

    patches_mod = ModuleType("patches.vllm")
    patches_mod.VLLMLogprobResult = type("VLLMLogprobResult", (), {})

    def _safe_generate(**_kwargs):
        return [[_kwargs["prompts"][0]]], None, 1.0

    patches_mod.safe_generate = _safe_generate
    monkeypatch.setitem(sys.modules, "patches.vllm", patches_mod)

    # Ensure the generation package path is importable
    sys.modules.pop("generation.vllm", None)
    module = importlib.import_module("generation.vllm")
    return module


def _ctx(**kwargs):
    defaults = dict(
        accelerator=SimpleNamespace(
            is_main_process=True,
            process_index=0,
            num_processes=1,
            state=None,
        ),
        vllm_sync_weights=False,
        vllm_url="http://localhost:8000/generate",
        vllm_rounds_cfg=0,
        vllm_request_logprobs=False,
        vllm_retry_sleep=0.0,
        vllm_backfill_local=False,
        vllm_stop_sequences=None,
        vllm_top_k=None,
        vllm_best_of=None,
        vllm_logit_bias=None,
        vllm_guided_json=None,
        vllm_guided_regex=None,
        vllm_request_id_prefix=None,
        vllm_timeout=10,
        vllm_max_retries=0,
        vllm_backoff=None,
        gen_stop_sequences=None,
        gen_top_k=None,
        gen_best_of=None,
        gen_temperature=0.1,
        gen_top_p=0.9,
        gen_frequency_penalty=0.0,
        gen_presence_penalty=0.0,
        max_completion_len=8,
        tokenizer=None,
        prompt_char_limit=None,
        generation_stats={
            "vllm_retry_rounds": 0,
            "vllm_backfilled_prompts": 0,
            "vllm_failed_prompts": 0,
            "vllm_excess_prompts": 0,
            "vllm_excess_completions": 0,
        },
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_optional_import_returns_none_and_module(monkeypatch):
    module = _load_vllm(monkeypatch)
    dummy = ModuleType("dummy_mod")
    monkeypatch.setitem(sys.modules, "dummy_mod", dummy)
    assert module._optional_import("dummy_mod") is dummy
    assert module._optional_import("does_not_exist") is None


def test_zero3_gather_factory_handles_zero3(monkeypatch):
    module = _load_vllm(monkeypatch)

    @contextmanager
    def _gather_ctx(_params):
        yield "gathered"

    zero_mod = ModuleType("deepspeed.zero")
    zero_mod.GatheredParameters = _gather_ctx
    deepspeed_mod = ModuleType("deepspeed")
    deepspeed_mod.zero = zero_mod
    monkeypatch.setitem(sys.modules, "deepspeed", deepspeed_mod)
    monkeypatch.setitem(sys.modules, "deepspeed.zero", zero_mod)

    accel = SimpleNamespace(
        state=SimpleNamespace(deepspeed_plugin=SimpleNamespace(zero_stage=3))
    )
    factory = module._zero3_gather_factory(accel)
    with factory([1, 2]) as token:
        assert token == "gathered"


def test_is_peft_model_safe_uses_accelerate_utils(monkeypatch):
    module = _load_vllm(monkeypatch)
    accel_utils = ModuleType("accelerate.utils")
    accel_utils.is_peft_model = lambda target: getattr(target, "peft", False)
    monkeypatch.setitem(sys.modules, "accelerate.utils", accel_utils)
    assert module._is_peft_model_safe(SimpleNamespace(peft=True)) is True
    monkeypatch.delitem(sys.modules, "accelerate.utils")
    assert module._is_peft_model_safe(object()) is False


def test_import_vllm_client_cls(monkeypatch):
    module = _load_vllm(monkeypatch)
    vllm_mod = ModuleType("trl.extras.vllm_client")

    class _Client:
        pass

    vllm_mod.VLLMClient = _Client
    monkeypatch.setitem(sys.modules, "trl.extras.vllm_client", vllm_mod)
    assert module._import_vllm_client_cls() is _Client
    monkeypatch.delitem(sys.modules, "trl.extras.vllm_client")
    assert module._import_vllm_client_cls() is None


def test_vllm_generation_state_pending_and_trim_single_prompt(monkeypatch):
    module = _load_vllm(monkeypatch)
    state = module._VLLMGenerationState(
        prompts=["p1", "p2"],
        target_counts=[2, 0],
        requested_n=2,
        round_limit=1,
        track_logprobs=True,
    )
    assert state.pending_indices() == [0]
    assert state.remaining_counts([0, 1]) == [2, 0]
    state.aggregated[0].extend(["a", "b", "c"])
    state.aggregated_meta[0].extend([None, None, None])
    trimmed, meta = state.trim()
    assert trimmed[0] == ["a", "b"]
    assert meta[0] == [None, None]
    state.drop_meta()
    assert state.aggregated_meta is None


def test_helper_base_url_strips_generate_and_scheme(monkeypatch):
    module = _load_vllm(monkeypatch)
    helper = module.VLLMGenerationHelper(_ctx(), lambda *_args: ([], None))
    assert helper._vllm_base_url("http://host:8000/generate") == "http://host:8000"
    assert helper._vllm_base_url("invalid://") == "invalid:"


def test_prepare_vllm_targets_deduplicates(monkeypatch):
    module = _load_vllm(monkeypatch)
    helper = module.VLLMGenerationHelper(_ctx(), lambda *_args: ([], None))
    monkeypatch.setenv("MAXENT_VLLM_DEDUP", "1")
    prompts, counts, mapping = helper._prepare_vllm_targets(
        ["a", "a", "b"], 2, [1, 2, 3]
    )
    assert prompts == ["a", "b"]
    assert counts == [1, 3]
    assert mapping == [0, 0, 1]
    with pytest.raises(ValueError):
        helper._prepare_vllm_targets(["a"], 1, [1, 2])


def test_resolve_round_limit_and_expand_dedup(monkeypatch):
    module = _load_vllm(monkeypatch)
    helper = module.VLLMGenerationHelper(_ctx(vllm_rounds_cfg=5), lambda *_: ([], None))
    assert helper._resolve_vllm_round_limit(3) == 5
    grouped, meta = helper._expand_dedup_results([["x"]], None, [0, 0])
    assert grouped == [["x"], ["x"]]
    assert meta is None


def test_vllm_generation_state_pending_and_trim(monkeypatch):
    module = _load_vllm(monkeypatch)
    state = module._VLLMGenerationState(
        prompts=["p1", "p2"],
        target_counts=[2, 1],
        requested_n=2,
        round_limit=2,
        track_logprobs=True,
    )
    state.aggregated[0].append("a")
    state.aggregated_meta[0].append("meta_a")
    assert state.pending_indices() == [0, 1]
    assert state.remaining_counts([0, 1]) == [1, 1]
    trimmed, trimmed_meta = state.trim()
    assert trimmed[0] == ["a"] and trimmed[1] == []
    assert trimmed_meta[0] == ["meta_a"] and trimmed_meta[1] == []
    state.drop_meta()
    assert state.aggregated_meta is None


def test_vllm_base_url_and_client_callable(monkeypatch):
    module = _load_vllm(monkeypatch)
    helper = module.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    assert helper._vllm_base_url("http://host:8000/generate") == "http://host:8000"
    helper._vllm_client = SimpleNamespace(update_named_param=lambda *_a, **_k: "ok")
    wrapped = helper._client_callable("update_named_param")
    assert callable(wrapped) and wrapped("n", None) == "ok"
    assert helper._client_callable("missing") is None


def test_ensure_vllm_client_handles_missing(monkeypatch):
    module = _load_vllm(monkeypatch)
    ctx = _ctx(vllm_sync_weights=True)
    helper = module.VLLMGenerationHelper(ctx, lambda *_: ([], None))
    # Missing VLLMClient should return False and not raise.
    assert helper._ensure_vllm_client() is False


def test_coalesce_grouped_outputs_merges_chunks(monkeypatch):
    module = _load_vllm(monkeypatch)
    helper = module.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    groups = [["a"], ["b"], ["c"], ["d"]]
    meta = [[1], [2], [3], [4]]
    regrouped, regrouped_meta = helper._coalesce_grouped_outputs(groups, 2, 2, meta)
    assert regrouped == [["a", "b"], ["c", "d"]]
    assert regrouped_meta == [[1, 2], [3, 4]]
    merged, merged_meta = helper._merge_group_chunk(
        [["x"], ["y"]], [[10], [11]], requested_n=1
    )
    assert merged == ["x"]
    assert merged_meta == [10]


def test_build_vllm_request_kwargs_and_latency(monkeypatch):
    module = _load_vllm(monkeypatch)
    ctx = _ctx(gen_stop_sequences=["stop"], gen_top_k=5, gen_best_of=2)
    helper = module.VLLMGenerationHelper(ctx, lambda *_: ([], None))
    kwargs = helper._build_vllm_request_kwargs(["p"], 1)
    assert kwargs["stop"] == ["stop"]
    assert kwargs["top_k"] == 5
    stats = ctx.generation_stats
    helper._record_vllm_latency(12.5)
    assert stats["vllm_last_latency_ms"] == 12.5
    assert stats["vllm_latency_calls"] == 1


def test_scatter_vllm_payload_single_rank(monkeypatch):
    module = _load_vllm(monkeypatch)
    accel = SimpleNamespace(
        is_main_process=True,
        process_index=0,
        num_processes=1,
    )
    helper = module.VLLMGenerationHelper(_ctx(accelerator=accel), lambda *_: ([], None))
    grouped, meta = helper._scatter_vllm_payload(
        flat_prompts=["p1", "p2"],
        offsets=[0],
        grouped_all=[["a"], ["b"]],
        meta_all=None,
    )
    assert grouped == [["a"], ["b"]]
    assert meta is None


def test_collective_gather_and_scatter_helpers(monkeypatch):
    module = _load_vllm(monkeypatch)

    # gather_object path
    class _AccelGather:
        def gather_object(self, value):
            return [value, ["peer"]]

    assert module._gather_object_list(_AccelGather(), ["local"]) == [
        ["local"],
        ["peer"],
    ]

    # dist path
    def _all_gather_object(out, val):
        out[0] = val
        out[1] = ["peer"]

    dist = ModuleType("torch.distributed")
    dist.__spec__ = SimpleNamespace(name="torch.distributed", has_location=False)
    dist.is_available = lambda: True
    dist.is_initialized = lambda: True
    dist.get_world_size = lambda: 2
    dist.all_gather_object = _all_gather_object
    monkeypatch.setattr(module.torch, "distributed", dist)  # type: ignore[attr-defined]
    gathered = module._gather_object_list(SimpleNamespace(), ["x"])
    assert len(gathered) == 2
    assert gathered[0] == ["x"]
    assert gathered[1] == ["peer"]

    # scatter_object path (accelerate)
    class _AccelScatter:
        num_processes = 2
        process_index = 1

        def scatter_object(self, payload, src=0):
            return ["from_scatter"]

    assert module._scatter_object(_AccelScatter(), [["a"], ["b"]], src=0) == [
        "from_scatter"
    ]

    # scatter_object dist fallback
    dist.scatter_object_list = lambda out, payload, src=0: out.__setitem__(0, ["dist"])
    monkeypatch.setattr(sys.modules["torch"], "distributed", dist)
    accel = SimpleNamespace(num_processes=2, process_index=1)
    assert module._scatter_object(accel, [["a"], ["b"]], src=0) == ["dist"]


def test_backfill_missing_drops_meta(monkeypatch):
    module = _load_vllm(monkeypatch)
    ctx = _ctx(vllm_backfill_local=True)
    helper = module.VLLMGenerationHelper(
        ctx, lambda prompts, *_: ([["fb"]] * len(prompts), None)
    )
    state = module._VLLMGenerationState(
        prompts=["p1"],
        target_counts=[1],
        requested_n=1,
        round_limit=1,
        track_logprobs=True,
    )
    state.aggregated_meta[0].append(None)
    helper._backfill_missing(state, [0])
    assert state.aggregated[0] == ["fb"]
    assert state.aggregated_meta is None
