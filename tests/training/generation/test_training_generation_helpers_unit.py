"""Targeted unit tests for training.generation.helpers (lightweight stubs)."""

from __future__ import annotations

import importlib
import sys
from contextlib import nullcontext
from types import SimpleNamespace

from maxent_grpo.generation.common import _DEFAULT_RETRY_LIMIT


def _install_torch_stub(monkeypatch):
    """Provide a minimal torch stub so generation helpers can import."""

    class _Dist:
        def is_available(self):
            return False

        def is_initialized(self):
            return False

        def get_world_size(self):
            return 1

        def all_gather_object(self, output_list, input_obj):
            if output_list:
                output_list[0] = input_obj

        def broadcast_object_list(self, *_args, **_kwargs):
            return None

    class _Tensor(list):
        def __init__(self, data, *args, **kwargs):
            super().__init__(data)

        def float(self):
            return self

    torch_stub = SimpleNamespace(
        distributed=_Dist(),
        Tensor=_Tensor,
        float32="f32",
        tensor=lambda vals, dtype=None, device=None: _Tensor(vals),
        device=lambda val="cpu": SimpleNamespace(type=str(val)),
    )
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    return torch_stub


def _import_helpers(monkeypatch):
    _install_torch_stub(monkeypatch)
    return importlib.reload(
        importlib.import_module("maxent_grpo.training.generation.helpers")
    )


def test_append_completion_group_creates_meta(monkeypatch):
    gh = _import_helpers(monkeypatch)
    comps = [["a"], []]
    meta = None
    meta = gh._append_completion_group(comps, meta, 1, ["b", "c"], [1])
    assert comps[1] == ["b", "c"]
    assert meta is not None and meta[1][0] == 1


def test_seed_and_pending(monkeypatch):
    gh = _import_helpers(monkeypatch)
    comps, meta = gh._seed_generation_groups_impl(2, [["x"], []], [[None], [None, 2]])
    assert comps[0] == ["x"]
    pending = gh._pending_generation_indices(comps, expected_generations=2)
    assert pending == [0, 1]
    assert gh._determine_retry_limit(0, None) == _DEFAULT_RETRY_LIMIT


def test_retry_incomplete_prompts(monkeypatch):
    gh = _import_helpers(monkeypatch)

    def _gen(prompts, expected, missing=None):
        # return one completion per prompt on initial call; retries add one more
        return [
            [p] * (1 if missing is None else missing[i]) for i, p in enumerate(prompts)
        ], [
            [None] * (1 if missing is None else missing[i])
            for i, _ in enumerate(prompts)
        ]

    state = gh._AggregatedGenerationState([["p1"], []], [[None], []])
    updated = gh._retry_incomplete_prompts_impl(
        ["p1", "p2"], _gen, expected_generations=2, aggregated=state, max_retry_rounds=1
    )
    assert all(len(g) >= 1 for g in updated.completions)


def test_zero3_gather_factory(monkeypatch):
    gh = _import_helpers(monkeypatch)
    acc = SimpleNamespace(
        state=SimpleNamespace(deepspeed_plugin=SimpleNamespace(zero_stage=0))
    )
    factory = gh._zero3_gather_factory(acc)
    ctx = factory([1, 2])
    assert isinstance(ctx, nullcontext)

    class GP:
        def __init__(self, params):
            self.params = params

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    monkeypatch.setitem(
        sys.modules,
        "deepspeed",
        SimpleNamespace(zero=SimpleNamespace(GatheredParameters=GP)),
    )
    acc = SimpleNamespace(
        state=SimpleNamespace(deepspeed_plugin=SimpleNamespace(zero_stage=3))
    )
    factory = gh._zero3_gather_factory(acc)
    with factory([42]) as gathered:
        assert isinstance(gathered, GP)
        assert gathered.params == [42]


def test_is_peft_model_safe(monkeypatch):
    gh = _import_helpers(monkeypatch)
    monkeypatch.setitem(
        sys.modules,
        "accelerate.utils",
        SimpleNamespace(is_peft_model=lambda target: target == "ok"),
    )
    assert gh._is_peft_model_safe("ok") is True
    assert gh._is_peft_model_safe("nope") is False


def test_is_peft_model_safe_handles_noncallable(monkeypatch):
    gh = _import_helpers(monkeypatch)
    monkeypatch.setitem(
        sys.modules, "accelerate.utils", SimpleNamespace(is_peft_model="nope")
    )
    assert gh._is_peft_model_safe(object()) is False


def test_import_vllm_client_cls(monkeypatch):
    gh = _import_helpers(monkeypatch)

    class _Client:
        pass

    monkeypatch.setitem(
        sys.modules, "trl.extras.vllm_client", SimpleNamespace(VLLMClient=_Client)
    )
    assert gh._import_vllm_client_cls() is _Client


def test_tokenize_expanded_prompts_fallback_mask(monkeypatch):
    gh = _import_helpers(monkeypatch)
    dummy = SimpleNamespace(
        ctx=SimpleNamespace(
            tokenizer=SimpleNamespace(), max_prompt_len=None, device="cpu"
        )
    )
    inputs, lengths = gh.CompletionGenerator._tokenize_expanded_prompts(
        dummy, ["a", "abcd"]
    )
    mask = inputs["attention_mask"]
    # Fallback mask should allow the sum/detach/cpu/tolist chain.
    assert mask.sum().detach().cpu().tolist() == lengths
    assert inputs.to("cuda") is inputs


def test_generate_with_vllm_short_circuits_for_empty_prompts(monkeypatch):
    gh = _import_helpers(monkeypatch)
    dummy = SimpleNamespace(ctx=SimpleNamespace())
    grouped, meta = gh.CompletionGenerator._generate_with_vllm(dummy, [], 2, None)
    assert grouped == []
    assert meta is None


def test_generate_vllm_collective_scatter_on_non_main_rank(monkeypatch):
    gh = _import_helpers(monkeypatch)
    accel = SimpleNamespace(num_processes=2, is_main_process=False)
    captured = {}
    dummy = SimpleNamespace(
        ctx=SimpleNamespace(accelerator=accel),
        _prompt_char_limit=lambda: 8,
        _flatten_prompts_for_broadcast=lambda prompts, counts: (["p"], [0], counts),
        _scatter_vllm_payload=lambda flat, offsets, grouped, meta: captured.setdefault(
            "payload", (flat, offsets, grouped, meta)
        ),
    )
    result = gh.CompletionGenerator._generate_vllm_collective(dummy, ["p"], 1, None)
    assert captured["payload"][2] is None and captured["payload"][3] is None
    assert isinstance(result, tuple)
    assert len(result) >= 2
    grouped_res, _ = result[0], result[1]
    assert grouped_res is not None


def test_scatter_object_fallback_selects_process_index(monkeypatch):
    gh = _import_helpers(monkeypatch)
    acc = SimpleNamespace(num_processes=3, process_index=1, scatter_object=None)
    value = gh._scatter_object(acc, [10, 20, 30], src=0)
    assert value == 20
