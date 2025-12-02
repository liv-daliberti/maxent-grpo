"""Expanded coverage for training.generation.vllm_adapter edge cases."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, cast
from types import ModuleType, SimpleNamespace
import sys

import pytest

import maxent_grpo.training.generation.vllm_adapter as vllm_adapter
from maxent_grpo.generation.vllm import VLLMGenerationHelper

safe_generate = None
time = None


class _DelegateHelper:
    def __init__(self):
        self.calls: list[tuple[str, object]] = []
        self._vllm_client = None
        self._vllm_sync_ready = False
        self._last_vllm_synced_step = None
        self._fsdp_cls = None
        self._fallback_generate = None
        self._generate_with_vllm = None
        self._scatter_vllm_payload = None

    def maybe_sync_weights(self, *args, **kwargs):
        self.calls.append(("sync_call", args, kwargs))

    def _invoke_vllm_requests(self, prompts, request_count):
        return ([], None, 0.0)

    def set_safe_generate(self, fn):
        self._safe_generate = fn

    def set_time_provider(self, provider):
        self._time = provider

    def merge_vllm_results(self, state, grouped, grouped_meta, pending):
        self.calls.append(("merge", state, grouped, grouped_meta, pending))

    def set_fallback_generate(self, fn):
        self._fallback_generate = fn

    def backfill_missing(self, state, missing):
        if hasattr(self, "_backfill_missing"):
            return self._backfill_missing(state, missing)
        self.calls.append(("backfill_call", missing))

    def record_vllm_failure(self, state, missing):
        if hasattr(self, "_record_vllm_failure"):
            return self._record_vllm_failure(state, missing)
        self.calls.append(("failure_call", missing))

    def _sync_model_params_to_vllm(self, model):
        self.calls.append(("sync_model", model))

    def _push_param_to_vllm(self, name, param):
        self.calls.append(("push", name, param))

    def _reset_vllm_cache(self):
        self.calls.append(("reset", None))

    def _sync_fsdp_params(self, model):
        self.calls.append(("fsdp", model))

    def _sync_peft_params(self, model, gather_factory):
        self.calls.append(("peft", model, gather_factory))

    def _sync_standard_params(self, model, gather_factory):
        self.calls.append(("standard", model, gather_factory))


class _DelegateGen(vllm_adapter.VLLMGenerationMixin):
    def __init__(self, helper=None):
        helper = helper or _DelegateHelper()
        self.ctx = SimpleNamespace(
            accelerator=SimpleNamespace(
                is_main_process=True, num_processes=1, process_index=0
            ),
            vllm_sync_weights=True,
            vllm_url="http://vllm",
            use_vllm=False,
        )
        self._vllm_helper = helper

    def _generate_local(self, *args, **kwargs):
        return ("local", args, kwargs)

    def _prompt_char_limit(self) -> int:
        return 32


def test_delegate_wrappers_and_property_setters():
    helper = _DelegateHelper()
    gen = _DelegateGen(helper)

    gen._fsdp_cls = "fsdp"
    assert helper._fsdp_cls == "fsdp"

    gen._sync_model_params_to_vllm("m", gen.ctx.accelerator)
    gen._push_param_to_vllm("p", "param")
    gen._reset_vllm_cache()
    gen._sync_fsdp_params("fsdp_model")
    gen._sync_peft_params("peft_model", lambda _p: nullcontext())
    gen._sync_standard_params("std_model", lambda _p: nullcontext())

    calls = [c[0] for c in helper.calls]
    assert calls == [
        "sync_model",
        "push",
        "reset",
        "fsdp",
        "peft",
        "standard",
    ]


def test_maybe_sync_weights_typeerror_path():
    class _TypeErrorHelper(_DelegateHelper):
        def maybe_sync_weights(self, *args, **kwargs):
            if args or kwargs:
                raise TypeError("boom")
            self.calls.append(("fallback",))

    helper = _TypeErrorHelper()
    gen = _DelegateGen(helper)
    gen._maybe_sync_vllm_weights()
    assert ("fallback",) in helper.calls


def test_retry_incomplete_prompts_wrapper():
    def _generator(prompts, expected, counts):
        return [[p] * expected for p in prompts], None

    comps, meta = vllm_adapter.VLLMGenerationMixin._retry_incomplete_prompts(
        None,
        ["a", "b"],
        _generator,
        2,
        [["a"], []],
        None,
        max_retry_rounds=1,
    )
    assert comps[0][:2] == ["a", "a"]
    assert meta is None


def test_request_vllm_batch_coalesces_and_records_latency(monkeypatch):
    recorded = {}

    class _ReqGen(_DelegateGen):
        def __init__(self):
            super().__init__(_DelegateHelper())
            self._vllm_helper._coalesce_grouped_outputs = (
                lambda groups, _count, _req, meta=None: (groups, meta)
            )

        def _invoke_vllm_requests(self, prompts, request_count):
            return [[p.upper()] for p in prompts], [["m"]] * len(prompts), 1.5

        def _record_vllm_latency(self, latency_ms):
            recorded["latency"] = latency_ms

    gen = _ReqGen()
    grouped, meta = gen._request_vllm_batch(["p"], 1)
    assert grouped == [["P"]]
    assert meta == [["m"]]
    assert recorded["latency"] == 1.5


def test_invoke_vllm_requests_uses_module_defaults(monkeypatch):
    sentinel_safe = object()
    sentinel_time = object()
    this_mod = sys.modules[__name__]
    monkeypatch.setattr(this_mod, "safe_generate", sentinel_safe, raising=False)
    monkeypatch.setattr(this_mod, "time", sentinel_time, raising=False)

    class _InvokeGen(_DelegateGen):
        def __init__(self):
            super().__init__(_DelegateHelper())

        def _invoke_vllm_requests(self, prompts, request_count):
            return super()._invoke_vllm_requests(prompts, request_count)

    gen = _InvokeGen()
    result = gen._invoke_vllm_requests(["p"], 1)
    assert gen._vllm_helper._safe_generate is sentinel_safe
    assert gen._vllm_helper._time is sentinel_time
    assert result is None or isinstance(result, tuple)


def test_backfill_and_failure_delegation():
    helper = _DelegateHelper()
    helper._backfill_missing = lambda state, missing: helper.calls.append(
        ("backfill", missing)
    )
    helper._record_vllm_failure = lambda state, missing: helper.calls.append(
        ("failure", missing)
    )
    gen = _DelegateGen(helper)
    gen._backfill_missing("state", [1, 2])
    gen._record_vllm_failure("state", [3])
    assert ("backfill", [1, 2]) in helper.calls
    assert ("failure", [3]) in helper.calls
    assert getattr(
        helper._fallback_generate, "__func__", helper._fallback_generate
    ) is (gen._generate_local.__func__)


def test_run_vllm_rounds_rewires_helper(monkeypatch):
    sentinel_time = object()
    helpers_mod = ModuleType("maxent_grpo.training.generation.helpers")
    helpers_mod.time = sentinel_time  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, helpers_mod.__name__, helpers_mod)
    monkeypatch.setattr(sys.modules[__name__], "time", sentinel_time, raising=False)

    class _RunHelper(_DelegateHelper):
        _execute_vllm_request = VLLMGenerationHelper._execute_vllm_request
        _request_vllm_batch = VLLMGenerationHelper._request_vllm_batch

        def _run_vllm_rounds(self, state):
            self.calls.append(("run", state))

    class _RunGen(_DelegateGen):
        def __init__(self):
            super().__init__(_RunHelper())

        def _execute_vllm_request(self, state, pending):
            return ("exec", state, pending)

        def _request_vllm_batch(self, prompts, n):
            return ("batch", prompts, n)

    gen = _RunGen()
    state: Any = {}
    gen._run_vllm_rounds(state)
    helper: _RunHelper = gen._vllm_helper  # type: ignore[assignment]
    assert ("run", state) in helper.calls
    assert helper._time is sentinel_time
    assert callable(helper._execute_vllm_request)
    assert callable(helper._request_vllm_batch)


def test_broadcast_vllm_payload_fills_missing_groups():
    helper = _DelegateHelper()
    gen = _DelegateGen(helper)
    result_grouped, result_meta = gen._broadcast_vllm_payload(
        ["p1", "p2"], [None, None]
    )
    assert result_grouped == [[], []]
    assert result_meta is None


def test_generate_vllm_collective_main_and_worker_paths():
    helper = _DelegateHelper()
    helper._flatten_prompts_for_broadcast = lambda prompts, counts=None: (
        prompts,
        [0, len(prompts)],
        counts,
    )  # type: ignore[attr-defined]
    helper._generate_with_vllm = lambda prompts, num_samples, counts: (
        [["a"] for _ in prompts],
        None,
    )  # type: ignore[attr-defined]
    helper._scatter_vllm_payload = lambda flat, offsets, grouped, meta: (
        grouped or [],
        meta,
    )  # type: ignore[attr-defined]

    class _CollectiveGen(_DelegateGen):
        def __init__(self, accel):
            super().__init__(helper)
            self.ctx.accelerator = accel

        def _generate_with_vllm(self, prompts, num_samples, counts=None):
            return helper._generate_with_vllm(prompts, num_samples, counts)

        def _scatter_vllm_payload(self, flat, offsets, grouped, meta):
            return helper._scatter_vllm_payload(flat, offsets, grouped, meta)

    main_accel = SimpleNamespace(is_main_process=True, num_processes=2)
    gen_main = _CollectiveGen(main_accel)
    grouped, meta = gen_main._generate_vllm_collective(["p"], 1, [1])
    assert grouped == [["a"]]
    assert meta is None

    worker_accel = SimpleNamespace(is_main_process=False, num_processes=2)
    gen_worker = _CollectiveGen(worker_accel)
    grouped_w, meta_w = gen_worker._generate_vllm_collective(["p"], 1, [1])
    assert grouped_w in ([], [["a"]])
    assert meta_w is None


def test_generate_vllm_collective_handles_none_scatter(monkeypatch):
    helper = _DelegateHelper()
    helper._flatten_prompts_for_broadcast = lambda prompts, counts=None: (
        prompts,
        [0, len(prompts)],
        counts,
    )  # type: ignore[attr-defined]
    helper._generate_with_vllm = lambda *a, **k: (None, None)  # type: ignore[attr-defined]
    helper._scatter_vllm_payload = lambda *a, **k: (None, None)  # type: ignore[attr-defined]

    class _CollectiveGen(_DelegateGen):
        def __init__(self, accel):
            super().__init__(helper)
            self.ctx.accelerator = accel

        def _generate_with_vllm(self, prompts, num_samples, counts=None):
            return helper._generate_with_vllm(prompts, num_samples, counts)

        def _scatter_vllm_payload(self, flat, offsets, grouped, meta):
            return helper._scatter_vllm_payload(flat, offsets, grouped, meta)

    worker_accel = SimpleNamespace(is_main_process=False, num_processes=2)
    gen_worker = _CollectiveGen(worker_accel)
    grouped, meta = gen_worker._generate_vllm_collective(["p"], 1, [1])
    assert grouped == [[""]] * 0 or grouped == [[]]
    assert meta is None or meta == [[]]


def test_generate_vllm_collective_handles_short_scatter_tuple():
    helper = _DelegateHelper()
    helper._flatten_prompts_for_broadcast = lambda prompts, counts=None: (
        prompts,
        [0, len(prompts)],
        counts,
    )
    helper._generate_with_vllm = lambda *a, **k: (["x"], None)
    helper._scatter_vllm_payload = lambda *a, **k: ("only-grouped",)

    class _CollectiveGen(_DelegateGen):
        def __init__(self, accel):
            super().__init__(helper)
            self.ctx.accelerator = accel

        def _generate_with_vllm(self, prompts, num_samples, counts=None):
            return helper._generate_with_vllm(prompts, num_samples, counts)

        def _scatter_vllm_payload(self, flat, offsets, grouped, meta):
            return helper._scatter_vllm_payload(flat, offsets, grouped, meta)

    worker_accel = SimpleNamespace(is_main_process=False, num_processes=2)
    gen_worker = _CollectiveGen(worker_accel)
    grouped, meta = gen_worker._generate_vllm_collective(["p"], 1, [1])
    assert grouped == []
    assert meta is None


def test_generate_vllm_collective_non_tuple_scatter():
    helper = _DelegateHelper()
    helper._flatten_prompts_for_broadcast = lambda prompts, counts=None: (
        prompts,
        [0, len(prompts)],
        counts,
    )
    helper._generate_with_vllm = lambda *a, **k: (["a"], None)
    helper._scatter_vllm_payload = lambda *a, **k: [["sentinel"]]

    class _CollectiveGen(_DelegateGen):
        def __init__(self, accel):
            super().__init__(helper)
            self.ctx.accelerator = accel

        def _generate_with_vllm(self, prompts, num_samples, counts=None):
            return helper._generate_with_vllm(prompts, num_samples, counts)

        def _scatter_vllm_payload(self, flat, offsets, grouped, meta):
            return helper._scatter_vllm_payload(flat, offsets, grouped, meta)

    worker_accel = SimpleNamespace(is_main_process=False, num_processes=2)
    gen_worker = _CollectiveGen(worker_accel)
    grouped, meta = gen_worker._generate_vllm_collective(["p"], 1, [1])
    assert grouped == [["sentinel"]]
    assert meta is None


def test_run_vllm_rounds_sets_legacy_hooks(monkeypatch):
    class _LegacyHelper(_DelegateHelper):
        def __init__(self):
            super().__init__()
            self._execute_vllm_request = (
                VLLMGenerationHelper._execute_vllm_request.__get__(self, type(self))
            )
            self._request_vllm_batch = VLLMGenerationHelper._request_vllm_batch.__get__(
                self, type(self)
            )

        def _run_vllm_rounds(self, state):
            self.calls.append(("run_rounds", state))

    helper = _LegacyHelper()
    gen = _DelegateGen(helper)
    gen.ctx.accelerator = SimpleNamespace(is_main_process=True, num_processes=1)
    marker_state = object()
    gen._run_vllm_rounds(marker_state)
    assert getattr(
        helper._execute_vllm_request, "__func__", helper._execute_vllm_request
    ) is getattr(gen._execute_vllm_request, "__func__", gen._execute_vllm_request)
    assert getattr(
        helper._request_vllm_batch, "__func__", helper._request_vllm_batch
    ) is getattr(gen._request_vllm_batch, "__func__", gen._request_vllm_batch)
    assert getattr(
        helper._fallback_generate, "__func__", helper._fallback_generate
    ) is getattr(gen._generate_local, "__func__", gen._generate_local)
    assert ("run_rounds", marker_state) in helper.calls


def test_generate_with_vllm_returns_empty_when_helper_missing_generate():
    helper = _DelegateHelper()
    gen = _DelegateGen(helper)
    gen.ctx.accelerator = SimpleNamespace()
    grouped, meta = gen._generate_with_vllm(["p1"], 1, None)
    assert grouped == [] and meta is None


def test_generate_vllm_collective_fills_none_grouped():
    helper = _DelegateHelper()
    helper._flatten_prompts_for_broadcast = lambda prompts, counts=None: (
        prompts,
        [0, len(prompts)],
        counts,
    )
    helper._generate_with_vllm = lambda *a, **k: (None, None)
    helper._scatter_vllm_payload = lambda *a, **k: (None, None)

    class _CollectiveGen(_DelegateGen):
        def __init__(self):
            super().__init__(helper)
            self.ctx.accelerator = SimpleNamespace(
                is_main_process=True, num_processes=2
            )

        def _generate_with_vllm(self, prompts, num_samples, counts=None):
            return helper._generate_with_vllm(prompts, num_samples, counts)

        def _scatter_vllm_payload(self, flat, offsets, grouped, meta):
            return helper._scatter_vllm_payload(flat, offsets, grouped, meta)

    gen = _CollectiveGen()
    grouped, meta = gen._generate_vllm_collective(["p"], 1, [1])
    assert grouped == [[]]
    assert meta is None


def test_scatter_object_index_error_returns_none(monkeypatch):
    accel = SimpleNamespace(num_processes=3, process_index=5)
    result = vllm_adapter._scatter_object(accel, ["a", "b"], src=0)
    assert result is None


def test_generate_routes_and_validates_counts():
    gen = _DelegateGen()
    empty = gen.generate([], 1)
    assert empty == ([], None)
    with pytest.raises(ValueError):
        gen.generate(["p"], 1, [1, 2])

    gen.ctx.use_vllm = True
    gen._generate_vllm_collective = lambda *a, **k: ("via_vllm", None)
    assert gen.generate(["p"], 1) == ("via_vllm", None)


def test_gather_and_broadcast_helpers(monkeypatch):
    class _Dist:
        def __init__(self):
            self._payload = None

        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_world_size(self):
            return 2

        def all_gather_object(self, gathered, value):
            gathered[0] = value
            gathered[1] = ["remote"]

        def broadcast_object_list(self, payload, src=0):
            self._payload = list(payload)

    dist_stub = _Dist()
    monkeypatch.setattr(vllm_adapter, "dist", dist_stub)
    accel = SimpleNamespace()
    gathered = vllm_adapter._gather_object_list(accel, ["a"])
    assert gathered == [["a"], ["remote"]]

    payload = [None]
    vllm_adapter._broadcast_object_list(accel, payload, src=0)
    assert dist_stub._payload == payload


def test_broadcast_object_list_prefers_accelerator(monkeypatch):
    recorded = {}

    def _broadcast(payload, src=0):
        recorded["payload"] = list(payload)
        recorded["src"] = src

    accel = SimpleNamespace(broadcast_object_list=_broadcast)
    vllm_adapter._broadcast_object_list(accel, ["x"], src=2)
    assert recorded == {"payload": ["x"], "src": 2}


def test_scatter_object_variants(monkeypatch):
    # num_processes <= 1 returns first/None
    accel_single = SimpleNamespace(
        num_processes=1, process_index=0, scatter_object=None
    )
    assert vllm_adapter._scatter_object(accel_single, None, src=0) is None
    assert vllm_adapter._scatter_object(accel_single, ["only"], src=0) == "only"

    # dist scatter path
    class _Dist:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def scatter_object_list(self, output, input_list, src=0):
            output[0] = "dist"

    monkeypatch.setattr(vllm_adapter, "dist", _Dist())
    accel_dist = SimpleNamespace(num_processes=2, process_index=0, scatter_object=None)
    assert vllm_adapter._scatter_object(accel_dist, ["a", "b"], src=0) == "dist"

    # fallback to index
    monkeypatch.setattr(vllm_adapter, "dist", None)
    accel_idx = SimpleNamespace(num_processes=3, process_index=1, scatter_object=None)
    assert vllm_adapter._scatter_object(accel_idx, ["x", "y", "z"], src=0) == "y"

    # graceful failure on out-of-range
    accel_oob = SimpleNamespace(num_processes=3, process_index=5, scatter_object=None)
    assert vllm_adapter._scatter_object(accel_oob, ["only"], src=0) is None

    # missing process_index attribute
    accel_no_index = SimpleNamespace(num_processes=2, scatter_object=None)
    assert vllm_adapter._scatter_object(accel_no_index, ["a", "b"], src=0) is None

    # TypeError from non-subscriptable input_list
    accel_idx2 = SimpleNamespace(num_processes=2, process_index=0, scatter_object=None)
    assert vllm_adapter._scatter_object(accel_idx2, 123, src=0) is None
