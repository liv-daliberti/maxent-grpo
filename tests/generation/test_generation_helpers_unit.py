"""
Additional unit tests for :mod:`maxent_grpo.training.rollout.helpers`.
"""

from __future__ import annotations

import importlib
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest

import maxent_grpo.training.rollout.helpers as helpers
from maxent_grpo.training.run_helpers import (
    GenerationPenaltyConfig,
    VLLMClientConfig,
)


def _make_generator(**overrides) -> helpers.CompletionGenerator:
    """Create a ``CompletionGenerator`` with lightweight stubs."""
    vllm_cfg = overrides.get(
        "vllm_cfg",
        VLLMClientConfig(
            url="http://vllm",
            rounds_cfg=1,
            retry_sleep=0.1,
            backfill_local=False,
            request_logprobs=False,
        ),
    )
    penalty = overrides.get("penalty", GenerationPenaltyConfig())
    ctx = helpers.GenerationContext(
        max_prompt_len=overrides.get("max_prompt_len", 10),
        max_completion_len=overrides.get("max_completion_len", 8),
        gen_temperature=overrides.get("gen_temperature", 1.0),
        gen_top_p=overrides.get("gen_top_p", 0.9),
        use_vllm=overrides.get("use_vllm", False),
        vllm=vllm_cfg,
        accelerator=overrides.get("accelerator", SimpleNamespace()),
        model=overrides.get("model", SimpleNamespace()),
        tokenizer=overrides.get("tokenizer", SimpleNamespace()),
        generation_stats=overrides.get("generation_stats", {}),
        device=overrides.get("device", "cpu"),
        penalty=penalty,
    )
    return helpers.CompletionGenerator(ctx)


def test_optional_import_handles_missing_and_present(monkeypatch):
    sentinel = object()

    def _fake_import(name):
        if name == "dummy_mod":
            return sentinel
        raise ImportError("missing")

    monkeypatch.setattr(importlib, "import_module", _fake_import)
    assert helpers._optional_import("dummy_mod") is sentinel
    assert helpers._optional_import("missing_mod") is None


def test_zero3_gather_factory_returns_gather_for_stage3(monkeypatch):
    class _Gather:
        def __init__(self, params):
            self.params = list(params)
            self.entered = False
            self.exited = False

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, exc_type, exc, tb):
            self.exited = True
            return False

    zero_mod = SimpleNamespace(GatheredParameters=_Gather)
    deepspeed_mod = ModuleType("deepspeed")
    deepspeed_mod.__spec__ = SimpleNamespace()
    deepspeed_mod.zero = zero_mod
    monkeypatch.setitem(sys.modules, "deepspeed", deepspeed_mod)
    accelerator = SimpleNamespace(
        state=SimpleNamespace(deepspeed_plugin=SimpleNamespace(zero_stage=3))
    )

    factory = helpers._zero3_gather_factory(accelerator)
    ctx = factory([1, 2])
    assert isinstance(ctx, _Gather)
    with ctx as handle:
        assert handle.entered is True
        assert handle.params == [1, 2]
    assert ctx.exited is True


def test_zero3_gather_factory_uses_nullcontext_when_not_stage3():
    accelerator = SimpleNamespace(
        state=SimpleNamespace(deepspeed_plugin=SimpleNamespace(zero_stage=0))
    )
    factory = helpers._zero3_gather_factory(accelerator)
    with factory(["ignored"]) as gathered:
        assert gathered is None


def test_is_peft_model_safe_handles_absent_and_true(monkeypatch):
    monkeypatch.delitem(sys.modules, "accelerate.utils", raising=False)
    assert helpers._is_peft_model_safe(object()) is False

    utils_mod = ModuleType("accelerate.utils")
    utils_mod.__spec__ = SimpleNamespace()
    utils_mod.is_peft_model = lambda target: target == "peft-target"
    monkeypatch.setitem(sys.modules, "accelerate.utils", utils_mod)
    assert helpers._is_peft_model_safe("peft-target") is True

    utils_mod.is_peft_model = lambda *_a, **_k: (_ for _ in ()).throw(ValueError())
    assert helpers._is_peft_model_safe("peft-target") is False


def test_build_local_prompt_requests_filters_zero_or_negative():
    generator = _make_generator()
    prompts = ["p0", "p1", "p2"]
    expanded, indices = generator._build_local_prompt_requests(prompts, [2, 0, -3])
    assert expanded == ["p0", "p0"]
    assert indices == [0, 0]


def test_prompt_char_limit_prefers_approx_over_env(monkeypatch):
    monkeypatch.setattr(helpers, "PROMPT_CHAR_LIMIT", 50)
    small_gen = _make_generator(max_prompt_len=0)
    large_gen = _make_generator(max_prompt_len=30)
    assert small_gen._prompt_char_limit() == 50
    assert large_gen._prompt_char_limit() == 120


def test_decode_sequences_strips_prompt_prefixes():
    tokenizer = SimpleNamespace(
        decode=lambda ids, skip_special_tokens=True: "|".join(str(i) for i in ids)
    )
    sequences = [[101, 102, 1, 2], [201, 3]]
    decoded = helpers.CompletionGenerator._decode_sequences(
        sequences,
        prompt_lengths=[2, 1],
        tokenizer=tokenizer,
    )
    assert decoded == ["1|2", "3"]


def test_record_vllm_latency_accumulates_totals():
    stats = {}
    generator = _make_generator(generation_stats=stats)
    generator._record_vllm_latency(10.0)
    generator._record_vllm_latency(5.5)
    assert stats["vllm_last_latency_ms"] == pytest.approx(5.5)
    assert stats["vllm_latency_total_ms"] == pytest.approx(15.5)
    assert stats["vllm_latency_calls"] == 2


def test_build_vllm_request_kwargs_respects_penalty_overrides():
    penalty = GenerationPenaltyConfig(
        gen_top_k=7,
        gen_best_of=3,
        gen_frequency_penalty=0.4,
        gen_presence_penalty=0.2,
        gen_stop_sequences=["override-stop"],
    )
    vllm_cfg = VLLMClientConfig(
        url="http://vllm",
        rounds_cfg=1,
        retry_sleep=0.1,
        backfill_local=False,
        request_logprobs=False,
        top_k=None,
        best_of=None,
        stop_sequences=["fallback"],
        frequency_penalty=0.1,
        presence_penalty=0.1,
    )
    generator = _make_generator(penalty=penalty, vllm_cfg=vllm_cfg, use_vllm=True)
    kwargs = generator._build_vllm_request_kwargs(["p"], request_count=2)
    assert kwargs["top_k"] == 7
    assert kwargs["best_of"] == 3
    assert kwargs["stop"] == ["override-stop"]
    assert kwargs["frequency_penalty"] == pytest.approx(0.4)
    assert kwargs["presence_penalty"] == pytest.approx(0.2)


def test_gather_object_list_returns_single_process_fallback(monkeypatch):
    accelerator = SimpleNamespace()
    dist_stub = SimpleNamespace(
        is_available=lambda: False, is_initialized=lambda: False
    )
    monkeypatch.setattr(helpers, "dist", dist_stub)
    gathered = helpers._gather_object_list(accelerator, ["only"])
    assert gathered == [["only"]]


def test_import_vllm_client_cls_handles_missing_and_present(monkeypatch):
    monkeypatch.delitem(sys.modules, "trl.extras.vllm_client", raising=False)
    assert helpers._import_vllm_client_cls() is None
    client_cls = type("VLLMClient", (), {})
    mod = ModuleType("trl.extras.vllm_client")
    mod.__spec__ = SimpleNamespace()
    mod.VLLMClient = client_cls
    monkeypatch.setitem(sys.modules, "trl.extras.vllm_client", mod)
    assert helpers._import_vllm_client_cls() is client_cls


def test_vllm_generation_state_pending_and_trim(monkeypatch):
    state = helpers._VLLMGenerationState(
        prompts=["p1", "p2"],
        target_counts=[1, 2],
        requested_n=1,
        round_limit=2,
        track_logprobs=True,
    )
    assert state.pending_indices() == [0, 1]
    state.aggregated[0].append("a")
    state.aggregated[1].append("b")
    assert state.remaining_counts([0, 1]) == [0, 1]
    state.aggregated_meta[1].append("meta")  # type: ignore[index]
    trimmed, meta = state.trim()
    assert trimmed == [["a"], ["b"]]
    assert meta == [[], ["meta"]]
    state.drop_meta()
    assert state.aggregated_meta is None


def test_resolve_local_counts_length_mismatch():
    generator = _make_generator()
    with pytest.raises(ValueError):
        generator._resolve_local_counts(["p0"], 2, [1, 2])


def test_merge_vllm_results_trims_overflow_and_meta():
    stats = {}
    generator = _make_generator(generation_stats=stats)
    state = helpers._VLLMGenerationState(
        prompts=["p0"],
        target_counts=[1],
        requested_n=2,
        round_limit=1,
        track_logprobs=True,
    )
    grouped = [["c1", "c2"]]
    grouped_meta = [[None, None]]
    generator._merge_vllm_results(state, grouped, grouped_meta, [0])
    assert state.aggregated[0] == ["c1"]
    assert state.aggregated_meta == [[None]]
    assert stats["vllm_excess_prompts"] == 1
    assert stats["vllm_excess_completions"] == 1


def test_coalesce_grouped_outputs_regroups_microgroups():
    groups = [["a"], ["b"], ["c"], ["d"]]
    meta = [["m1"], ["m2"], ["m3"], ["m4"]]
    regrouped, regrouped_meta = helpers.CompletionGenerator._coalesce_grouped_outputs(
        groups, prompt_count=2, requested_n=2, meta=meta
    )
    assert regrouped == [["a", "b"], ["c", "d"]]
    assert regrouped_meta == [["m1", "m2"], ["m3", "m4"]]


def test_coalesce_grouped_outputs_returns_none_on_mismatch():
    groups = [["a"], ["b"], ["c"]]
    meta = [[None], [None], [None]]
    regrouped, regrouped_meta = helpers.CompletionGenerator._coalesce_grouped_outputs(
        groups, prompt_count=2, requested_n=1, meta=meta
    )
    assert regrouped == groups
    assert regrouped_meta is None


def test_prepare_vllm_targets_dedup_enabled(monkeypatch):
    generator = _make_generator()
    monkeypatch.setenv("MAXENT_VLLM_DEDUP", "1")
    prompts = ["p", "p", "q"]
    counts = [1, -2, 3]
    unique_prompts, target_counts, mapping = generator._prepare_vllm_targets(
        prompts, num_samples=5, per_prompt_counts=counts
    )
    assert unique_prompts == ["p", "q"]
    assert target_counts == [1, 3]
    assert mapping == [0, 0, 1]


def test_execute_vllm_request_groups_by_need(monkeypatch):
    generator = _make_generator()
    state = helpers._VLLMGenerationState(
        prompts=["a", "b"],
        target_counts=[2, 1],
        requested_n=2,
        round_limit=1,
        track_logprobs=False,
    )
    calls = []

    def _fake_request(prompts, need):
        calls.append((tuple(prompts), need))
        grouped = [[f"{p}-{idx}" for idx in range(need)] for p in prompts]
        return grouped, None

    monkeypatch.setattr(generator._vllm_helper, "_request_vllm_batch", _fake_request)
    success = generator._execute_vllm_request(state, [0, 1])
    assert success is True
    assert calls[0] == (("a",), 2)
    assert calls[1] == (("b",), 1)
    assert len(state.aggregated[0]) == 2
    assert len(state.aggregated[1]) == 1


def test_run_vllm_rounds_completes_without_retry(monkeypatch):
    stats = {"vllm_retry_rounds": 0}
    generator = _make_generator(use_vllm=True, generation_stats=stats)
    state = helpers._VLLMGenerationState(
        prompts=["p"],
        target_counts=[1],
        requested_n=1,
        round_limit=2,
        track_logprobs=False,
    )

    def _execute(state_obj, pending):
        state_obj.aggregated[0] = ["done"]
        return True

    monkeypatch.setattr(generator._vllm_helper, "_execute_vllm_request", _execute)
    generator._run_vllm_rounds(state)
    assert state.pending_indices() == []
    assert stats["vllm_retry_rounds"] == 0


def test_generate_handles_empty_and_mismatch():
    generator = _make_generator()
    assert generator.generate([], 2) == ([], None)
    with pytest.raises(ValueError):
        generator.generate(["p0"], 2, [1, 2])


def test_generate_vllm_collective_with_single_rank(monkeypatch):
    class _Accel:
        def __init__(self):
            self.is_main_process = True
            self.num_processes = 1
            self.process_index = 0

        def gather_object(self, value):
            return [value]

    accel = _Accel()
    generator = _make_generator(use_vllm=True, accelerator=accel)
    received = {}

    def _fake_generate(prompts, num_samples, flat_counts, **_kwargs):
        received["args"] = (list(prompts), num_samples, flat_counts)
        return [["a"], ["b"]], None

    monkeypatch.setattr(generator, "_generate_with_vllm", _fake_generate)
    grouped, meta = generator._generate_vllm_collective(["x", "y"], num_samples=1)
    assert grouped == [["a"], ["b"]]
    assert meta is None
    assert received["args"][0] == ["x", "y"]
    assert received["args"][2] is None


def test_scatter_object_uses_dist_when_available(monkeypatch):
    output_slots = []

    class _Accel:
        num_processes = 2
        process_index = 0

    def _scatter(output, input_list, src=0):
        output_slots.append((list(output), input_list, src))
        output[0] = input_list[0] if input_list else "from-src"

    dist_stub = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        scatter_object_list=_scatter,
    )
    monkeypatch.setattr(helpers, "dist", dist_stub)
    accel = _Accel()
    result_src = helpers._scatter_object(accel, [["payload"], ["other"]], src=0)
    assert result_src == ["payload"]

    accel.process_index = 1
    result_other = helpers._scatter_object(accel, [["payload"], ["other"]], src=0)
    assert result_other == "from-src"


def test_generation_context_accessors_and_dict():
    penalty = GenerationPenaltyConfig(gen_top_k=5, gen_best_of=2)
    ctx = helpers.GenerationContext(
        max_prompt_len=4,
        max_completion_len=8,
        gen_temperature=1.0,
        gen_top_p=0.9,
        use_vllm=True,
        vllm=VLLMClientConfig(
            url="http://host/generate",
            rounds_cfg=2,
            retry_sleep=0.1,
            backfill_local=False,
            request_logprobs=False,
        ),
        accelerator=SimpleNamespace(),
        model=SimpleNamespace(),
        tokenizer=SimpleNamespace(),
        generation_stats={},
        device="cpu",
        penalty=penalty,
    )
    ctx.gen_top_k = 7
    ctx.gen_best_of = 3
    ctx.gen_stop_sequences = ["stop"]
    info = ctx.as_dict()
    assert info["max_prompt_len"] == 4
    assert info["vllm_url"].startswith("http://host")
    assert ctx.gen_stop_sequences == ["stop"]
    assert ctx.gen_best_of == 3


def test_vllm_base_url_and_client(monkeypatch):
    generator = _make_generator(use_vllm=True)
    assert generator._vllm_base_url("http://h:8/generate") == "http://h:8"
    assert generator._vllm_base_url("foo/generate") == "foo"
    assert generator._vllm_base_url("http://h:8") == "http://h:8"

    ctx = generator.ctx
    ctx.vllm.sync_weights = True
    ctx.accelerator = SimpleNamespace(is_main_process=False)
    assert generator._ensure_vllm_client() is False

    ctx.accelerator = SimpleNamespace(is_main_process=True)
    monkeypatch.setattr(helpers, "_import_vllm_client_cls", lambda: None)
    assert generator._ensure_vllm_client() is False

    calls = {}

    class _Client:
        def __init__(self, base_url):
            calls["url"] = base_url

        def init_communicator(self):
            calls["init"] = True

    monkeypatch.setattr(helpers, "_import_vllm_client_cls", lambda: _Client)
    assert generator._ensure_vllm_client() is True
    assert calls["url"] == "http://vllm"
    assert calls["init"] is True


def test_maybe_sync_vllm_weights_respects_step(monkeypatch):
    generator = _make_generator(use_vllm=True, generation_stats={"current_step": 1})
    generator.ctx.vllm.sync_weights = True
    generator.ctx.accelerator = SimpleNamespace(
        is_main_process=True, unwrap_model=lambda m: m, wait_for_everyone=lambda: None
    )
    sync_calls = {"count": 0}
    monkeypatch.setattr(generator, "_ensure_vllm_client", lambda: True)
    monkeypatch.setattr(
        generator,
        "_sync_model_params_to_vllm",
        lambda *_: sync_calls.__setitem__("count", sync_calls["count"] + 1),
    )
    generator._maybe_sync_vllm_weights()
    generator._maybe_sync_vllm_weights()  # same step should no-op
    generator.ctx.generation_stats["current_step"] = 2
    generator._maybe_sync_vllm_weights()
    assert sync_calls["count"] == 2


def test_sync_standard_and_peft_params(monkeypatch):
    generator = _make_generator(use_vllm=True)
    pushed = []
    generator._vllm_client = object()
    monkeypatch.setattr(
        generator._vllm_helper,
        "_push_param_to_vllm",
        lambda name, param: pushed.append((name, getattr(param, "data", param))),
    )

    class _Param:
        def __init__(self, name):
            self.data = f"data-{name}"

    class _ModelStd:
        def parameters(self):
            return [_Param("a")]

        def named_parameters(self):
            return [("w1", _Param("w1"))]

    generator._sync_standard_params(_ModelStd(), lambda params: nullcontext())
    assert ("w1", "data-w1") in pushed

    class _PEFT:
        def __init__(self):
            self.merge_called = False
            self.unmerge_called = False
            self.prefix = "skip"

        def parameters(self):
            return [_Param("p")]

        def named_parameters(self):
            return [
                ("base_model.model.layer", _Param("layer")),
                ("skip.adapter", _Param("adapter")),
                ("modules_to_save.default.x", _Param("x")),
                ("original_module.y", _Param("y")),
            ]

        def merge_adapter(self):
            self.merge_called = True

        def unmerge_adapter(self):
            self.unmerge_called = True

    pushed.clear()
    peft = _PEFT()
    generator._sync_peft_params(peft, lambda params: nullcontext())
    assert peft.merge_called and peft.unmerge_called
    assert ("layer", "data-layer") in pushed
    assert ("x", "data-x") in pushed
    assert all("original_module" not in name for name, _ in pushed)


def test_sync_fsdp_params_pushes_once(monkeypatch):
    generator = _make_generator(use_vllm=True)
    pushed = []
    monkeypatch.setattr(
        generator._vllm_helper,
        "_push_param_to_vllm",
        lambda name, param: pushed.append(name),
    )

    class _Param:
        def __init__(self, name):
            self.data = name

    class DummyFSDP:
        def __init__(self, name):
            self._name = name

        def named_children(self):
            return []

        def named_parameters(self):
            return [(f"{self._name}_p", _Param(f"{self._name}_p"))]

        @staticmethod
        def summon_full_params(module, recurse=False, writeback=False):
            class _Ctx:
                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, exc_type, exc, tb):
                    return False

            return _Ctx()

    fsdp_model = DummyFSDP("root")
    generator._vllm_helper._fsdp_cls = DummyFSDP
    generator._sync_model_params_to_vllm(fsdp_model, SimpleNamespace())
    assert pushed == ["root_p"]


def test_tokenize_and_run_local(monkeypatch):
    class _Tensor(list):
        def __init__(self, data):
            super().__init__(data)

        def sum(self, dim):
            assert dim == 1
            return _Tensor([len(row) for row in self])

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return list(self)

        def to(self, device):
            self.device = device
            return self

    class _Encoder(dict):
        def __init__(self, prompts):
            super().__init__(input_ids=_Tensor([[1] * len(p) for p in prompts]))
            self["attention_mask"] = _Tensor([[1] * len(p) for p in prompts])

        def to(self, device):
            self.device = device
            return self

    tok_calls = {}

    def _tokenizer(prompts, **kwargs):
        tok_calls["prompts"] = list(prompts)
        return _Encoder(prompts)

    class _Model:
        def generate(self, **kwargs):
            return [[0, 1], [0, 2]]

    decoded_out = ["a", "b"]
    generator = _make_generator(
        tokenizer=_tokenizer,
        model=_Model(),
        accelerator=SimpleNamespace(unwrap_model=lambda m: m),
    )
    monkeypatch.setattr(
        generator, "_decode_sequences", lambda seqs, lens, tok: decoded_out
    )
    enc_inputs, lengths = generator._tokenize_expanded_prompts(["aa", "bbb"])
    assert lengths == [2, 3]
    out = generator._run_local_model(enc_inputs, lengths)
    assert out == decoded_out


def test_generate_local_and_summarize(monkeypatch):
    generator = _make_generator()
    grouped, meta = generator._generate_local([], 2)
    assert grouped == [] and meta is None
    summary = generator._summarize_grouped([["abc"], [], ["xyz"]], limit=2)
    assert "0:len=1" in summary and "1:len=0" in summary


def test_invoke_vllm_requests_recurses(monkeypatch):
    calls = {"count": 0}

    def _safe_generate(**kwargs):
        if len(kwargs["prompts"]) > 1:
            calls["count"] += 1
            raise RuntimeError("split me")
        return [["ok"]], None, 1.0

    monkeypatch.setattr(helpers, "safe_generate", _safe_generate)
    generator = _make_generator(use_vllm=True)
    result = generator._invoke_vllm_requests(["p1", "p2"], 1)
    assert result is not None
    groups, meta, latency = result
    assert groups == [["ok"], ["ok"]]
    assert meta is None
    assert latency == pytest.approx(2.0)


def test_backfill_missing_and_record_failure(monkeypatch):
    stats = {"vllm_backfilled_prompts": 0}
    generator = _make_generator(use_vllm=True, generation_stats=stats)
    generator.ctx.vllm.backfill_local = True
    state = helpers._VLLMGenerationState(
        prompts=["p0", "p1"],
        target_counts=[2, 1],
        requested_n=1,
        round_limit=1,
        track_logprobs=True,
    )
    state.aggregated = [["a"], []]
    state.aggregated_meta = [["m"], []]
    monkeypatch.setattr(
        generator, "_generate_local", lambda prompts, n, counts: ([["b"], ["c"]], None)
    )
    generator._vllm_helper._fallback_generate = generator._generate_local
    generator._backfill_missing(state, [1])
    assert state.aggregated[1] == ["b"]
    assert state.aggregated_meta is None
    assert stats["vllm_backfilled_prompts"] == 1

    stats_fail = {"vllm_failed_prompts": 0}
    generator.ctx.vllm.backfill_local = False
    generator.ctx.generation_stats = stats_fail
    generator._record_vllm_failure(state, [0])
    assert stats_fail["vllm_failed_prompts"] == 1


def test_merge_group_chunk_and_prepare_vllm_targets(monkeypatch):
    merged, meta = helpers.CompletionGenerator._merge_group_chunk(
        [["a"], ["b"], ["c"]], [["m1"], ["m2"], ["m3"]], requested_n=2
    )
    assert merged == ["a", "b"]
    assert meta == ["m1", "m2"]

    generator = _make_generator()
    with pytest.raises(ValueError):
        generator._prepare_vllm_targets(["p"], 1, [1, 2])

    unique, counts, mapping = generator._prepare_vllm_targets(["p"], 2, None)
    assert unique == ["p"] and counts == [2] and mapping is None
    expanded, expanded_meta = generator._expand_dedup_results([["x"]], None, None)
    assert expanded == [["x"]] and expanded_meta is None


def test_run_vllm_rounds_retries(monkeypatch):
    stats = {"vllm_retry_rounds": 0, "vllm_failed_prompts": 0}
    generator = _make_generator(use_vllm=True, generation_stats=stats)
    state = helpers._VLLMGenerationState(
        prompts=["p"],
        target_counts=[1],
        requested_n=1,
        round_limit=3,
        track_logprobs=False,
    )
    attempts = {"count": 0}

    def _exec(state_obj, pending):
        attempts["count"] += 1
        if attempts["count"] >= 2:
            state_obj.aggregated[0] = ["ok"]
            return True
        return False

    monkeypatch.setattr(generator._vllm_helper, "_execute_vllm_request", _exec)
    generator._run_vllm_rounds(state)
    assert attempts["count"] == 2
    assert stats["vllm_retry_rounds"] == 1


def test_flatten_and_broadcast(monkeypatch):
    class _Accel:
        def __init__(self):
            self.num_processes = 1
            self.process_index = 0

        def gather_object(self, value):
            return [value, value]

    accel = _Accel()
    generator = _make_generator(accelerator=accel)
    flat, offsets, flat_counts = generator._flatten_prompts_for_broadcast(
        ["a", "b"], per_prompt_counts=[1, 2]
    )
    assert flat == ["a", "b", "a", "b"]
    assert offsets == [0, 2]
    assert flat_counts == [1, 2, 1, 2]

    payload = []
    dist_stub = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        broadcast_object_list=lambda obj, src=0: payload.append((list(obj), src)),
    )
    monkeypatch.setattr(helpers, "dist", dist_stub)
    helpers._broadcast_object_list(SimpleNamespace(), [1, 2, 3], src=1)
    assert payload[0][0] == [1, 2, 3] and payload[0][1] == 1


def test_generate_empty_and_vllm_flag(monkeypatch):
    generator = _make_generator(use_vllm=True)
    assert generator.generate([], 1) == ([], None)
    monkeypatch.setattr(
        generator, "_generate_vllm_collective", lambda *args, **kwargs: ("vllm", "meta")
    )
    out = generator.generate(["p"], 1)
    assert out == ("vllm", "meta")


def test_maybe_sync_vllm_weights_skips_when_client_absent(monkeypatch):
    generator = _make_generator(use_vllm=True)
    generator.ctx.generation_stats["current_step"] = 1
    monkeypatch.setattr(generator, "_ensure_vllm_client", lambda: False)
    monkeypatch.setattr(
        generator,
        "_sync_model_params_to_vllm",
        lambda *_: (_ for _ in ()).throw(RuntimeError("should not run")),
    )
    generator._maybe_sync_vllm_weights()  # should be a no-op without raising


def test_push_param_and_reset_cache_noops(monkeypatch):
    generator = _make_generator(use_vllm=True)
    generator._vllm_client = SimpleNamespace()  # missing update_named_param
    generator._push_param_to_vllm("w", SimpleNamespace(data="d"))  # should not crash

    class _Client:
        def reset_prefix_cache(self):
            raise AttributeError("boom")

    generator._vllm_client = _Client()
    generator._reset_vllm_cache()  # swallows errors


def test_resolve_vllm_round_limit_uses_config():
    generator = _make_generator(use_vllm=True)
    generator.ctx.vllm.rounds_cfg = 5
    assert generator._resolve_vllm_round_limit(2) == 5
    generator.ctx.vllm.rounds_cfg = 0
    assert generator._resolve_vllm_round_limit(2) == 2


def test_generate_local_truncates_before_build(monkeypatch):
    generator = _make_generator()
    seen_prompts = {}
    monkeypatch.setattr(helpers, "_truncate_prompt", lambda p, limit=None: p[:1])

    def _build(prompts, counts):
        seen_prompts["prompts"] = list(prompts)
        return ["x"], [0]

    monkeypatch.setattr(generator, "_build_local_prompt_requests", _build)
    monkeypatch.setattr(
        generator, "_tokenize_expanded_prompts", lambda prompts: (prompts, [1])
    )
    monkeypatch.setattr(generator, "_run_local_model", lambda enc, lens: ["out"])
    grouped, meta = generator._generate_local(["abcd"], 1, None)
    assert grouped == [["out"]]
    assert meta is None
    assert seen_prompts["prompts"] == ["a"]


def test_summarize_grouped_shows_overflow():
    summary = helpers.CompletionGenerator._summarize_grouped(
        [["a"], ["b"], ["c"]], limit=2
    )
    assert "...(" in summary and "0:len=1" in summary


def test_request_vllm_batch_returns_none(monkeypatch):
    generator = _make_generator(use_vllm=True)
    monkeypatch.setattr(generator, "_invoke_vllm_requests", lambda prompts, n: None)
    grouped, meta = generator._request_vllm_batch(["p"], 1)
    assert grouped is None and meta is None


def test_merge_vllm_results_without_meta(monkeypatch):
    stats = {}
    generator = _make_generator(use_vllm=True, generation_stats=stats)
    state = helpers._VLLMGenerationState(
        prompts=["p0"],
        target_counts=[2],
        requested_n=2,
        round_limit=1,
        track_logprobs=False,
    )
    generator._merge_vllm_results(
        state, grouped=[["a", "b"]], grouped_meta=None, pending_indices=[0]
    )
    assert state.aggregated[0] == ["a", "b"]


def test_record_vllm_failure_updates_stats():
    stats = {"vllm_failed_prompts": 0}
    generator = _make_generator(use_vllm=True, generation_stats=stats)
    state = helpers._VLLMGenerationState(
        prompts=["p0"],
        target_counts=[1],
        requested_n=1,
        round_limit=1,
        track_logprobs=False,
    )
    generator._record_vllm_failure(state, [0])
    assert stats["vllm_failed_prompts"] == 1


def test_coalesce_grouped_passthrough_cases():
    groups = [["a"], ["b"]]
    meta = [[None], [None]]
    regrouped, regrouped_meta = helpers.CompletionGenerator._coalesce_grouped_outputs(
        groups, prompt_count=2, requested_n=1, meta=meta
    )
    assert regrouped == groups and regrouped_meta == meta

    regrouped2, meta2 = helpers.CompletionGenerator._coalesce_grouped_outputs(
        [["x"], ["y"], ["z"]], prompt_count=2, requested_n=0, meta=None
    )
    assert regrouped2 == [["x"], ["y"], ["z"]] and meta2 is None


def test_expand_dedup_results_missing_index():
    expanded, meta = helpers.CompletionGenerator._expand_dedup_results(
        grouped=[["a"]], meta=[["m"]], mapping=[0, 2]
    )
    assert expanded == [["a"], []]
    assert meta == [["m"], []]


def test_execute_vllm_request_returns_false(monkeypatch):
    generator = _make_generator(use_vllm=True)
    state = helpers._VLLMGenerationState(
        prompts=["p"],
        target_counts=[1],
        requested_n=1,
        round_limit=1,
        track_logprobs=False,
    )
    monkeypatch.setattr(
        generator._vllm_helper,
        "_request_vllm_batch",
        lambda *args, **kwargs: (None, None),
    )
    assert generator._execute_vllm_request(state, [0]) is False


def test_scatter_vllm_payload_multirank(monkeypatch):
    generator = _make_generator(use_vllm=True)

    class _Accel:
        def __init__(self):
            self.num_processes = 2
            self.process_index = 1
            self.is_main_process = False

    generator.ctx.accelerator = _Accel()
    monkeypatch.setattr(
        generator._vllm_helper,
        "_scatter_object",
        lambda accel, payload, src=0: ([["g1"]], [["m1"]]),
    )
    grouped, meta = generator._scatter_vllm_payload(
        flat_prompts=["p0", "p1"],
        offsets=[0, 1],
        grouped_all=[["g0"], ["g1"]],
        meta_all=[["m0"], ["m1"]],
    )
    assert grouped == [["g1"]] and meta == [["m1"]]


def test_generate_vllm_collective_main_process(monkeypatch):
    class _Accel:
        def __init__(self):
            self.num_processes = 1
            self.process_index = 0
            self.is_main_process = True

        def gather_object(self, value):
            return [value]

    generator = _make_generator(use_vllm=True, accelerator=_Accel())
    monkeypatch.setattr(
        generator,
        "_generate_with_vllm",
        lambda prompts, num_samples, counts: ([["ok"]], [["m"]]),
    )
    grouped, meta = generator._generate_vllm_collective(["p"], 1)
    assert grouped == [["ok"]] and meta == [["m"]]


def test_gather_object_list_dist(monkeypatch):
    payloads = []

    class _Dist:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_world_size(self):
            return 2

        def all_gather_object(self, out, val):
            out[0] = ["a"]
            out[1] = ["b"]
            payloads.append((out, val))

    accel = SimpleNamespace()
    monkeypatch.setattr(helpers, "dist", _Dist())
    gathered = helpers._gather_object_list(accel, ["x"])
    assert gathered == [["a"], ["b"]]


def test_scatter_object_prefers_accelerator(monkeypatch):
    class _Accel:
        num_processes = 2
        process_index = 1

        def scatter_object(self, input_list, src=0):
            return "from-accel"

    accel = _Accel()
    monkeypatch.setattr(
        helpers,
        "dist",
        SimpleNamespace(is_available=lambda: False, is_initialized=lambda: False),
    )
    result = helpers._scatter_object(accel, [["a"], ["b"]], src=0)
    assert result == "from-accel"


def test_dist_fallback_created_on_reload(monkeypatch):
    import importlib

    stub_torch = SimpleNamespace(distributed=None)
    stub_accel = ModuleType("accelerate")
    stub_accel.Accelerator = lambda **_k: None
    stub_tf = ModuleType("transformers")
    stub_tf.PreTrainedModel = type("PreTrainedModel", (), {})
    stub_tf.PreTrainedTokenizer = type("PreTrainedTokenizer", (), {})

    with monkeypatch.context() as m:
        m.setitem(sys.modules, "torch", stub_torch)
        m.setitem(sys.modules, "accelerate", stub_accel)
        m.setitem(sys.modules, "transformers", stub_tf)
        mod = importlib.reload(
            importlib.import_module("maxent_grpo.training.rollout.helpers")
        )
        dist_obj = mod.dist
        assert dist_obj.is_available() is False
        assert dist_obj.is_initialized() is False
        assert dist_obj.get_world_size() == 1
        out = [None]
        dist_obj.all_gather_object(out, "x")
        assert out[0] == "x"
        dist_obj.broadcast_object_list([])
    importlib.reload(importlib.import_module("maxent_grpo.training.rollout.helpers"))


def test_ensure_vllm_client_reuses_existing(monkeypatch):
    generator = _make_generator(use_vllm=True)
    generator.ctx.vllm.sync_weights = True
    generator.ctx.accelerator = SimpleNamespace(is_main_process=True)
    generator._vllm_client = object()
    generator._vllm_sync_ready = True
    monkeypatch.setattr(
        helpers,
        "_import_vllm_client_cls",
        lambda: (_ for _ in ()).throw(RuntimeError("should not load")),
    )
    assert generator._ensure_vllm_client() is True


def test_maybe_sync_vllm_weights_unwrap_failure(monkeypatch):
    generator = _make_generator(use_vllm=True, generation_stats={"current_step": None})
    generator.ctx.vllm.sync_weights = True
    called = {}

    class _Accel:
        is_main_process = True

        def unwrap_model(self, model):
            raise AttributeError("nope")

        def wait_for_everyone(self):
            called["wait"] = True

    generator.ctx.accelerator = _Accel()
    monkeypatch.setattr(generator, "_ensure_vllm_client", lambda: True)
    monkeypatch.setattr(
        generator,
        "_sync_model_params_to_vllm",
        lambda model, accel: called.setdefault("synced", model),
    )
    generator._maybe_sync_vllm_weights()
    assert called["synced"] is generator.ctx.model
    assert called.get("wait") is True


def test_sync_model_params_peft_path(monkeypatch):
    generator = _make_generator(use_vllm=True)
    calls = {"reset": 0}
    monkeypatch.setattr(
        generator._vllm_helper,
        "_reset_vllm_cache",
        lambda: calls.__setitem__("reset", calls["reset"] + 1),
    )
    monkeypatch.setattr(generator._vllm_helper, "_is_peft_model_safe", lambda *_: True)
    monkeypatch.setattr(
        generator._vllm_helper,
        "_sync_peft_params",
        lambda *a, **k: calls.setdefault("peft", True),
    )
    generator._sync_model_params_to_vllm(SimpleNamespace(), SimpleNamespace())
    assert calls["peft"] is True
    assert calls["reset"] == 1


def test_generate_local_skips_zero_targets(monkeypatch):
    generator = _make_generator()
    grouped, meta = generator._generate_local(
        ["p0"], num_samples=1, per_prompt_counts=[0]
    )
    assert grouped == [[]]
    assert meta is None


def test_invoke_vllm_requests_combines_meta(monkeypatch):
    def _safe_generate(**kwargs):
        return [["a"], ["b"]], [[None], [None]], 1.0

    monkeypatch.setattr(helpers, "safe_generate", _safe_generate)
    generator = _make_generator(use_vllm=True)
    grouped, meta, latency = generator._invoke_vllm_requests(["p1", "p2"], 1)
    assert grouped == [["a"], ["b"]]
    assert meta == [[None], [None]]
    assert latency == pytest.approx(1.0)


def test_run_vllm_rounds_runtime_error_sleep(monkeypatch):
    slept = {}

    def _sleep(secs):
        slept["t"] = secs

    monkeypatch.setattr(helpers.time, "sleep", _sleep)
    stats = {"vllm_retry_rounds": 0}
    generator = _make_generator(use_vllm=True, generation_stats=stats)
    generator.ctx.vllm.retry_sleep = 0.5
    state = helpers._VLLMGenerationState(
        prompts=["p"],
        target_counts=[1],
        requested_n=1,
        round_limit=2,
        track_logprobs=False,
    )
    calls = {"count": 0}

    def _exec(state_obj, pending):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("fail once")
        state_obj.aggregated[0] = ["done"]
        return True

    monkeypatch.setattr(generator._vllm_helper, "_execute_vllm_request", _exec)
    generator._run_vllm_rounds(state)
    assert slept["t"] == pytest.approx(0.5)
    assert calls["count"] == 2


def test_merge_group_chunk_no_meta():
    merged, meta = helpers.CompletionGenerator._merge_group_chunk(
        [["a"], ["b"]], None, requested_n=0
    )
    assert merged == ["a", "b"]
    assert meta is None


def test_broadcast_vllm_payload_defaults(monkeypatch):
    generator = _make_generator(use_vllm=True)
    payload = [None, None]
    called = {}

    class _Accel:
        def broadcast_object_list(self, obj, src=0):
            called["payload"] = obj
            called["src"] = src

    generator.ctx.accelerator = _Accel()
    grouped, meta = generator._broadcast_vllm_payload(["p1", "p2"], payload)
    assert grouped == [[], []] and meta is None
    assert called["src"] == 0


def test_scatter_vllm_payload_main_process(monkeypatch):
    generator = _make_generator(use_vllm=True)

    class _Accel:
        def __init__(self):
            self.num_processes = 2
            self.process_index = 0
            self.is_main_process = True

    generator.ctx.accelerator = _Accel()
    captured = {}

    def _scatter(accel, payload, src=0):
        captured["payload"] = payload
        return payload[accel.process_index]

    monkeypatch.setattr(generator._vllm_helper, "_scatter_object", _scatter)
    grouped, meta = generator._scatter_vllm_payload(
        flat_prompts=["p0", "p1"],
        offsets=[0, 1],
        grouped_all=[["g0"], ["g1"]],
        meta_all=[["m0"], ["m1"]],
    )
    assert captured["payload"][0][0] == [["g0"]]
    assert grouped == [["g0"]]
    assert meta == [["m0"]]


def test_expand_dedup_results_handles_meta_none():
    grouped, meta = helpers.CompletionGenerator._expand_dedup_results(
        grouped=[["a"]], meta=None, mapping=[0]
    )
    assert grouped == [["a"]] and meta is None


def test_vllm_round_limit_respects_env(monkeypatch):
    generator = _make_generator(use_vllm=True)
    generator.ctx.vllm.rounds_cfg = -1
    assert generator._resolve_vllm_round_limit(3) == 3


def test_build_vllm_request_kwargs_tokenizer_passthrough(monkeypatch):
    tok = SimpleNamespace()
    generator = _make_generator(use_vllm=True, tokenizer=tok)
    kwargs = generator._build_vllm_request_kwargs(["p"], 2)
    assert kwargs["tokenizer"] is tok


def test_request_vllm_batch_warns_on_mismatch(monkeypatch, caplog):
    caplog.set_level("WARNING")
    generator = _make_generator(use_vllm=True)
    monkeypatch.setattr(
        generator,
        "_invoke_vllm_requests",
        lambda prompts, n: ([["a"], ["b"], ["c"]], None, 1.0),
    )
    grouped, meta = generator._request_vllm_batch(["p1", "p2"], 1)
    assert grouped is None and meta is None
    assert "vLLM grouped outputs len" in caplog.text


def test_run_local_model_uses_top_k_none(monkeypatch):
    class _GenModel:
        def __init__(self):
            self.called = False

        def generate(self, **kwargs):
            self.called = True
            assert kwargs["top_k"] is None
            return [[0]]

    tok = SimpleNamespace(decode=lambda ids, skip_special_tokens=True: "ok")
    accel = SimpleNamespace(unwrap_model=lambda m: m)
    generator = _make_generator(model=_GenModel(), accelerator=accel, tokenizer=tok)
    enc, lens = {"input_ids": [[1]], "attention_mask": [[1]]}, [1]
    out = generator._run_local_model(enc, lens)
    assert out == ["ok"]
    assert generator.ctx.model.called is True  # type: ignore[attr-defined]


def test_backfill_missing_respects_need_zero(monkeypatch):
    stats = {"vllm_backfilled_prompts": 0}
    generator = _make_generator(use_vllm=True, generation_stats=stats)
    generator.ctx.vllm.backfill_local = True
    state = helpers._VLLMGenerationState(
        prompts=["p0"],
        target_counts=[1],
        requested_n=1,
        round_limit=1,
        track_logprobs=False,
    )
    state.aggregated = [["x"]]
    monkeypatch.setattr(generator, "_generate_local", lambda *a, **k: ([["y"]], None))
    generator._backfill_missing(state, [])
    assert state.aggregated == [["x"]]


def test_prepare_vllm_targets_negative_counts(monkeypatch):
    generator = _make_generator()
    prompts, counts, mapping = generator._prepare_vllm_targets(
        ["p"], 1, per_prompt_counts=[-5]
    )
    assert counts == [0]
    assert mapping is None


def test_execute_vllm_request_skips_zero_need(monkeypatch):
    generator = _make_generator(use_vllm=True)
    state = helpers._VLLMGenerationState(
        prompts=["p"],
        target_counts=[0],
        requested_n=1,
        round_limit=1,
        track_logprobs=False,
    )
    assert generator._execute_vllm_request(state, [0]) is True


def test_flatten_prompts_for_broadcast_handles_counts(monkeypatch):
    class _Accel:
        def __init__(self):
            self.num_processes = 1
            self.process_index = 0

        def gather_object(self, value):
            return [value]

    gen = _make_generator(accelerator=_Accel())
    flat_prompts, offsets, flat_counts = gen._flatten_prompts_for_broadcast(
        ["a"], per_prompt_counts=[3]
    )
    assert flat_prompts == ["a"]
    assert offsets == [0]
    assert flat_counts == [3]


def test_pluck_rank_outputs_fills_missing(monkeypatch):
    class _Accel:
        process_index = 0

    gen = _make_generator(accelerator=_Accel())
    grouped, meta = gen._pluck_rank_outputs(
        [["x"], None], meta_all=[[None], None], offsets=[0], prompts=["p"]
    )
    assert grouped == [["x"]]
    assert meta == [[None]]
