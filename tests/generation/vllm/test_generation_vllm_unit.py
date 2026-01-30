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

Unit tests for lightweight helpers in ``generation.vllm``.
"""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

from tests.helpers.vllm import make_vllm_context

import pytest


def _load_vllm(monkeypatch):
    """Import generation.vllm with torch/accelerate stubs installed."""

    repo_root = Path(__file__).resolve().parent.parent
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

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

    # Ensure optional TRL VLLMClient import is treated as missing unless explicitly stubbed.
    trl_stub = ModuleType("trl")
    trl_extras = ModuleType("trl.extras")
    trl_stub.extras = trl_extras
    monkeypatch.setitem(sys.modules, "trl", trl_stub)
    monkeypatch.setitem(sys.modules, "trl.extras", trl_extras)
    monkeypatch.setitem(
        sys.modules, "trl.extras.vllm_client", ModuleType("trl.extras.vllm_client")
    )

    accelerate_stub = ModuleType("accelerate")
    accelerate_stub.Accelerator = type("Accelerator", (), {})
    monkeypatch.setitem(sys.modules, "accelerate", accelerate_stub)

    patches_mod = ModuleType("maxent_grpo.patches.vllm")
    patches_mod.VLLMLogprobResult = type("VLLMLogprobResult", (), {})

    def _safe_generate(**_kwargs):
        return [[_kwargs["prompts"][0]]], None, 1.0

    patches_mod.safe_generate = _safe_generate
    monkeypatch.setitem(sys.modules, "maxent_grpo.patches.vllm", patches_mod)

    # Ensure the generation package path is importable
    sys.modules.pop("maxent_grpo.generation.vllm", None)
    module = importlib.import_module("maxent_grpo.generation.vllm")
    return module


def _ctx(**kwargs):
    return make_vllm_context(**kwargs)


def test_optional_import_returns_none_and_module(monkeypatch):
    module = _load_vllm(monkeypatch)

    class _Client:
        pass

    dummy = ModuleType("trl.extras.vllm_client")
    dummy.VLLMClient = _Client
    monkeypatch.setitem(sys.modules, "trl.extras.vllm_client", dummy)
    assert module._import_vllm_client_cls() is _Client
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
    accel_utils.is_peft_model = lambda _target: (_ for _ in ()).throw(TypeError("boom"))
    monkeypatch.setitem(sys.modules, "accelerate.utils", accel_utils)
    assert module._is_peft_model_safe(object()) is False


def test_is_peft_model_safe_handles_noncallable_and_exception(monkeypatch):
    module = _load_vllm(monkeypatch)
    accel_utils = ModuleType("accelerate.utils")
    accel_utils.is_peft_model = "not-callable"
    monkeypatch.setitem(sys.modules, "accelerate.utils", accel_utils)
    assert module._is_peft_model_safe(object()) is False
    accel_utils.is_peft_model = lambda _t: (_ for _ in ()).throw(TypeError("bad"))
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


def test_vllm_generation_state_validates_mismatched_targets(monkeypatch):
    module = _load_vllm(monkeypatch)
    with pytest.raises(ValueError):
        module._VLLMGenerationState(
            prompts=["only_one"],
            target_counts=[1, 2],
            requested_n=1,
            round_limit=1,
            track_logprobs=False,
        )
    with pytest.raises(ValueError):
        module._VLLMGenerationState(
            prompts=["p1"],
            target_counts=[1, 2],
            requested_n=1,
            round_limit=1,
            track_logprobs=False,
        )
    state_simple = module._VLLMGenerationState(
        prompts=["p1"],
        target_counts=[1],
        requested_n=1,
        round_limit=1,
        track_logprobs=False,
    )
    state_simple.aggregated[0].append("z")
    trimmed_simple, meta_simple = state_simple.trim()
    assert trimmed_simple == [["z"]]
    assert meta_simple is None


def test_helper_base_url_strips_generate_and_scheme(monkeypatch):
    module = _load_vllm(monkeypatch)
    helper = module.VLLMGenerationHelper(_ctx(), lambda *_args: ([], None))
    assert helper._vllm_base_url("http://host:8000/generate") == "http://host:8000"
    assert helper._vllm_base_url("invalid://") == "invalid:"
    monkeypatch.setattr(
        "urllib.parse.urlparse", lambda _url: (_ for _ in ()).throw(ValueError("bad"))
    )
    assert helper._vllm_base_url("weird/generate") == "weird"
    monkeypatch.setattr(
        "urllib.parse.urlparse",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad")),
    )
    assert helper._vllm_base_url("weird") == "weird"


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
    helper_default = module.VLLMGenerationHelper(
        _ctx(vllm_rounds_cfg=0), lambda *_: ([], None)
    )
    assert helper_default._resolve_vllm_round_limit(2) == 2


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


def test_ensure_vllm_client_requires_init(monkeypatch):
    module = _load_vllm(monkeypatch)

    class _Client:
        def __init__(self, base_url=None):
            self.base_url = base_url

    monkeypatch.setattr(module, "_import_vllm_client_cls", lambda: _Client)
    helper = module.VLLMGenerationHelper(
        _ctx(vllm_sync_weights=True), lambda *_: ([], None)
    )
    assert helper._ensure_vllm_client() is False
    assert helper._vllm_sync_ready is False


def test_ensure_vllm_client_skips_non_main_and_missing_init(monkeypatch):
    module = _load_vllm(monkeypatch)
    ctx = _ctx(
        vllm_sync_weights=True, accelerator=SimpleNamespace(is_main_process=False)
    )
    helper = module.VLLMGenerationHelper(ctx, lambda *_: ([], None))
    assert helper._ensure_vllm_client() is False

    class _ClientNoInit:
        pass

    monkeypatch.setattr(module, "_import_vllm_client_cls", lambda: _ClientNoInit)
    ctx_main = _ctx(vllm_sync_weights=True)
    helper = module.VLLMGenerationHelper(ctx_main, lambda *_: ([], None))
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


def test_coalesce_grouped_outputs_regroups_microgroups_with_logprob_meta(monkeypatch):
    module = _load_vllm(monkeypatch)
    helper = module.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    # Mirror the "raw groups = prompt_count * req_n, each group has len=1" case.
    prompt_count = 6
    requested_n = 8
    total_groups = prompt_count * requested_n
    groups = [[f"c{i}"] for i in range(total_groups)]
    meta = [
        [{"logprob_sum": -float(i), "token_count": 1}] for i in range(total_groups)
    ]
    regrouped, regrouped_meta = helper._coalesce_grouped_outputs(
        groups, prompt_count, requested_n, meta
    )
    assert len(regrouped) == prompt_count
    assert all(len(entry) == requested_n for entry in regrouped)
    assert regrouped_meta is not None
    assert len(regrouped_meta) == prompt_count
    assert all(len(entry) == requested_n for entry in regrouped_meta)
    assert regrouped_meta[0][0] is not None and regrouped_meta[0][0]["logprob_sum"] == 0.0
    assert regrouped_meta[0][1] is not None and regrouped_meta[0][1]["logprob_sum"] == -1.0


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
    # default path uses PROMPT_CHAR_LIMIT
    assert helper._prompt_char_limit() == module.PROMPT_CHAR_LIMIT
    ctx.prompt_char_limit = 5
    assert helper._prompt_char_limit() == 5
    # PROMPT_CHAR_LIMIT <= 0 falls back to approx chars
    monkeypatch.setattr(module, "PROMPT_CHAR_LIMIT", 0)
    ctx.prompt_char_limit = None
    ctx.max_prompt_len = 2
    assert helper._prompt_char_limit() == 8
    # approx chars <= 0 returns PROMPT_CHAR_LIMIT when positive
    monkeypatch.setattr(module, "PROMPT_CHAR_LIMIT", 9)
    ctx.max_prompt_len = 0
    assert helper._prompt_char_limit() == 9


def test_build_vllm_request_kwargs_attaches_metadata(monkeypatch):
    module = _load_vllm(monkeypatch)
    ctx = _ctx()
    ctx.generation_stats["dataset_name"] = "train-ds"
    ctx.generation_stats["model_id"] = "org/model"
    ctx.model = SimpleNamespace(name_or_path="org/model")
    helper = module.VLLMGenerationHelper(ctx, lambda *_: ([], None))
    kwargs = helper._build_vllm_request_kwargs(["p"], 1)
    assert kwargs["metadata"] == {"dataset": "train-ds", "model_id": "org/model"}


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
    ctx_disabled = _ctx(vllm_backfill_local=False)
    helper_disabled = module.VLLMGenerationHelper(ctx_disabled, lambda *_: ([], None))
    state_disabled = module._VLLMGenerationState(
        prompts=["p1"],
        target_counts=[1],
        requested_n=1,
        round_limit=1,
        track_logprobs=False,
    )
    helper_disabled._backfill_missing(state_disabled, [0])
    assert state_disabled.aggregated[0] == []


def test_backfill_missing_skips_satisfied_prompts(monkeypatch):
    module = _load_vllm(monkeypatch)
    ctx = _ctx(vllm_backfill_local=True)
    helper = module.VLLMGenerationHelper(
        ctx, lambda prompts, *_: ([["extra"]] * len(prompts), None)
    )
    state = module._VLLMGenerationState(
        prompts=["p1"],
        target_counts=[1],
        requested_n=1,
        round_limit=1,
        track_logprobs=True,
    )
    state.aggregated[0].append("done")
    state.aggregated_meta[0].append("meta")
    helper._backfill_missing(state, [0])
    # Already satisfied prompt stays unchanged; meta dropped to save memory.
    assert state.aggregated[0] == ["done"]
    assert state.aggregated_meta is None


def test_ensure_vllm_client_initializes(monkeypatch):
    module = _load_vllm(monkeypatch)
    created = {}

    class _Client:
        def __init__(self, base_url):
            created["url"] = base_url

        def init_communicator(self):
            created["init"] = True

    monkeypatch.setattr(module, "_import_vllm_client_cls", lambda: _Client)
    ctx = _ctx(vllm_sync_weights=True)
    helper = module.VLLMGenerationHelper(ctx, lambda *_: ([], None))
    assert helper._ensure_vllm_client() is True
    assert helper._vllm_sync_ready is True
    assert created["url"].endswith("localhost:8000")
    # cached path should short circuit
    assert helper._ensure_vllm_client() is True


def test_maybe_sync_weights_pushes_params(monkeypatch):
    module = _load_vllm(monkeypatch)
    pushes = []

    class _Client:
        def update_named_param(self, name, data):
            pushes.append((name, data))

        def reset_prefix_cache(self):
            pushes.append("reset")

    class _Model:
        def __init__(self):
            self._p1 = SimpleNamespace(data="d1")
            self._p2 = SimpleNamespace(data="d2")

        def parameters(self):
            return [self._p1, self._p2]

        def named_parameters(self):
            return [("p1", self._p1), ("p2", self._p2)]

    calls = {"wait": 0}
    accel = SimpleNamespace(
        is_main_process=True,
        process_index=0,
        num_processes=1,
        unwrap_model=lambda m: m,
        wait_for_everyone=lambda: calls.__setitem__("wait", calls["wait"] + 1),
    )
    ctx = _ctx(
        accelerator=accel,
        vllm_sync_weights=True,
        generation_stats={"current_step": 5},
        model=_Model(),
    )
    helper = module.VLLMGenerationHelper(ctx, lambda *_: ([], None))
    helper._vllm_client = _Client()
    helper._vllm_sync_ready = True
    helper.maybe_sync_weights()
    assert ("p1", "d1") in pushes and ("p2", "d2") in pushes
    assert "reset" in pushes
    assert ctx.generation_stats["vllm_weight_syncs"] == 1
    assert helper._last_vllm_synced_step == 5
    assert calls["wait"] == 1


def test_maybe_sync_weights_skips_when_already_synced(monkeypatch):
    module = _load_vllm(monkeypatch)
    ctx = _ctx(generation_stats={"current_step": 3})
    helper = module.VLLMGenerationHelper(ctx, lambda *_: ([], None))
    helper._vllm_sync_ready = True
    helper._last_vllm_synced_step = 3
    helper._sync_model_params_to_vllm = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("should skip")
    )
    helper.maybe_sync_weights()


def test_maybe_sync_weights_skips_repeat_and_fallback_model(monkeypatch):
    module = _load_vllm(monkeypatch)
    ctx = _ctx(
        vllm_sync_weights=True,
        generation_stats={"current_step": 2},
        model=SimpleNamespace(),
        accelerator=SimpleNamespace(
            is_main_process=True,
            process_index=0,
            num_processes=1,
            unwrap_model=lambda *_: (_ for _ in ()).throw(TypeError("nope")),
            wait_for_everyone=lambda: None,
        ),
    )
    helper = module.VLLMGenerationHelper(ctx, lambda *_: ([], None))
    helper._vllm_client = SimpleNamespace()
    helper._vllm_sync_ready = True
    helper._last_vllm_synced_step = 2
    helper.maybe_sync_weights()  # short-circuits due to identical step
    helper._last_vllm_synced_step = None
    marker = {}
    monkeypatch.setattr(
        helper,
        "_sync_model_params_to_vllm",
        lambda model: marker.setdefault("used_model", model),
    )
    helper.maybe_sync_weights()
    assert marker["used_model"] is ctx.model


def test_maybe_sync_weights_returns_when_client_missing(monkeypatch):
    module = _load_vllm(monkeypatch)
    ctx = _ctx(vllm_sync_weights=True)
    helper = module.VLLMGenerationHelper(ctx, lambda *_: ([], None))
    monkeypatch.setattr(helper, "_ensure_vllm_client", lambda: False)
    helper.maybe_sync_weights()  # should simply return


def test_sync_peft_params_merges_and_filters(monkeypatch):
    module = _load_vllm(monkeypatch)
    monkeypatch.setattr(module, "_is_peft_model_safe", lambda _m: True)
    pushed = []

    class _Client:
        def update_named_param(self, name, data):
            pushed.append(name)

        def reset_prefix_cache(self):
            pushed.append("reset")

    class _Model:
        prefix = "skip"

        def __init__(self):
            self.called = []
            self.p = SimpleNamespace(data="x")

        def merge_adapter(self):
            self.called.append("merge")

        def unmerge_adapter(self):
            self.called.append("unmerge")

        def parameters(self):
            return [self.p]

        def named_parameters(self):
            return [
                ("base_model.model.layer.weight", self.p),
                ("base_model.model.skip.adapter.weight", self.p),
                ("modules_to_save.default.head.weight", self.p),
                ("original_module.layer", self.p),
            ]

    model = _Model()
    helper = module.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    helper._vllm_client = _Client()
    helper._vllm_sync_ready = True
    helper._sync_model_params_to_vllm(model)
    assert "merge" in model.called and "unmerge" in model.called
    assert "layer.weight" in pushed
    assert "head.weight" in pushed
    assert all("original_module" not in name for name in pushed)
    assert pushed[-1] == "reset"


def test_sync_model_params_handles_fsdp(monkeypatch):
    module = _load_vllm(monkeypatch)
    pushed = []

    class _Param(SimpleNamespace):
        pass

    class _FSDP:
        def __init__(self, children=None, name="root"):
            self.children = children or []
            self._params = [("p", _Param(data=name))]

        def named_children(self):
            return [("child", child) for child in self.children]

        def named_parameters(self):
            return self._params

        def parameters(self):
            return [p for _, p in self._params]

        @classmethod
        def summon_full_params(cls, _module, recurse=False, writeback=False):
            @contextmanager
            def _ctx():
                yield

            return _ctx()

    child = _FSDP(name="child")
    root = _FSDP(children=[child])
    dist = ModuleType("torch.distributed")
    dist.fsdp = SimpleNamespace(FullyShardedDataParallel=_FSDP)
    dist.is_available = lambda: False
    dist.is_initialized = lambda: False
    monkeypatch.setattr(module, "torch", SimpleNamespace(distributed=dist))

    helper = module.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    helper._push_param_to_vllm = lambda name, param: pushed.append((name, param.data))
    helper._sync_model_params_to_vllm(root)
    assert ("child.p", "child") in pushed


def test_sync_model_params_fsdp_branch(monkeypatch):
    module = _load_vllm(monkeypatch)
    pushed = []

    class _Param:
        def __init__(self, data):
            self.data = data

    class FSDP:
        def __init__(self, children=None, params=None):
            self._children = children or []
            self._params = params or []

        def named_children(self):
            return [(f"child{i}", child) for i, child in enumerate(self._children)]

        def named_parameters(self):
            return [(f"w{i}", p) for i, p in enumerate(self._params)]

        @staticmethod
        def summon_full_params(_module, recurse=False, writeback=False):
            @contextmanager
            def _ctx():
                yield

            return _ctx()

    child = FSDP(params=[_Param("p1")])
    root = FSDP(children=[child])
    module.torch.distributed = SimpleNamespace(
        fsdp=SimpleNamespace(FullyShardedDataParallel=FSDP)
    )
    helper = module.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    helper._vllm_client = SimpleNamespace(
        update_named_param=lambda name, data: pushed.append((name, data)),
        reset_prefix_cache=lambda: pushed.append("reset"),
    )
    helper._vllm_sync_ready = True
    helper._sync_model_params_to_vllm(root)
    assert ("child0.w0", "p1") in pushed
    assert "reset" in pushed


def test_sync_fsdp_params_returns_when_cls_missing(monkeypatch):
    module = _load_vllm(monkeypatch)
    helper = module.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    helper._fsdp_cls = None

    class _Dummy:
        def named_children(self):
            return []

        def named_parameters(self):
            return []

    helper._push_param_to_vllm = lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("should not push")
    )
    helper._sync_fsdp_params(_Dummy())


def test_invoke_vllm_requests_recurses_and_combines(monkeypatch):
    module = _load_vllm(monkeypatch)

    def _safe_generate(**kwargs):
        if len(kwargs["prompts"]) > 1:
            raise RuntimeError("split please")
        return [[kwargs["prompts"][0] + "!"]], [[None]], 1.0

    monkeypatch.setattr(module, "safe_generate", _safe_generate)
    helper = module.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    grouped, meta, latency = helper._invoke_vllm_requests(["a", "b"], 1)
    assert grouped == [["a!"], ["b!"]]
    assert meta == [[None], [None]]
    assert latency == 2.0


def test_invoke_vllm_requests_single_failure_and_recursive_none(monkeypatch):
    module = _load_vllm(monkeypatch)

    def _safe_generate(**kwargs):
        raise RuntimeError("nope")

    monkeypatch.setattr(module, "safe_generate", _safe_generate)
    helper = module.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    assert helper._invoke_vllm_requests(["solo"], 1) is None

    def _safe_generate_split(**kwargs):
        raise RuntimeError("split")

    monkeypatch.setattr(module, "safe_generate", _safe_generate_split)
    assert helper._invoke_vllm_requests(["a", "b"], 1) is None


def test_coalesce_grouped_outputs_mismatch_returns_meta_none(monkeypatch):
    module = _load_vllm(monkeypatch)
    helper = module.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    groups = [["a"], ["b"], ["c"]]
    meta = [["m1"], ["m2"], ["m3"]]
    regrouped, regrouped_meta = helper._coalesce_grouped_outputs(
        groups, prompt_count=2, requested_n=1, meta=meta
    )
    assert regrouped == groups
    assert regrouped_meta is None


def test_coalesce_grouped_outputs_edge_cases(monkeypatch):
    module = _load_vllm(monkeypatch)
    helper = module.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    groups = [["x"], ["y"]]
    regrouped, meta = helper._coalesce_grouped_outputs(
        groups, prompt_count=0, requested_n=1, meta=None
    )
    assert regrouped == groups and meta is None
    regrouped, meta = helper._coalesce_grouped_outputs(
        groups, prompt_count=3, requested_n=1, meta=[[], []]
    )
    assert meta is None
    regrouped, meta = helper._coalesce_grouped_outputs(
        [["a"], ["b"]], prompt_count=2, requested_n=2, meta=None
    )
    assert regrouped == [["a"], ["b"]]


def test_coalesce_grouped_outputs_returns_original_on_mismatch(monkeypatch):
    module = _load_vllm(monkeypatch)
    helper = module.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    groups = [["a"], ["b"], ["c"], ["d"]]
    meta = [[1], [2], [3], [4]]
    regrouped, meta_out = helper._coalesce_grouped_outputs(
        groups,
        prompt_count=2,
        requested_n=1,
        meta=meta,
    )
    assert regrouped == groups
    assert meta_out == meta


def test_merge_vllm_results_handles_overflow(monkeypatch):
    module = _load_vllm(monkeypatch)
    ctx = _ctx()
    helper = module.VLLMGenerationHelper(ctx, lambda *_: ([], None))
    state = module._VLLMGenerationState(
        prompts=["p"],
        target_counts=[1],
        requested_n=1,
        round_limit=1,
        track_logprobs=False,
    )
    helper._merge_vllm_results(state, [["a", "b"]], None, [0])
    assert state.aggregated[0] == ["a"]
    assert ctx.generation_stats["vllm_excess_prompts"] == 1
    assert ctx.generation_stats["vllm_excess_completions"] == 1


def test_merge_vllm_results_handles_meta(monkeypatch):
    module = _load_vllm(monkeypatch)
    helper = module.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    state = module._VLLMGenerationState(
        prompts=["p"],
        target_counts=[2],
        requested_n=2,
        round_limit=1,
        track_logprobs=True,
    )
    helper._merge_vllm_results(state, [["a", "b"]], [[None, "m"]], [0])
    assert state.aggregated_meta[0] == [None, "m"]


def test_merge_vllm_results_truncates_meta_on_overflow(monkeypatch):
    module = _load_vllm(monkeypatch)
    ctx = _ctx()
    helper = module.VLLMGenerationHelper(ctx, lambda *_: ([], None))
    state = module._VLLMGenerationState(
        prompts=["p"],
        target_counts=[1],
        requested_n=1,
        round_limit=1,
        track_logprobs=True,
    )
    helper._merge_vllm_results(state, [["a", "b"]], [["m1", "m2"]], [0])
    assert state.aggregated_meta[0] == ["m1"]
    assert ctx.generation_stats["vllm_excess_completions"] == 1


def test_merge_group_chunk_truncates_and_handles_missing_meta(monkeypatch):
    module = _load_vllm(monkeypatch)
    merged, merged_meta = module.VLLMGenerationHelper._merge_group_chunk(
        chunk=[["a"], ["b"]],
        meta_chunk=None,
        requested_n=1,
    )
    assert merged == ["a"]
    assert merged_meta is None


def test_merge_group_chunk_drops_meta_when_slice_missing(monkeypatch):
    module = _load_vllm(monkeypatch)
    merged, merged_meta = module.VLLMGenerationHelper._merge_group_chunk(
        chunk=[["a"], ["b"]],
        meta_chunk=[[1]],
        requested_n=0,
    )
    assert merged == ["a", "b"]
    assert merged_meta is None


def test_record_vllm_failure_increments(monkeypatch):
    module = _load_vllm(monkeypatch)
    ctx = _ctx()
    helper = module.VLLMGenerationHelper(ctx, lambda *_: ([], None))
    state = module._VLLMGenerationState(
        prompts=["p"],
        target_counts=[2],
        requested_n=2,
        round_limit=2,
        track_logprobs=False,
    )
    helper._record_vllm_failure(state, [0])
    assert ctx.generation_stats["vllm_failed_prompts"] == 1


def test_merge_group_chunk_handles_missing_meta(monkeypatch):
    module = _load_vllm(monkeypatch)
    merged, merged_meta = module.VLLMGenerationHelper._merge_group_chunk(
        [["x"], ["y"]],
        None,
        requested_n=0,
    )
    assert merged == ["x", "y"]
    assert merged_meta is None
    merged_trim, merged_meta_trim = module.VLLMGenerationHelper._merge_group_chunk(
        [["a"], ["b"]],
        [[1], [2]],
        requested_n=1,
    )
    assert merged_trim == ["a"]
    assert merged_meta_trim == [1]


def test_client_callable_and_push_reset_guards(monkeypatch):
    module = _load_vllm(monkeypatch)
    helper = module.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    assert helper._client_callable("missing") is None
    helper._vllm_client = SimpleNamespace(update_named_param="notcallable")
    assert helper._client_callable("update_named_param") is None
    helper._vllm_client = None
    helper._reset_vllm_cache()  # no client, no-op

    calls = []

    def _callable(name=None):
        calls.append(name)
        if name == "raise":
            raise ValueError("fail")

    helper._vllm_client = SimpleNamespace(
        update_named_param=lambda *_a, **_k: calls.append("update"),
        reset_prefix_cache=lambda: _callable("reset"),
    )
    helper._push_param_to_vllm("n", SimpleNamespace(data="d"))
    helper._push_param_to_vllm("n", None)  # no-op
    helper._reset_vllm_cache()
    helper._vllm_client = SimpleNamespace(reset_prefix_cache=lambda: _callable("raise"))
    helper._reset_vllm_cache()
    helper._vllm_client = SimpleNamespace(
        update_named_param=lambda *_a, **_k: (_ for _ in ()).throw(ValueError("boom"))
    )
    helper._push_param_to_vllm("name", SimpleNamespace(data="d"))
    assert "update" in calls and "reset" in calls


def test_push_param_to_vllm_skips_when_callable_missing(monkeypatch):
    module = _load_vllm(monkeypatch)
    helper = module.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    helper._vllm_client = SimpleNamespace(update_named_param="not-callable")
    # Should simply return without attempting to access param.data
    helper._push_param_to_vllm("ignored", SimpleNamespace(data="d"))


def test_sync_fsdp_params_dedups_within_child(monkeypatch):
    module = _load_vllm(monkeypatch)
    pushed = []

    class _Param:
        def __init__(self, data):
            self.data = data

    class _ChildFSDP:
        def __init__(self):
            self._params = [_Param("first"), _Param("second")]

        def named_children(self):
            return []

        def named_parameters(self):
            return [
                ("_fsdp_wrapped_module.shared", self._params[0]),
                ("shared", self._params[1]),
            ]

        @staticmethod
        @contextmanager
        def summon_full_params(_module, recurse=False, writeback=False):
            yield

    class _Root(_ChildFSDP):
        def __init__(self):
            super().__init__()
            self.child = _ChildFSDP()

        def named_children(self):
            return [("child", self.child)]

    dist = SimpleNamespace(
        fsdp=SimpleNamespace(FullyShardedDataParallel=_ChildFSDP),
        is_available=lambda: False,
        is_initialized=lambda: False,
    )
    module.torch.distributed = dist
    helper = module.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    helper._push_param_to_vllm = lambda name, param: pushed.append(name)
    helper._sync_model_params_to_vllm(_Root())
    # Only one push despite duplicate clean names after stripping wrappers.
    assert pushed == ["child.shared"]


def test_sync_model_params_walks_summon_full_params(monkeypatch):
    module = _load_vllm(monkeypatch)
    pushed = []
    reset = {}

    class _Param:
        def __init__(self, name):
            self.data = name

    @contextmanager
    def _ctx_mgr():
        yield

    class _Child:
        def __init__(self, name):
            self._name = name

        def named_children(self):
            return []

        def named_parameters(self):
            return [(f"{self._name}_w", _Param(self._name))]

    class _Root:
        def __init__(self):
            self.child = _Child("child")

        @staticmethod
        def summon_full_params(*_a, **_k):
            return _ctx_mgr()

        def named_children(self):
            return [("child", self.child)]

        def named_parameters(self):
            return [("root_w", _Param("root"))]

    helper = module.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    helper._push_param_to_vllm = lambda name, param: pushed.append(name)
    helper._reset_vllm_cache = lambda: reset.setdefault("done", True)
    helper._sync_model_params_to_vllm(_Root())
    assert "child.child_w" in pushed
    assert reset.get("done") is True


def test_sync_standard_params_handles_missing_named(monkeypatch):
    module = _load_vllm(monkeypatch)

    class _Model:
        def parameters(self):
            return [1]

        named_parameters = None

    helper = module.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    helper._sync_standard_params(_Model(), lambda params: nullcontext())


def test_sync_peft_params_invokes_merge_and_unmerge(monkeypatch):
    module = _load_vllm(monkeypatch)
    pushed = []
    calls = {"merge": 0, "unmerge": 0}

    class _Param:
        def __init__(self, name):
            self.data = name

    class _Model:
        prefix = "skipme"

        def __init__(self):
            self._params = [
                ("base_model.model.layer", _Param("a")),
                ("modules_to_save.default.kept", _Param("b")),
                ("original_module.x", _Param("c")),
                ("skipme.extra", _Param("d")),
            ]

        def parameters(self):
            return [p for _, p in self._params]

        def named_parameters(self):
            return self._params

        def merge_adapter(self):
            calls["merge"] += 1

        def unmerge_adapter(self):
            calls["unmerge"] += 1

    helper = module.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    helper._push_param_to_vllm = lambda name, param: pushed.append(name)
    helper._sync_peft_params(_Model())
    assert calls["merge"] == 1 and calls["unmerge"] == 1
    # Should skip original_module and prefix-matching names, strip wrappers.
    assert pushed == ["layer", "kept"]


def test_client_callable_returns_none_for_non_callable(monkeypatch):
    module = _load_vllm(monkeypatch)
    helper = module.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    helper._vllm_client = SimpleNamespace(update_named_param="notcallable")
    assert helper._client_callable("update_named_param") is None


def test_prepare_vllm_targets_dedup_mapping(monkeypatch):
    module = _load_vllm(monkeypatch)
    helper = module.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    monkeypatch.setenv("MAXENT_VLLM_DEDUP", "1")
    prompts, counts, mapping = helper._prepare_vllm_targets(
        ["a", "b", "a"], num_samples=2, per_prompt_counts=None
    )
    assert prompts == ["a", "b"]
    assert counts == [2, 2]
    assert mapping == [0, 1, 0]


def test_build_scatter_payload_trims(monkeypatch):
    module = _load_vllm(monkeypatch)
    helper = module.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    payload = helper._build_scatter_payload(
        offsets=[0, 1],
        world_size=2,
        flat_prompts=["p0", "p1"],
        grouped_all=[["a"], ["b"]],
        meta_all=[[1], [2]],
    )
    assert payload[0][0] == [["a"]]
    assert payload[1][0] == [["b"]]
    assert payload[0][1] == [[1]]
    assert payload[1][1] == [[2]]


def test_generate_pipeline_hooks(monkeypatch):
    module = _load_vllm(monkeypatch)
    ctx = _ctx()
    helper = module.VLLMGenerationHelper(ctx, lambda *_: ([], None))
    calls = {}
    monkeypatch.setattr(
        helper, "maybe_sync_weights", lambda: calls.setdefault("sync", True)
    )
    monkeypatch.setattr(
        helper,
        "_prepare_vllm_targets",
        lambda prompts, n, counts: calls.setdefault("prep", (prompts, [n], [0])),
    )
    monkeypatch.setattr(
        helper, "_resolve_vllm_round_limit", lambda n: calls.setdefault("round", n)
    )
    monkeypatch.setattr(
        helper, "_run_vllm_rounds", lambda state: calls.setdefault("ran", state)
    )
    monkeypatch.setattr(
        helper,
        "_expand_dedup_results",
        lambda grouped, meta, mapping: (["done"], mapping),
    )
    out = helper.generate(["p"], 1, None)
    assert out == (["done"], [0])
    assert calls["prep"][0] == ["p"]


def test_generate_collective_non_main(monkeypatch):
    module = _load_vllm(monkeypatch)
    scatter_called = {}

    def _scatter(flat, offsets, grouped, meta):
        scatter_called["flat"] = flat
        scatter_called["offsets"] = offsets
        scatter_called["grouped"] = grouped
        scatter_called["meta"] = meta
        return ["local"], meta

    accel = SimpleNamespace(
        is_main_process=False,
        process_index=1,
        num_processes=2,
    )
    helper = module.VLLMGenerationHelper(_ctx(accelerator=accel), lambda *_: ([], None))
    monkeypatch.setattr(helper, "_scatter_vllm_payload", _scatter)
    grouped, meta = helper.generate_collective(["p1"], 1, [1])
    assert grouped == ["local"]
    assert scatter_called["flat"]


def test_prepare_targets_no_dedupe_and_expand_meta(monkeypatch):
    module = _load_vllm(monkeypatch)
    helper = module.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    monkeypatch.setenv("MAXENT_VLLM_DEDUP", "0")
    prompts, counts, mapping = helper._prepare_vllm_targets(["a", "b"], 2, None)
    assert mapping is None
    grouped_same, meta_same = helper._expand_dedup_results([["x"]], None, None)
    assert grouped_same == [["x"]] and meta_same is None
    grouped, meta = helper._expand_dedup_results([["x"]], [[None]], [0, 1])
    assert grouped[1] == []
    assert meta[1] == []


def test_run_vllm_rounds_handles_failures(monkeypatch):
    module = _load_vllm(monkeypatch)
    ctx = _ctx(vllm_retry_sleep=0.01, vllm_backfill_local=True)
    helper = module.VLLMGenerationHelper(ctx, lambda *_: ([], None))
    calls = {"exec": 0, "backfill": 0, "record": 0}

    def _exec(state, pending):
        calls["exec"] += 1
        if calls["exec"] == 1:
            raise RuntimeError("boom")
        return False

    monkeypatch.setattr(helper, "_execute_vllm_request", _exec)
    monkeypatch.setattr(
        helper,
        "_backfill_missing",
        lambda *args: calls.__setitem__("backfill", calls["backfill"] + 1),
    )
    monkeypatch.setattr(
        helper,
        "_record_vllm_failure",
        lambda *args: calls.__setitem__("record", calls["record"] + 1),
    )
    monkeypatch.setattr(module.time, "sleep", lambda *_a, **_k: None)
    state = module._VLLMGenerationState(
        prompts=["p"],
        target_counts=[1],
        requested_n=1,
        round_limit=2,
        track_logprobs=False,
    )
    helper._run_vllm_rounds(state)
    assert calls["exec"] >= 2
    assert calls["backfill"] == 1
    assert calls["record"] == 1


def test_run_vllm_rounds_exits_when_no_pending(monkeypatch):
    module = _load_vllm(monkeypatch)
    helper = module.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    state = module._VLLMGenerationState(
        prompts=["p"],
        target_counts=[0],
        requested_n=1,
        round_limit=3,
        track_logprobs=False,
    )
    helper._run_vllm_rounds(state)
    # No retries because there were no pending prompts.
    assert helper.ctx.generation_stats["vllm_retry_rounds"] == 0


def test_run_vllm_rounds_breaks_after_runtime_error(monkeypatch):
    module = _load_vllm(monkeypatch)
    helper = module.VLLMGenerationHelper(
        _ctx(vllm_backfill_local=False), lambda *_: ([], None)
    )
    monkeypatch.setattr(
        helper,
        "_execute_vllm_request",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("fail")),
    )
    state = module._VLLMGenerationState(
        prompts=["p"],
        target_counts=[1],
        requested_n=1,
        round_limit=1,
        track_logprobs=False,
    )
    with pytest.raises(module.GenerationServiceError) as exc_info:
        helper._run_vllm_rounds(state)
    payload = exc_info.value.payload.to_dict()
    assert payload["prompt_count"] == 1
    assert payload["attempt"] == 1
    assert helper.ctx.generation_stats["vllm_retry_failures"] == 1
    assert helper.ctx.generation_stats["vllm_last_error"]["prompt_count"] == 1


def test_request_vllm_batch_mismatch_returns_none(monkeypatch):
    module = _load_vllm(monkeypatch)
    helper = module.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    monkeypatch.setattr(
        helper, "_invoke_vllm_requests", lambda *_: ([["a"]], None, 1.0)
    )
    grouped, meta = helper._request_vllm_batch(["p1", "p2"], 1)
    assert grouped is None and meta is None

    def _invoke_ok(prompts, request_count):
        return [[p] for p in prompts for _ in range(2)], None, 1.0

    monkeypatch.setattr(helper, "_invoke_vllm_requests", _invoke_ok)
    grouped2, meta2 = helper._request_vllm_batch(["p1", "p2"], 2)
    assert grouped2 == [["p1", "p1"], ["p2", "p2"]]
    assert meta2 is None


def test_request_vllm_batch_returns_none_on_failed_request(monkeypatch):
    module = _load_vllm(monkeypatch)
    helper = module.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    monkeypatch.setattr(helper, "_invoke_vllm_requests", lambda *_a, **_k: None)
    called = {}
    monkeypatch.setattr(
        helper,
        "_record_vllm_latency",
        lambda *_a, **_k: called.setdefault("latency", True),
    )
    grouped, meta = helper._request_vllm_batch(["p"], 1)
    assert grouped is None and meta is None
    assert "latency" not in called


def test_summarize_grouped_mixed_types(monkeypatch):
    module = _load_vllm(monkeypatch)
    summary = module.VLLMGenerationHelper._summarize_grouped([["a"], "b"] * 5, limit=2)
    assert "type=str" in summary or "len=1" in summary
    assert "..." in summary


def test_prepare_targets_and_expand_out_of_range(monkeypatch):
    module = _load_vllm(monkeypatch)
    helper = module.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    monkeypatch.setenv("MAXENT_VLLM_DEDUP", "1")
    prompts, counts, mapping = helper._prepare_vllm_targets(["a", "a"], 1, [1, 2])
    grouped, meta = helper._expand_dedup_results([["x"]], [[None]], mapping)
    assert grouped == [["x"], ["x"]]
    assert meta == [[None], [None]]


def test_backfill_and_failure_logging(monkeypatch):
    module = _load_vllm(monkeypatch)
    ctx = _ctx(vllm_backfill_local=False)
    helper = module.VLLMGenerationHelper(ctx, lambda *_: ([], None))
    state = module._VLLMGenerationState(
        prompts=["p"],
        target_counts=[2],
        requested_n=2,
        round_limit=1,
        track_logprobs=False,
    )
    helper._record_vllm_failure(state, [0])
    assert ctx.generation_stats["vllm_failed_prompts"] == 1


def test_flatten_prompts_with_counts(monkeypatch):
    module = _load_vllm(monkeypatch)

    class _Accel:
        def gather_object(self, value):
            if value and isinstance(value[0], str):
                return [value, ["peer"]]
            return [value, value]

    helper = module.VLLMGenerationHelper(
        _ctx(accelerator=_Accel()), lambda *_: ([], None)
    )
    prompts, offsets, counts = helper._flatten_prompts_for_broadcast(
        ["p0"], per_prompt_counts=[1]
    )
    assert prompts == ["p0", "peer"]
    assert offsets[0] == 0
    assert counts == [1, 1]


def test_gather_object_list_default_and_dist(monkeypatch):
    module = _load_vllm(monkeypatch)
    accel = SimpleNamespace(gather_object=None)
    module.torch.distributed = SimpleNamespace(
        is_available=lambda: False, is_initialized=lambda: False
    )
    assert module._gather_object_list(accel, ["x"]) == [["x"]]

    def _all_gather_object(out, val):
        out[0] = val
        out[1] = ["p"]

    dist = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        get_world_size=lambda: 2,
        all_gather_object=_all_gather_object,
    )
    module.torch.distributed = dist  # type: ignore[attr-defined]
    gathered = module._gather_object_list(SimpleNamespace(), ["z"])
    assert gathered == [["z"], ["p"]]


def test_scatter_object_paths(monkeypatch):
    module = _load_vllm(monkeypatch)
    accel = SimpleNamespace(num_processes=1, process_index=0)
    assert module._scatter_object(accel, None) is None
    assert module._scatter_object(accel, [["x"]]) == ["x"]

    accel2 = SimpleNamespace(num_processes=2, process_index=1, scatter_object=None)
    module.torch.distributed = SimpleNamespace(
        is_available=lambda: False,
        is_initialized=lambda: False,
    )
    assert module._scatter_object(accel2, None) is None
    assert module._scatter_object(accel2, [["a"], ["b"]], src=0) == ["b"]

    accel3 = SimpleNamespace(
        num_processes=2,
        process_index=0,
        scatter_object=lambda payload, src=0: ["scatter"],
    )
    assert module._scatter_object(accel3, [["a"], ["b"]], src=0) == ["scatter"]

    dist = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        scatter_object_list=lambda out, payload, src=0: out.__setitem__(0, ["dist"]),
    )
    module.torch.distributed = dist  # type: ignore[attr-defined]
    assert module._scatter_object(accel2, [["a"], ["b"]], src=0) == ["dist"]


def test_execute_vllm_request_handles_counts_and_none(monkeypatch):
    module = _load_vllm(monkeypatch)
    helper = module.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    state = module._VLLMGenerationState(
        prompts=["p1"],
        target_counts=[0],
        requested_n=1,
        round_limit=1,
        track_logprobs=False,
    )
    assert helper._execute_vllm_request(state, [0]) is True

    state2 = module._VLLMGenerationState(
        prompts=["p1", "p2"],
        target_counts=[1, 1],
        requested_n=1,
        round_limit=1,
        track_logprobs=False,
    )
    helper._request_vllm_batch = lambda prompts, need: (
        [["x"] for _ in prompts],
        [[None] for _ in prompts],
    )
    assert helper._execute_vllm_request(state2, [0, 1]) is True
    state_fail = module._VLLMGenerationState(
        prompts=["p1", "p2"],
        target_counts=[1, 1],
        requested_n=1,
        round_limit=1,
        track_logprobs=False,
    )
    helper._request_vllm_batch = lambda *_args: (None, None)
    assert helper._execute_vllm_request(state_fail, [0, 1]) is False


def test_generate_collective_scatter_multi_rank(monkeypatch):
    module = _load_vllm(monkeypatch)
    scatter_calls = {}

    def _scatter_object(accel, payload, src=0):
        scatter_calls["payload"] = payload
        return payload[accel.process_index]

    monkeypatch.setattr(module, "_scatter_object", _scatter_object)
    accel = SimpleNamespace(
        is_main_process=True,
        process_index=0,
        num_processes=2,
        gather_object=lambda value: (
            [value, ["peer"]] if value and isinstance(value[0], str) else [value, value]
        ),
    )
    helper = module.VLLMGenerationHelper(_ctx(accelerator=accel), lambda *_: ([], None))
    helper.generate = lambda prompts, num_samples, counts: (
        [["g0"], ["g1"]],
        [["m0"], ["m1"]],
    )
    grouped, meta = helper.generate_collective(["p0", "p1"], 1, [1, 1])
    assert grouped == [["g0"], ["g1"]]
    assert meta == [["m0"], ["m1"]]
    assert scatter_calls["payload"][0][0] == [["g0"], ["g1"]]


def test_scatter_vllm_payload_non_main_real(monkeypatch):
    module = _load_vllm(monkeypatch)

    class _Accel:
        num_processes = 2
        process_index = 1
        is_main_process = False

        def gather_object(self, value):
            return [value, ["peer"]]

        def scatter_object(self, payload, src=0):
            return (["local"], None)

    helper = module.VLLMGenerationHelper(
        _ctx(accelerator=_Accel()), lambda *_: ([], None)
    )
    grouped, meta = helper._scatter_vllm_payload(
        flat_prompts=["p0", "peer"],
        offsets=[0, 1],
        grouped_all=None,
        meta_all=None,
    )
    assert grouped == ["local"]
    assert meta is None
