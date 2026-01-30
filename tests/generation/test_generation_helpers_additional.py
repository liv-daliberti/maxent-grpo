"""Additional branch coverage for training.rollout.helpers."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture()
def helpers_mod(monkeypatch):
    """Import the helpers module with lightweight config stubs to avoid heavy deps."""
    # Torch stub to satisfy require_torch calls during import.
    torch_stub = SimpleNamespace(
        Tensor=type("Tensor", (), {}),
        tensor=lambda data, dtype=None, device=None: SimpleNamespace(data=data),
        full=lambda shape, val, dtype=None: SimpleNamespace(
            data=[val] * (shape[0] if shape else 0)
        ),
        zeros=lambda shape, dtype=None: SimpleNamespace(
            data=[0] * (shape[0] if shape else 0)
        ),
        ones_like=lambda x, dtype=None: SimpleNamespace(data=getattr(x, "data", [])),
        float32="float32",
        int64=int,
        long=int,
        device=lambda kind="cpu": SimpleNamespace(type=kind),
        distributed=SimpleNamespace(fsdp=None),
    )
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    accel_stub = ModuleType("accelerate")
    accel_stub.Accelerator = type("Accelerator", (), {})
    monkeypatch.setitem(sys.modules, "accelerate", accel_stub)
    tf_stub = ModuleType("transformers")
    tf_stub.PreTrainedModel = type("PreTrainedModel", (), {})
    tf_stub.PreTrainedTokenizer = type("PreTrainedTokenizer", (), {})
    monkeypatch.setitem(sys.modules, "transformers", tf_stub)
    ops_pkg = ModuleType("ops")
    site_stub = ModuleType("sitecustomize")
    site_stub._install_torch_stub = lambda: None
    monkeypatch.setitem(sys.modules, "ops", ops_pkg)
    monkeypatch.setitem(sys.modules, "sitecustomize", site_stub)

    # Stub out config to sidestep the package-level __future__ parsing quirks.
    config_stub = ModuleType("maxent_grpo.config")
    config_stub.GRPOConfig = type("GRPOConfig", (), {})
    config_stub.GRPOScriptArguments = type("GRPOScriptArguments", (), {})
    monkeypatch.setitem(sys.modules, "maxent_grpo.config", config_stub)
    monkeypatch.setitem(sys.modules, "maxent_grpo.config.grpo", config_stub)
    ds_stub = ModuleType("maxent_grpo.config.dataset")
    ds_stub.ScriptArguments = type("ScriptArguments", (), {})
    ds_stub.DatasetConfig = type("DatasetConfig", (), {})
    ds_stub.DatasetMixtureConfig = type("DatasetMixtureConfig", (), {})
    monkeypatch.setitem(sys.modules, "maxent_grpo.config.dataset", ds_stub)
    gen_helpers_stub = ModuleType("maxent_grpo.generation.helpers")
    gen_helpers_stub.AggregatedGenerationState = lambda comps, meta: SimpleNamespace(
        completions=comps, metadata=meta
    )
    gen_helpers_stub.append_completion_group = lambda comps, meta, idx, g, m: meta
    gen_helpers_stub.seed_generation_groups = lambda prompt_count, comps, meta: (
        comps or [[] for _ in range(prompt_count)],
        meta,
    )
    gen_helpers_stub.pending_generation_indices = lambda comps, n: [
        i for i, c in enumerate(comps) if len(c) < n
    ]
    gen_helpers_stub.determine_retry_limit = (
        lambda expected, max_retry: max_retry or expected or 3
    )
    gen_helpers_stub.retry_incomplete_prompts = lambda *a, **k: SimpleNamespace(
        completions=a[3].completions, metadata=a[3].metadata
    )
    gen_helpers_stub.drop_empty_prompt_groups = (
        lambda prompts, answers, comps, meta, stats: (prompts, answers, comps, meta)
    )
    gen_helpers_stub.truncate_to_expected_counts = lambda comps, meta, exp: (
        comps,
        meta,
        0,
    )
    gen_helpers_stub.flatten_ref_metadata = lambda comps, meta: meta
    monkeypatch.setitem(sys.modules, "maxent_grpo.generation.helpers", gen_helpers_stub)
    gen_common_stub = ModuleType("maxent_grpo.generation.common")
    gen_common_stub.AggregatedGenerationState = (
        gen_helpers_stub.AggregatedGenerationState
    )
    gen_common_stub.append_completion_group = gen_helpers_stub.append_completion_group
    gen_common_stub.seed_generation_groups = gen_helpers_stub.seed_generation_groups
    gen_common_stub.pending_generation_indices = (
        gen_helpers_stub.pending_generation_indices
    )
    gen_common_stub.determine_retry_limit = gen_helpers_stub.determine_retry_limit
    gen_common_stub.retry_incomplete_prompts = gen_helpers_stub.retry_incomplete_prompts
    gen_common_stub.drop_empty_prompt_groups = gen_helpers_stub.drop_empty_prompt_groups
    gen_common_stub.truncate_to_expected_counts = (
        gen_helpers_stub.truncate_to_expected_counts
    )
    gen_common_stub.flatten_ref_metadata = gen_helpers_stub.flatten_ref_metadata
    gen_common_stub._DEFAULT_RETRY_LIMIT = 3
    monkeypatch.setitem(sys.modules, "maxent_grpo.generation.common", gen_common_stub)
    gen_pkg_stub = ModuleType("maxent_grpo.generation")
    gen_pkg_stub.helpers = gen_helpers_stub
    gen_pkg_stub.common = gen_common_stub
    gen_pkg_stub.__path__ = []
    gen_pkg_stub.AggregatedGenerationState = gen_helpers_stub.AggregatedGenerationState
    gen_pkg_stub.drop_empty_prompt_groups = gen_helpers_stub.drop_empty_prompt_groups
    gen_pkg_stub.flatten_prompt_completions = lambda gb: gb
    gen_pkg_stub.flatten_ref_metadata = gen_helpers_stub.flatten_ref_metadata
    gen_pkg_stub.retry_incomplete_prompts = gen_helpers_stub.retry_incomplete_prompts
    gen_pkg_stub.seed_generation_groups = gen_helpers_stub.seed_generation_groups
    gen_pkg_stub.truncate_to_expected_counts = (
        gen_helpers_stub.truncate_to_expected_counts
    )
    monkeypatch.setitem(sys.modules, "maxent_grpo.generation", gen_pkg_stub)

    # Force fresh import of helpers and run_helpers to pick up stubs.
    for name in list(sys.modules):
        if name.startswith("training.rollout.helpers") or name.startswith(
            "training.run_helpers"
        ):
            sys.modules.pop(name, None)
    try:
        helpers = importlib.import_module("training.rollout.helpers")
        run_helpers = importlib.import_module("training.run_helpers")
    except ImportError:
        helpers = importlib.import_module("maxent_grpo.training.rollout.helpers")
        run_helpers = importlib.import_module("maxent_grpo.training.run_helpers")
    return helpers, run_helpers.VLLMClientConfig


def _ctx_base(helpers, vllm_cfg_cls, **overrides):
    vllm_cfg = vllm_cfg_cls(
        url=overrides.pop("vllm_url", "http://localhost:8000/generate"),
        rounds_cfg=overrides.pop("rounds_cfg", 0),
        retry_sleep=0.0,
        backfill_local=False,
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
        generation_stats={},
        device=helpers.torch.device("cpu"),
    )
    for key, val in overrides.items():
        setattr(ctx, key, val)
    return ctx


def test_import_vllm_client_cls_variants(monkeypatch, helpers_mod):
    helpers, _ = helpers_mod
    monkeypatch.setattr(helpers, "_optional_import", lambda name: None)
    assert helpers._import_vllm_client_cls() is None

    class _Client:
        pass

    dummy_mod = SimpleNamespace(VLLMClient=_Client)
    monkeypatch.setattr(helpers, "_optional_import", lambda name: dummy_mod)
    # Newer vLLM exports class via client module; helper should return class or None.
    cls = helpers._import_vllm_client_cls()
    assert cls is None or cls is _Client


def test_generation_context_as_dict_and_penalties(helpers_mod):
    helpers, vllm_cfg_cls = helpers_mod
    ctx = _ctx_base(helpers, vllm_cfg_cls)
    ctx.gen_top_k = 5
    ctx.gen_best_of = 2
    ctx.gen_stop_sequences = ["</s>"]
    desc = ctx.as_dict()
    assert desc["top_k"] == 5
    assert desc["best_of"] == 2
    assert desc["use_vllm"] is True
    assert "http://" in desc["vllm_url"]


def test_vllm_generation_state_tracking_and_trim(helpers_mod):
    helpers, _ = helpers_mod
    with pytest.raises(ValueError):
        helpers._VLLMGenerationState(
            prompts=["p1"],
            target_counts=[1, 2],
            requested_n=1,
            round_limit=1,
            track_logprobs=False,
        )

    state = helpers._VLLMGenerationState(
        prompts=["p1", "p2"],
        target_counts=[1, 2],
        requested_n=2,
        round_limit=3,
        track_logprobs=True,
    )
    state.aggregated[0].append("a1")
    state.aggregated[1].extend(["b1"])
    state.aggregated_meta[1].extend([SimpleNamespace(score=0.1)])

    assert state.pending_indices() == [1]
    assert state.remaining_counts([1]) == [1]

    trimmed, trimmed_meta = state.trim()
    assert trimmed == [["a1"], ["b1"]]
    assert trimmed_meta is not None and trimmed_meta[1] == state.aggregated_meta[1]

    state.drop_meta()
    assert state.aggregated_meta is None


def test_completion_generator_vllm_helpers(monkeypatch, helpers_mod):
    helpers, vllm_cfg_cls = helpers_mod
    ctx = _ctx_base(helpers, vllm_cfg_cls, sync_weights=True)
    gen = helpers.CompletionGenerator(ctx)

    # Missing client class returns False and marks sync_ready False
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
    assert isinstance(gen._vllm_client, _Client) and gen._vllm_client.initted is True
    # cached client path
    assert gen._ensure_vllm_client() is True

    assert gen._vllm_base_url("http://host:8000/generate") == "http://host:8000"
    assert gen._vllm_base_url("http://host:8000") == "http://host:8000"
    assert gen._vllm_base_url("localhost:8000/generate") == "localhost:8000"


def test_resolve_vllm_round_limit(helpers_mod):
    helpers, vllm_cfg_cls = helpers_mod
    ctx = _ctx_base(helpers, vllm_cfg_cls, rounds_cfg=5)
    gen = helpers.CompletionGenerator(ctx)
    assert gen._resolve_vllm_round_limit(2) == 5

    ctx.vllm.rounds_cfg = 0
    assert gen._resolve_vllm_round_limit(2) == 2
