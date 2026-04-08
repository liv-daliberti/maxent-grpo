"""Coverage for colocated vLLM helpers used by the training rollout."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
import sys

import pytest

import maxent_grpo.training.rollout.vllm_adapter as vllm_adapter
import maxent_grpo.training.rollout.vllm_colocate as vllm_colocate


def _install_stub_vllm(monkeypatch, outputs=None):
    mod = ModuleType("vllm")

    class DummyLLM:
        last_init = None
        last_params = None

        def __init__(
            self,
            model,
            dtype=None,
            gpu_memory_utilization=None,
            tensor_parallel_size=None,
            max_model_len=None,
            trust_remote_code=None,
        ):
            DummyLLM.last_init = {
                "model": model,
                "dtype": dtype,
                "gpu_memory_utilization": gpu_memory_utilization,
                "tensor_parallel_size": tensor_parallel_size,
                "max_model_len": max_model_len,
                "trust_remote_code": trust_remote_code,
            }

        def generate(self, prompts, params):
            DummyLLM.last_params = getattr(params, "kwargs", None)
            if outputs is not None:
                return outputs
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            text=f"{prompt}-out",
                            cumulative_logprob=None,
                            output_token_ids=[1],
                            logprobs=[{"logprob": -0.1}],
                        )
                    ]
                )
                for prompt in prompts
            ]

    class DummySamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    mod.LLM = DummyLLM
    mod.SamplingParams = DummySamplingParams
    monkeypatch.setattr(
        vllm_colocate, "optional_import", lambda name: mod if name == "vllm" else None
    )
    return mod, DummyLLM


def test_configure_vllm_mode_colocate_preserves_configured_sync(monkeypatch):
    monkeypatch.delenv("MAXENT_VLLM_COLOCATE_SYNC", raising=False)

    class _Helper:
        def __init__(self):
            self.batcher = None

        def set_request_batcher(self, fn):
            self.batcher = fn

    class _Engine:
        def __init__(self, ctx, fallback):
            self.ctx = ctx
            self.fallback = fallback
            self.client = object()

        def request_batch(self, prompts, count):
            return [], None

        def sync_client(self):
            return self.client

    mod = ModuleType("maxent_grpo.training.rollout.vllm_colocate")
    mod.ColocateVLLMEngine = _Engine
    monkeypatch.setitem(sys.modules, mod.__name__, mod)

    gen = vllm_adapter.VLLMGenerationMixin.__new__(vllm_adapter.VLLMGenerationMixin)
    gen.ctx = SimpleNamespace(
        use_vllm=True, vllm_mode="colocate", vllm_sync_weights=True
    )
    gen._vllm_helper = _Helper()
    gen._generate_local = lambda *a, **k: ("local", a, k)
    gen._vllm_colocate_engine = None

    gen._configure_vllm_mode()

    assert gen.ctx.vllm_sync_weights is True
    assert gen.ctx.vllm_sync_interval_steps == 10
    assert gen._vllm_client is gen._vllm_colocate_engine.client
    assert gen._vllm_sync_ready is True
    assert isinstance(gen._vllm_colocate_engine, _Engine)
    assert gen._vllm_helper.batcher == gen._vllm_colocate_engine.request_batch


def test_configure_vllm_mode_colocate_env_can_disable_sync(monkeypatch):
    monkeypatch.setenv("MAXENT_VLLM_COLOCATE_SYNC", "0")

    class _Helper:
        def __init__(self):
            self.batcher = None

        def set_request_batcher(self, fn):
            self.batcher = fn

    class _Engine:
        def __init__(self, ctx, fallback):
            self.ctx = ctx
            self.fallback = fallback

        def request_batch(self, prompts, count):
            return [], None

    mod = ModuleType("maxent_grpo.training.rollout.vllm_colocate")
    mod.ColocateVLLMEngine = _Engine
    monkeypatch.setitem(sys.modules, mod.__name__, mod)

    gen = vllm_adapter.VLLMGenerationMixin.__new__(vllm_adapter.VLLMGenerationMixin)
    gen.ctx = SimpleNamespace(
        use_vllm=True, vllm_mode="colocate", vllm_sync_weights=True
    )
    gen._vllm_helper = _Helper()
    gen._generate_local = lambda *a, **k: ("local", a, k)
    gen._vllm_colocate_engine = None

    gen._configure_vllm_mode()

    assert gen.ctx.vllm_sync_weights is False
    assert isinstance(gen._vllm_colocate_engine, _Engine)
    assert gen._vllm_helper.batcher == gen._vllm_colocate_engine.request_batch


def test_configure_vllm_mode_colocate_enables_sync(monkeypatch):
    monkeypatch.setenv("MAXENT_VLLM_COLOCATE_SYNC", "1")
    monkeypatch.delenv("MAXENT_VLLM_COLOCATE_SYNC_INTERVAL", raising=False)

    class _Helper:
        def __init__(self):
            self.batcher = None

        def set_request_batcher(self, fn):
            self.batcher = fn

    class _Engine:
        def __init__(self, ctx, fallback):
            self.ctx = ctx
            self.fallback = fallback
            self.client = object()

        def request_batch(self, prompts, count):
            return [], None

        def sync_client(self):
            return self.client

    mod = ModuleType("maxent_grpo.training.rollout.vllm_colocate")
    mod.ColocateVLLMEngine = _Engine
    monkeypatch.setitem(sys.modules, mod.__name__, mod)

    gen = vllm_adapter.VLLMGenerationMixin.__new__(vllm_adapter.VLLMGenerationMixin)
    gen.ctx = SimpleNamespace(
        use_vllm=True,
        vllm_mode="colocate",
        vllm_sync_weights=False,
        vllm_sync_interval_steps=1,
    )
    gen._vllm_helper = _Helper()
    gen._generate_local = lambda *a, **k: ("local", a, k)
    gen._vllm_colocate_engine = None

    gen._configure_vllm_mode()

    assert gen.ctx.vllm_sync_weights is True
    assert gen.ctx.vllm_sync_interval_steps == 1
    assert gen._vllm_client is gen._vllm_colocate_engine.client
    assert gen._vllm_sync_ready is True
    assert gen._vllm_helper.batcher == gen._vllm_colocate_engine.request_batch


def test_configure_vllm_mode_colocate_sync_interval_override(monkeypatch):
    monkeypatch.setenv("MAXENT_VLLM_COLOCATE_SYNC", "1")
    monkeypatch.setenv("MAXENT_VLLM_COLOCATE_SYNC_INTERVAL", "3")

    class _Helper:
        def __init__(self):
            self.batcher = None

        def set_request_batcher(self, fn):
            self.batcher = fn

    class _Engine:
        def __init__(self, ctx, fallback):
            self.ctx = ctx
            self.fallback = fallback

        def request_batch(self, prompts, count):
            return [], None

        def sync_client(self):
            return object()

    mod = ModuleType("maxent_grpo.training.rollout.vllm_colocate")
    mod.ColocateVLLMEngine = _Engine
    monkeypatch.setitem(sys.modules, mod.__name__, mod)

    gen = vllm_adapter.VLLMGenerationMixin.__new__(vllm_adapter.VLLMGenerationMixin)
    gen.ctx = SimpleNamespace(
        use_vllm=True,
        vllm_mode="colocate",
        vllm_sync_weights=False,
        vllm_sync_interval_steps=1,
    )
    gen._vllm_helper = _Helper()
    gen._generate_local = lambda *a, **k: ("local", a, k)
    gen._vllm_colocate_engine = None

    gen._configure_vllm_mode()

    assert gen.ctx.vllm_sync_weights is True
    assert gen.ctx.vllm_sync_interval_steps == 3


def test_configure_vllm_mode_colocate_preserves_interval(monkeypatch):
    monkeypatch.setenv("MAXENT_VLLM_COLOCATE_SYNC", "1")
    monkeypatch.delenv("MAXENT_VLLM_COLOCATE_SYNC_INTERVAL", raising=False)

    class _Helper:
        def __init__(self):
            self.batcher = None

        def set_request_batcher(self, fn):
            self.batcher = fn

    class _Engine:
        def __init__(self, ctx, fallback):
            self.ctx = ctx
            self.fallback = fallback

        def request_batch(self, prompts, count):
            return [], None

        def sync_client(self):
            return object()

    mod = ModuleType("maxent_grpo.training.rollout.vllm_colocate")
    mod.ColocateVLLMEngine = _Engine
    monkeypatch.setitem(sys.modules, mod.__name__, mod)

    gen = vllm_adapter.VLLMGenerationMixin.__new__(vllm_adapter.VLLMGenerationMixin)
    gen.ctx = SimpleNamespace(
        use_vllm=True,
        vllm_mode="colocate",
        vllm_sync_weights=False,
        vllm_sync_interval_steps=5,
    )
    gen._vllm_helper = _Helper()
    gen._generate_local = lambda *a, **k: ("local", a, k)
    gen._vllm_colocate_engine = None

    gen._configure_vllm_mode()

    assert gen.ctx.vllm_sync_weights is True
    assert gen.ctx.vllm_sync_interval_steps == 5


def test_colocate_build_llm_uses_env_overrides(monkeypatch):
    _install_stub_vllm(monkeypatch)
    monkeypatch.setenv("MAXENT_VLLM_COLOCATE_GPU_UTIL", "0.75")
    monkeypatch.setenv("MAXENT_VLLM_COLOCATE_TP", "2")
    monkeypatch.setenv("MAXENT_VLLM_COLOCATE_MAX_MODEL_LEN", "1024")

    ctx = SimpleNamespace(
        model_name_or_path="model-id",
        training_args=SimpleNamespace(fp16=True, bf16=False, trust_remote_code=True),
    )
    engine = vllm_colocate.ColocateVLLMEngine(ctx, lambda *a, **k: ([], None))
    _ = engine._get_llm()

    init = engine._llm.last_init
    assert init["model"] == "model-id"
    assert init["dtype"] == "float16"
    assert init["gpu_memory_utilization"] == 0.75
    assert init["tensor_parallel_size"] == 2
    assert init["max_model_len"] == 1024
    assert init["trust_remote_code"] is True


def test_colocate_request_batch_falls_back_without_vllm(monkeypatch):
    monkeypatch.setattr(vllm_colocate, "optional_import", lambda name: None)
    seen = {}

    def _fallback(prompts, count, _counts):
        seen["prompts"] = prompts
        seen["count"] = count
        return [["ok"]], None

    ctx = SimpleNamespace(prompt_char_limit=4)
    engine = vllm_colocate.ColocateVLLMEngine(ctx, _fallback)
    grouped, meta = engine.request_batch(["abcdefgh"], 1)

    assert grouped == [["ok"]]
    assert meta is None
    assert seen["prompts"] == ["abcd"]
    assert seen["count"] == 1


def test_colocate_request_batch_parses_logprobs_and_latency(monkeypatch):
    outputs = [
        SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    text="out",
                    cumulative_logprob=None,
                    output_token_ids=[7, 8],
                    logprobs=[{"logprob": -0.2}, {"logprob": -0.3}],
                )
            ]
        )
    ]
    _install_stub_vllm(monkeypatch, outputs=outputs)
    times = iter([1.0, 1.25])
    monkeypatch.setattr(
        vllm_colocate, "time", SimpleNamespace(time=lambda: next(times))
    )

    ctx = SimpleNamespace(
        prompt_char_limit=10,
        gen_temperature=0.5,
        gen_top_p=0.9,
        gen_top_k=None,
        gen_best_of=None,
        gen_frequency_penalty=0.0,
        gen_presence_penalty=0.0,
        max_completion_len=5,
        vllm_request_logprobs=True,
        vllm_stop_sequences=None,
        vllm_logit_bias=None,
        vllm_guided_json=None,
        vllm_guided_regex=None,
        generation_stats={},
    )
    engine = vllm_colocate.ColocateVLLMEngine(ctx, lambda *a, **k: ([], None))
    grouped, meta = engine.request_batch(["prompt"], 1)

    assert grouped == [["out"]]
    assert meta is not None
    entry = meta[0][0]
    assert entry.logprob_sum == pytest.approx(-0.5)
    assert entry.token_count == 2
    assert entry.token_logprobs == [-0.2, -0.3]
    stats = ctx.generation_stats
    assert stats["vllm_latency_calls"] == 1
    assert stats["vllm_last_latency_ms"] == pytest.approx(250.0)


def test_colocate_build_sampling_params_applies_vocab_guard(monkeypatch):
    _, dummy_llm = _install_stub_vllm(monkeypatch)
    monkeypatch.setattr(
        vllm_colocate,
        "merge_invalid_token_block_logit_bias",
        lambda ctx, bias: {"151665": -100.0},
    )
    monkeypatch.setattr(
        vllm_colocate,
        "resolve_allowed_token_ids",
        lambda ctx: [0, 1, 2],
    )
    monkeypatch.setattr(
        vllm_colocate,
        "resolve_blocked_token_ids",
        lambda ctx: [5, 6],
    )

    ctx = SimpleNamespace(
        prompt_char_limit=10,
        gen_temperature=0.5,
        gen_top_p=0.9,
        gen_top_k=None,
        gen_best_of=None,
        gen_frequency_penalty=0.0,
        gen_presence_penalty=0.0,
        max_completion_len=5,
        vllm_request_logprobs=False,
        vllm_stop_sequences=None,
        vllm_logit_bias=None,
        vllm_guided_json=None,
        vllm_guided_regex=None,
        generation_stats={},
    )
    engine = vllm_colocate.ColocateVLLMEngine(ctx, lambda *a, **k: ([], None))
    grouped, meta = engine.request_batch(["prompt"], 1)

    assert grouped == [["prompt-out"]]
    assert meta is not None
    assert dummy_llm.last_params is not None
    assert dummy_llm.last_params["logit_bias"] == {"151665": -100.0}
    assert dummy_llm.last_params["allowed_token_ids"] == [0, 1, 2]
    assert dummy_llm.last_params["_bad_words_token_ids"] == [[5], [6]]


def test_colocate_forwards_include_stop_str_in_output(monkeypatch):
    _mod, dummy_llm = _install_stub_vllm(monkeypatch)

    ctx = SimpleNamespace(
        prompt_char_limit=10,
        gen_temperature=0.5,
        gen_top_p=0.9,
        gen_top_k=None,
        gen_best_of=None,
        gen_frequency_penalty=0.0,
        gen_presence_penalty=0.0,
        max_completion_len=5,
        vllm_request_logprobs=False,
        vllm_stop_sequences=["</answer>"],
        vllm_include_stop_str_in_output=True,
        vllm_logit_bias=None,
        vllm_guided_json=None,
        vllm_guided_regex=None,
        generation_stats={},
    )
    engine = vllm_colocate.ColocateVLLMEngine(ctx, lambda *a, **k: ([], None))
    grouped, meta = engine.request_batch(["prompt"], 1)

    assert grouped == [["prompt-out"]]
    assert meta is not None
    assert dummy_llm.last_params is not None
    assert dummy_llm.last_params["stop"] == ["</answer>"]
    assert dummy_llm.last_params["include_stop_str_in_output"] is True


def test_colocate_sync_client_buffers_and_flushes(monkeypatch):
    class _Engine:
        def __init__(self):
            self.updates = []
            self.reset_calls = 0

        def _apply_param_updates(self, updates):
            self.updates.append(list(updates))

        def _reset_prefix_cache(self):
            self.reset_calls += 1

    class _Param:
        def __init__(self, size):
            self._size = size

        def detach(self):
            return self

        def numel(self):
            return self._size

        def element_size(self):
            return 1

    engine = _Engine()
    client = vllm_colocate.ColocateVLLMClient(engine)
    client._chunk_bytes = 10

    client.update_named_param("a", _Param(6))
    assert engine.updates == []

    client.update_named_param("b", _Param(6))
    assert len(engine.updates) == 1
    assert engine.updates[0][0][0] == "a"

    client.flush()
    assert len(engine.updates) == 2
    assert engine.updates[1][0][0] == "b"

    client.reset_prefix_cache()
    assert engine.reset_calls == 1
