"""Coverage for vLLM adapter fallback wiring in _generate_with_vllm."""

from __future__ import annotations

from types import SimpleNamespace

import maxent_grpo.training.generation.vllm_adapter as vllm_adapter


def test_generate_with_vllm_sets_fallback_and_handles_missing_generate(monkeypatch):
    ctx = SimpleNamespace(
        prompt_char_limit=None,
        accelerator=SimpleNamespace(),
        generation_stats={},
        vllm_sync_weights=False,
    )

    class _Helper:
        def __init__(self):
            self._fallback_generate = None
            self.generate = None
            self._generate_called = False

    gen = vllm_adapter.VLLMGenerationMixin.__new__(vllm_adapter.VLLMGenerationMixin)
    gen.ctx = ctx
    helper = _Helper()
    gen._vllm_helper = helper
    gen._generate_local = lambda *a, **k: ("local", a, k)
    gen._prompt_char_limit = lambda: 7

    grouped, meta = gen._generate_with_vllm(["p"], 1)
    assert grouped == []
    assert meta is None
    # Fallback set even when generate is missing.
    assert helper._fallback_generate is gen._generate_local


def test_generate_with_vllm_calls_helper_generate(monkeypatch):
    ctx = SimpleNamespace(
        prompt_char_limit=None,
        accelerator=SimpleNamespace(),
        generation_stats={},
        vllm_sync_weights=False,
    )

    class _Helper:
        def __init__(self):
            self._fallback_generate = None
            self.called_with = None

        def generate(self, prompts, num_samples, counts, **kwargs):
            self.called_with = (prompts, num_samples, counts, kwargs)
            return [["ok"]], None

    gen = vllm_adapter.VLLMGenerationMixin.__new__(vllm_adapter.VLLMGenerationMixin)
    gen.ctx = ctx
    helper = _Helper()
    gen._vllm_helper = helper
    gen._generate_local = lambda *a, **k: ("local", a, k)
    gen._prompt_char_limit = lambda: 5
    gen._sync_model_params_to_vllm = lambda model, accel: ("sync", model, accel)
    gen._ensure_vllm_client = lambda: True

    grouped, meta = gen._generate_with_vllm(["p1", "p2"], 2, [1, 2])
    assert grouped == [["ok"]]
    assert meta is None
    assert helper._fallback_generate is gen._generate_local
    prompts, num_samples, counts, kwargs = helper.called_with
    assert prompts == ["p1", "p2"]
    assert num_samples == 2
    assert counts == [1, 2]
    assert callable(kwargs["sync_model"])
