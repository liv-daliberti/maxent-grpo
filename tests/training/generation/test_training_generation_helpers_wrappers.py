"""
Unit tests covering thin wrapper branches in training.rollout.helpers.
"""

from types import SimpleNamespace

import maxent_grpo.training.rollout.helpers as helpers


class _StubAccel:
    def __init__(self, world: int = 1, rank: int = 0):
        self.num_processes = world
        self.process_index = rank
        self.is_main_process = rank == 0

    def unwrap_model(self, model):
        return model


def _make_helper(use_vllm: bool = False) -> helpers.CompletionGenerator:
    tokenizer = SimpleNamespace(
        decode=lambda ids, skip_special_tokens=True: f"dec:{list(ids)}"
    )
    model = SimpleNamespace(generate=lambda **_k: [[1, 2, 3, 4]])
    ctx = SimpleNamespace(
        accelerator=_StubAccel(),
        generation_stats={},
        vllm_request_logprobs=False,
        vllm_sync_weights=False,
        vllm_url="http://localhost",
        prompt_char_limit=None,
        max_prompt_len=8,
        max_completion_len=2,
        gen_temperature=0.7,
        gen_top_p=0.9,
        gen_top_k=None,
        use_vllm=use_vllm,
        tokenizer=tokenizer,
        model=model,
        device=SimpleNamespace(type="cpu"),
    )
    return helpers.CompletionGenerator(ctx)


def test_build_local_prompt_requests_skips_non_positive():
    gen = _make_helper()
    prompts = ["p1", "p2", "p3"]
    expanded, indices = gen._build_local_prompt_requests(prompts, [2, 0, -3])
    assert expanded == ["p1", "p1"]
    assert indices == [0, 0]


def test_run_local_model_fallback_without_generate():
    gen = _make_helper()
    gen.ctx.model = SimpleNamespace()  # no generate attr triggers fallback branch
    encoder_inputs = [[10, 11, 12]]
    out = gen._run_local_model(encoder_inputs, [2])
    # decode should receive slice after prompt length (12)
    assert out == ["dec:[12]"]


def test_summarize_grouped_delegates(monkeypatch):
    called = {}

    def _fake_summary(groups, limit=8):
        called["args"] = (groups, limit)
        return "summary"

    monkeypatch.setattr(
        helpers.VLLMGenerationHelper, "_summarize_grouped", _fake_summary
    )
    res = helpers.CompletionGenerator._summarize_grouped([["a", "b"]], limit=3)
    assert res == "summary"
    assert called["args"] == ([["a", "b"]], 3)


def test_generate_with_vllm_wraps_helper(monkeypatch):
    gen = _make_helper(use_vllm=True)
    called = {}

    class _StubVLLM:
        def __init__(self):
            self._fallback_generate = None

        def generate(self, prompts, n, counts, ensure_client=None, sync_model=None):
            called["args"] = (prompts, n, counts)
            called["ensure"] = ensure_client
            called["sync"] = sync_model
            return [["x"]], None

        def set_fallback_generate(self, fn):
            self._fallback_generate = fn

    stub = _StubVLLM()
    gen._vllm_helper = stub
    grouped, meta = gen._generate_with_vllm(["p"], 1, None)
    assert grouped == [["x"]] and meta is None
    assert called["args"] == (["p"], 1, None)
    assert callable(called["ensure"]) and callable(called["sync"])


def test_scatter_vllm_payload_wrapper(monkeypatch):
    gen = _make_helper()
    marker = object()

    class _StubVLLM:
        def _scatter_vllm_payload(self, *args):
            return marker, None

    gen._vllm_helper = _StubVLLM()
    grouped, meta = gen._scatter_vllm_payload([], [], None, None)
    assert grouped is marker and meta is None


def test_generate_raises_on_per_prompt_mismatch():
    gen = _make_helper()
    try:
        gen.generate(["a", "b"], num_samples=1, per_prompt_counts=[1])
    except ValueError:
        return
    assert False, "Expected ValueError for per_prompt_counts length mismatch"


def test_refresh_vllm_globals_updates_adapter(monkeypatch):
    helpers = __import__("maxent_grpo.training.rollout.helpers", fromlist=["dist"])
    marker_dist = object()
    marker_safe = object()
    retry_fn = lambda *_a, **_k: "retry"  # - simple stub

    monkeypatch.setattr(helpers, "dist", marker_dist)
    monkeypatch.setattr(helpers, "safe_generate", marker_safe)
    monkeypatch.setattr(helpers, "_retry_incomplete_prompts", retry_fn)
    monkeypatch.setattr(helpers._vllm_adapter, "dist", None)
    monkeypatch.setattr(helpers._vllm_adapter, "safe_generate", None)

    helpers._refresh_vllm_globals()

    assert helpers._vllm_adapter.dist is marker_dist
    assert helpers._vllm_adapter.safe_generate is marker_safe
    assert helpers._retry_incomplete_prompts_impl is retry_fn


def test_scatter_object_fallback_returns_local_entry():
    accel = _StubAccel(world=3, rank=1)
    result = helpers._scatter_object(accel, ["r0", "r1", "r2"])
    assert result == "r1"
