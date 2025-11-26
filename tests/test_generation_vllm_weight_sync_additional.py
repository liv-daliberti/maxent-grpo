"""Targeted tests for vllm_weight_sync edge paths."""

from __future__ import annotations

from types import SimpleNamespace

from maxent_grpo.generation.vllm_weight_sync import VLLMWeightSyncMixin


class _MixinUnderTest(VLLMWeightSyncMixin):
    def __init__(self, ctx):
        self.ctx = ctx
        self._vllm_client = None
        self._vllm_sync_ready = False
        self._last_vllm_synced_step = None
        self._fsdp_cls = None
        self._gather_factory = None
        self._is_peft_model_safe = lambda _m: False

    def _vllm_base_url(self, url):
        return url

    def _push_param_to_vllm(self, name, param):
        calls = self.ctx.generation_stats.setdefault("push_calls", [])
        calls.append((name, param))

    def _reset_vllm_cache(self):
        self.ctx.generation_stats["reset"] = True

    def _sync_peft_params(self, model):
        self.ctx.generation_stats["peft"] = model

    def _sync_standard_params(self, model):
        self.ctx.generation_stats["standard"] = model


def test_ensure_vllm_client_handles_init_signatures(monkeypatch):
    ctx = SimpleNamespace(
        accelerator=SimpleNamespace(is_main_process=True),
        vllm_url="http://host",
        vllm_sync_weights=True,
        generation_stats={},
    )
    mixin = _MixinUnderTest(ctx)

    class _Client:
        def __init__(self, base_url=None):
            self.base_url = base_url
            self.init_calls = []

        def init_communicator(self, *args):
            self.init_calls.append(args)

    mixin._import_vllm_client_cls = lambda: _Client
    assert mixin._ensure_vllm_client() is True
    assert isinstance(mixin._vllm_client, _Client)
    assert mixin._vllm_sync_ready is True
    assert mixin._vllm_client.base_url == ctx.vllm_url
    assert mixin._vllm_client.init_calls == [tuple()]


def test_maybe_sync_weights_respects_step_and_stats(monkeypatch):
    ctx = SimpleNamespace(
        accelerator=SimpleNamespace(
            unwrap_model=lambda m: m, wait_for_everyone=lambda: None
        ),
        vllm_url="http://host",
        vllm_sync_weights=True,
        generation_stats={"current_step": 1},
        model="model",
    )
    mixin = _MixinUnderTest(ctx)
    mixin._vllm_sync_ready = True
    mixin._last_vllm_synced_step = 0
    # Guard against NameError in maybe_sync_weights when inspecting the callable.
    monkeypatch.setattr(
        __import__(
            "maxent_grpo.generation.vllm_weight_sync", fromlist=["SimpleNamespace"]
        ),
        "SimpleNamespace",
        SimpleNamespace,
    )

    called = {}

    def _sync_model(model, visited=None):
        called["model"] = model
        called["visited"] = visited

    mixin._sync_model_params_to_vllm = _sync_model  # type: ignore[assignment]
    mixin.maybe_sync_weights(ensure_client=lambda: True)

    assert called["model"] == "model"
    assert isinstance(called["visited"], set)
    assert ctx.generation_stats["vllm_weight_syncs"] == 1
    assert mixin._last_vllm_synced_step == 1

    # Second call with same step should short-circuit sync.
    called.clear()
    mixin.maybe_sync_weights(ensure_client=lambda: True)
    assert called == {}


def test_sync_model_params_to_vllm_handles_fsdp_and_peft(monkeypatch):
    ctx = SimpleNamespace(
        generation_stats={},
        vllm_url="http://host",
        vllm_sync_weights=True,
    )
    mixin = _MixinUnderTest(ctx)

    class _Param:
        def __init__(self, name):
            self.name = name
            self.data = name

    class _FsdpModule:
        def __init__(self):
            self.children_called = False

        def named_children(self):
            self.children_called = True
            return [("child", _Leaf())]

    class _Leaf:
        def named_parameters(self):
            return [("p", _Param("fsdp_param"))]

    class _FsdpNoParams:
        def named_children(self):
            return [("child", SimpleNamespace(named_parameters=None))]

    mixin._fsdp_cls = _FsdpModule
    mixin._sync_model_params_to_vllm(_FsdpModule())
    assert ctx.generation_stats.get("reset") is True

    ctx.generation_stats.clear()

    mixin._fsdp_cls = _FsdpModule
    mixin._sync_model_params_to_vllm(_FsdpModule(), visited={"child.p"})
    assert ctx.generation_stats.get("push_calls") is None
    assert ctx.generation_stats.get("reset") is True

    ctx.generation_stats.clear()

    mixin._fsdp_cls = _FsdpNoParams
    mixin._sync_model_params_to_vllm(_FsdpNoParams())
    assert ctx.generation_stats.get("push_calls") is None
    assert ctx.generation_stats.get("reset") is True

    ctx.generation_stats.clear()

    class _PeftModel:
        pass

    mixin._is_peft_model_safe = lambda model: isinstance(model, _PeftModel)  # type: ignore[assignment]
    mixin._sync_model_params_to_vllm(_PeftModel())
    assert ctx.generation_stats["peft"].__class__ is _PeftModel

    ctx.generation_stats.clear()

    mixin._is_peft_model_safe = lambda _m: False  # type: ignore[assignment]
    mixin._sync_model_params_to_vllm(SimpleNamespace())
    assert ctx.generation_stats["standard"] is not None
