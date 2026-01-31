"""Additional coverage for training.rollout.vllm_adapter."""

from __future__ import annotations

from types import SimpleNamespace


import maxent_grpo.training.rollout.vllm_adapter as vllm_adapter


class _DummyHelper:
    def __init__(self, ctx, _fallback):
        self.ctx = ctx
        self._vllm_client = None
        self._vllm_sync_ready = False
        self._fallback_generate = None
        self._safe_generate = None
        self._time = None
        self._stored_base = None

    def set_safe_generate(self, fn):
        self._safe_generate = fn

    def set_time_provider(self, provider):
        self._time = provider

    def set_fallback_generate(self, fn):
        self._fallback_generate = fn

    def _vllm_base_url(self, url: str) -> str:
        self._stored_base = url
        return f"norm:{url}"

    def _invoke_vllm_requests(self, prompts, request_count):
        return (prompts, request_count, self._safe_generate, self._time)

    def _run_vllm_rounds(self, state):
        self._run_state = state


class _DummyGenerator(vllm_adapter.VLLMGenerationMixin):
    def __init__(self, ctx, helper_cls=_DummyHelper):
        self.ctx = ctx
        self._vllm_helper = helper_cls(ctx, self._generate_local)

    def _generate_local(self, *args, **kwargs):  # pragma: no cover - placeholder
        return "local"

    def _execute_vllm_request(self, state, pending):
        return ("exec", state, pending)

    def _request_vllm_batch(self, prompts, n):
        return ("batch", prompts, n)


def test_ensure_vllm_client_handles_import_error_and_typeerror(monkeypatch):
    monkeypatch.setattr(
        vllm_adapter.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError()),
    )
    ctx = SimpleNamespace(
        vllm_sync_weights=True,
        accelerator=SimpleNamespace(is_main_process=True),
        vllm_url="http://vllm",
    )
    gen = _DummyGenerator(ctx)

    class _Client:
        def __init__(self, *args, **kwargs):
            if kwargs:
                raise TypeError("no kwargs supported")
            self.init_called = False

        def init_communicator(self):
            self.init_called = True

    gen._import_vllm_client_cls = lambda: _Client

    ready = gen._ensure_vllm_client()

    assert ready is True
    assert isinstance(gen._vllm_helper._vllm_client, _Client)
    assert gen._vllm_helper._vllm_sync_ready is True
    assert gen._vllm_helper._vllm_client.init_called is True
    assert gen._vllm_helper._stored_base == ctx.vllm_url


def test_invoke_vllm_requests_uses_defaults_on_import_error(monkeypatch):
    monkeypatch.setattr(
        vllm_adapter.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError()),
    )

    def _sentinel_safe(*_a, **_k):
        return "safe"

    sentinel_safe = _sentinel_safe
    sentinel_time = SimpleNamespace()
    monkeypatch.setattr(vllm_adapter, "safe_generate", sentinel_safe)
    monkeypatch.setattr(vllm_adapter, "time", sentinel_time)
    gen = _DummyGenerator(SimpleNamespace())

    result = gen._invoke_vllm_requests(["p"], 2)

    assert gen._vllm_helper._safe_generate is sentinel_safe
    assert gen._vllm_helper._time is sentinel_time
    assert result == (["p"], 2, sentinel_safe, sentinel_time)


def test_run_vllm_rounds_sets_time_and_fallback_on_import_error(monkeypatch):
    monkeypatch.setattr(
        vllm_adapter.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError()),
    )
    sentinel_time = object()
    monkeypatch.setattr(vllm_adapter, "time", sentinel_time)
    gen = _DummyGenerator(SimpleNamespace())
    state = object()

    gen._run_vllm_rounds(state)

    assert gen._vllm_helper._run_state is state
    assert gen._vllm_helper._time is sentinel_time
    assert callable(gen._vllm_helper._fallback_generate)
    assert gen._vllm_helper._fallback_generate.__func__ is gen._generate_local.__func__


def test_scatter_object_returns_none_for_missing_and_out_of_range(monkeypatch):
    monkeypatch.setattr(vllm_adapter, "dist", None)

    accel_none = SimpleNamespace(num_processes=2, process_index=0, scatter_object=None)
    assert vllm_adapter._scatter_object(accel_none, None, src=0) is None

    accel_oob = SimpleNamespace(num_processes=3, process_index=5, scatter_object=None)
    assert vllm_adapter._scatter_object(accel_oob, ["only"], src=0) is None


def _make_missing_module_generator(ctx, helper_cls=_DummyHelper):
    """Return a generator whose module isn't importable to exercise import fallbacks."""

    missing_cls = type(
        "MissingModuleGenerator",
        (_DummyGenerator,),
        {"__module__": "missing.module"},
    )
    return missing_cls(ctx, helper_cls)


def test_ensure_vllm_client_handles_missing_helpers_module(monkeypatch):
    monkeypatch.setattr(
        vllm_adapter.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError()),
    )
    ctx = SimpleNamespace(
        vllm_sync_weights=True,
        accelerator=SimpleNamespace(is_main_process=True),
        vllm_url="http://vllm",
    )
    gen = _make_missing_module_generator(ctx)

    class _Client:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs
            self.init_called = False

        def init_communicator(self):
            self.init_called = True

    gen._import_vllm_client_cls = lambda: _Client

    ready = gen._ensure_vllm_client()

    assert ready is True
    assert isinstance(gen._vllm_helper._vllm_client, _Client)
    assert gen._vllm_helper._vllm_sync_ready is True
    # ensure fallback base-url logic still ran
    assert gen._vllm_helper._stored_base == ctx.vllm_url


def test_invoke_vllm_requests_handles_missing_helpers_module(monkeypatch):
    monkeypatch.setattr(
        vllm_adapter.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError()),
    )

    def _sentinel_safe(*_a, **_k):
        return "safe"

    sentinel_safe = _sentinel_safe
    sentinel_time = SimpleNamespace()
    monkeypatch.setattr(vllm_adapter, "safe_generate", sentinel_safe)
    monkeypatch.setattr(vllm_adapter, "time", sentinel_time)
    gen = _make_missing_module_generator(SimpleNamespace())

    result = gen._invoke_vllm_requests(["p"], 2)

    assert gen._vllm_helper._safe_generate is sentinel_safe
    assert gen._vllm_helper._time is sentinel_time
    assert result == (["p"], 2, sentinel_safe, sentinel_time)


def test_run_vllm_rounds_handles_missing_helpers_module(monkeypatch):
    monkeypatch.setattr(
        vllm_adapter.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError()),
    )
    sentinel_time = object()
    monkeypatch.setattr(vllm_adapter, "time", sentinel_time)
    gen = _make_missing_module_generator(SimpleNamespace())
    state = object()

    gen._run_vllm_rounds(state)

    assert gen._vllm_helper._run_state is state
    assert gen._vllm_helper._time is sentinel_time
