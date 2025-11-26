"""Coverage for vLLM weight sync mixin edge cases."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace


from maxent_grpo.generation.vllm_weight_sync import VLLMWeightSyncMixin


class _Ctx:
    def __init__(self):
        self.accelerator = SimpleNamespace(
            is_main_process=True,
            unwrap_model=lambda m: m,
            wait_for_everyone=lambda: None,
        )
        self.generation_stats = {}
        self.vllm_sync_weights = True
        self.vllm_url = "http://localhost/generate"
        self.model = None


class _Helper(VLLMWeightSyncMixin):
    def __init__(self, ctx=None):
        self.ctx = ctx or _Ctx()
        self._vllm_client = None
        self._vllm_sync_ready = False
        self._last_vllm_synced_step = None
        self._fsdp_cls = None
        self._gather_factory = None

    # Expose protected helpers for tests
    def push_param(self, name, param):
        return self._push_param_to_vllm(name, param)

    def reset_cache(self):
        return self._reset_vllm_cache()


def test_base_url_strips_generate_suffix():
    helper = _Helper()
    assert helper._vllm_base_url("https://x.y/generate") == "https://x.y"
    assert helper._vllm_base_url("https://x.y/other") == "https://x.y"
    assert helper._vllm_base_url("https://x.y") == "https://x.y"
    assert helper._vllm_base_url("no-scheme/generate") == "no-scheme"


def test_ensure_client_handles_missing_and_noncallable(monkeypatch):
    helper = _Helper()

    def _none_cls():
        return None

    assert helper._ensure_vllm_client(import_vllm_client_cls=_none_cls) is False
    assert helper._vllm_sync_ready is False

    helper = _Helper()
    helper.ctx.vllm_sync_weights = True

    def _bad_cls():
        return "not-callable"

    assert helper._ensure_vllm_client(import_vllm_client_cls=_bad_cls) is False
    assert helper._vllm_client is None

    class _Client:
        def __init__(self, base_url=None):
            self.base_url = base_url
            self.init_calls = 0

        def init_communicator(self):
            self.init_calls += 1

    def _good_cls():
        return _Client

    assert helper._ensure_vllm_client(import_vllm_client_cls=_good_cls) is True
    assert isinstance(helper._vllm_client, _Client)
    assert helper._vllm_client.base_url == "http://localhost"


def test_maybe_sync_weights_skips_when_not_ready(monkeypatch):
    helper = _Helper()
    helper.ctx.vllm_sync_weights = False
    helper.maybe_sync_weights()
    assert helper._vllm_client is None

    helper.ctx.vllm_sync_weights = True
    helper.ctx.accelerator.is_main_process = False
    helper.maybe_sync_weights()
    assert helper._vllm_client is None


def test_maybe_sync_weights_updates_stats_and_step(monkeypatch):
    helper = _Helper()
    helper.ctx.generation_stats["current_step"] = 5

    def _ensure():
        return True

    synced = []

    def _sync_model(model):
        synced.append(model)

    helper.maybe_sync_weights(ensure_client=_ensure, sync_model=_sync_model)
    assert synced == [None]
    assert helper.ctx.generation_stats["vllm_weight_syncs"] == 1
    assert helper._last_vllm_synced_step == 5
    # Should no-op on same step
    helper.maybe_sync_weights(ensure_client=_ensure, sync_model=_sync_model)
    assert helper.ctx.generation_stats["vllm_weight_syncs"] == 1


def test_sync_standard_params_uses_gather_and_push(monkeypatch):
    helper = _Helper()
    pushed = []
    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, param: pushed.append(name)
    )
    # Fake gather context to ensure it is entered without errors.
    helper._gather_factory = lambda params: nullcontext()

    class _Model:
        def __init__(self):
            self.called = False

        def parameters(self):
            return [SimpleNamespace(data=1)]

        def named_parameters(self):
            return [
                ("a", SimpleNamespace(data=1)),
                ("_checkpoint_wrapped_module.b", SimpleNamespace(data=2)),
            ]

        def named_children(self):
            return [("child", _Child())]

    class _Child:
        def parameters(self):
            return []

        def named_parameters(self):
            return [("c", SimpleNamespace(data=3))]

        def named_children(self):
            return []

    helper._sync_standard_params(_Model())
    assert pushed == ["a", "b", "child.c"]


def test_sync_peft_params_merges_and_filters(monkeypatch):
    helper = _Helper()
    pushed = []
    helper._gather_factory = lambda params: nullcontext()

    class _Model:
        prefix = "skipme"

        def merge_adapter(self):
            pushed.append("merge")

        def unmerge_adapter(self):
            pushed.append("unmerge")

        def parameters(self):
            return [1, 2]

        def named_parameters(self):
            return [
                ("modules_to_save.default.a", SimpleNamespace(data=1)),
                ("base_model.model.b", SimpleNamespace(data=2)),
                ("skipme.c", SimpleNamespace(data=3)),
                ("original_module.d", SimpleNamespace(data=4)),
            ]

    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, param: pushed.append(name)
    )
    helper._sync_peft_params(_Model())
    assert pushed == ["merge", "a", "b", "unmerge"]


def test_sync_fsdp_params_handles_nested(monkeypatch):
    helper = _Helper()
    pushed = []
    helper._gather_factory = lambda params: nullcontext()

    class _FSDP:
        def __init__(self, child=None, name="root"):
            self.child = child
            self.name = name

        def parameters(self):
            return [SimpleNamespace(data=self.name)]

        def named_parameters(self):
            return [(f"{self.name}_p", SimpleNamespace(data=self.name))]

        def named_children(self):
            return [("child", self.child)] if self.child else []

        @staticmethod
        def summon_full_params(module, recurse=False, writeback=False):
            return nullcontext()

    child = _FSDP(name="child")
    root = _FSDP(child=child, name="root")
    helper._fsdp_cls = _FSDP
    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, param: pushed.append(name)
    )
    helper._sync_fsdp_params(root)
    assert pushed == ["root_p", "child.child_p"]


def test_sync_model_params_uses_summon_full_params(monkeypatch):
    helper = _Helper()
    pushed = []
    helper._reset_vllm_cache = lambda: pushed.append("reset")  # record reset
    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, param: pushed.append(name)
    )

    class _Model:
        def summon_full_params(self):
            return None

        def named_children(self):
            return []

        def named_parameters(self):
            return [("p", SimpleNamespace(data=1))]

    helper._sync_model_params_to_vllm(_Model())
    assert pushed == ["p", "reset"]


def test_sync_model_params_recursive_walk(monkeypatch):
    helper = _Helper()
    pushed = []
    reset_called = []
    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, param: pushed.append(name)
    )
    helper._reset_vllm_cache = lambda: reset_called.append(True)

    class _Child:
        def named_parameters(self):
            return [("b", SimpleNamespace(data=2))]

        def named_children(self):
            return []

    class _Model:
        def __init__(self):
            self._toggle = False
            self.child = _Child()

        def __getattr__(self, name):
            if name == "summon_full_params":
                if not self._toggle:
                    self._toggle = True
                    raise AttributeError
                return lambda: None
            raise AttributeError

        def named_parameters(self):
            return [("a", SimpleNamespace(data=1))]

        def named_children(self):
            return [("child", self.child)]

    helper._sync_model_params_to_vllm(_Model())
    assert reset_called == [True]


def test_ensure_client_falls_back_to_init_args(monkeypatch):
    helper = _Helper()
    calls = []

    class _Client:
        def __init__(self, base_url=None):
            self.base_url = base_url

        def init_communicator(self, client, base_url):
            calls.append((client, base_url))

    assert helper._ensure_vllm_client(import_vllm_client_cls=lambda: _Client) is True
    assert helper._vllm_sync_ready is True
    assert calls[-1][0] is helper._vllm_client
    assert calls[-1][1] == helper._vllm_base_url(helper.ctx.vllm_url)


def test_sync_model_params_fsdp_branch_uses_visited(monkeypatch):
    helper = _Helper()
    pushed = []
    helper._reset_vllm_cache = lambda: pushed.append("reset")
    helper._fsdp_cls = None
    helper._gather_factory = lambda params: nullcontext()

    class _FSDP:
        def __init__(self, params):
            self._params = params

        def named_children(self):
            return []

        def named_parameters(self):
            return self._params

    # Inject fsdp class and ensure visited set is honored.
    helper._fsdp_cls = _FSDP
    model = _FSDP(
        [("seen", SimpleNamespace(data=1)), ("fresh", SimpleNamespace(data=2))]
    )
    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, param: pushed.append(name)
    )
    helper._sync_model_params_to_vllm(model, visited={"seen"})
    assert "fresh" in pushed
    assert "seen" not in pushed  # visited prevents duplicate push
    assert pushed[-1] == "reset"


def test_sync_model_params_walk_honors_existing_visited(monkeypatch):
    helper = _Helper()
    pushed = []
    reset_called = []
    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, param: pushed.append(name)
    )
    helper._reset_vllm_cache = lambda: reset_called.append(True)

    class _Child:
        def named_parameters(self):
            return [("child_param", SimpleNamespace(data=2))]

        def named_children(self):
            return []

    class _Model:
        def summon_full_params(self):
            return None

        def named_parameters(self):
            return [("root_param", SimpleNamespace(data=1))]

        def named_children(self):
            return [("child", _Child())]

    helper._sync_model_params_to_vllm(_Model(), visited={"root_param"})
    assert pushed == ["child.child_param"]
    assert reset_called == [True]


def test_client_callable_wraps_and_handles_missing():
    helper = _Helper()
    # No client means None is returned
    assert helper._client_callable("update_named_param") is None
    helper._vllm_client = SimpleNamespace(echo=lambda x: f"echo:{x}")
    wrapped = helper._client_callable("echo")
    assert callable(wrapped)
    assert wrapped("v") == "echo:v"


def test_sync_model_params_falls_back_without_summon(monkeypatch):
    helper = _Helper()
    helper._fsdp_cls = None
    helper._is_peft_model_safe = lambda _m: False
    marker = {}
    helper._sync_standard_params = lambda model: marker.setdefault("model", model)

    class _Model:
        def parameters(self):
            return []

        def named_parameters(self):
            return []

        def named_children(self):
            return []

    helper._sync_model_params_to_vllm(_Model())
    assert marker["model"].__class__ is _Model
