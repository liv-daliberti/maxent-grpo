"""Extra coverage for vllm_weight_sync edge cases."""

from __future__ import annotations

import inspect
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import maxent_grpo.generation.vllm_weight_sync as weight_sync


class _WeightSyncUnderTest(weight_sync.VLLMWeightSyncMixin):
    def __init__(self):
        self.ctx = SimpleNamespace(
            accelerator=SimpleNamespace(
                is_main_process=True, unwrap_model=lambda m: m, wait_for_everyone=None
            ),
            generation_stats={},
            model=None,
            vllm_sync_weights=True,
            vllm_url="http://example",
        )
        self._vllm_client = None
        self._vllm_sync_ready = True
        self._last_vllm_synced_step = None
        self._fsdp_cls = None
        self._gather_factory = None


def test_maybe_sync_weights_handles_signature_error(monkeypatch):
    waits: list[str] = []
    helper = _WeightSyncUnderTest()
    helper.ctx.generation_stats["current_step"] = 3
    helper.ctx.model = "m"
    helper.ctx.accelerator = SimpleNamespace(
        unwrap_model=lambda m: f"uw-{m}",
        wait_for_everyone=lambda: waits.append("wait"),
        is_main_process=True,
    )

    calls: list[str] = []

    def _sync_fn(model):
        calls.append(model)

    real_signature = inspect.signature

    def _raising_signature(fn):
        if fn is _sync_fn:
            raise TypeError("no sig")
        return real_signature(fn)

    monkeypatch.setattr(
        "maxent_grpo.generation.vllm_weight_sync.inspect.signature", _raising_signature
    )
    helper.maybe_sync_weights(ensure_client=lambda: True, sync_model=_sync_fn)
    assert calls == ["uw-m"]
    assert helper.ctx.generation_stats["vllm_weight_syncs"] == 1
    assert helper._last_vllm_synced_step == 3
    assert waits == ["wait"]


def test_client_callable_handles_missing_and_present():
    helper = _WeightSyncUnderTest()
    helper._vllm_client = SimpleNamespace(
        missing="nope", present=lambda: "ok"  # type: ignore[assignment]
    )
    assert helper._client_callable("missing") is None
    wrapped = helper._client_callable("present")
    assert wrapped is not None
    assert wrapped() == "ok"


def test_sync_model_params_uses_distributed_fsdp(monkeypatch):
    pushed: list[str] = []
    helper = _WeightSyncUnderTest()

    class _Child:
        def named_parameters(self):
            return [("p_child", SimpleNamespace(data="child"))]

        def named_children(self):
            return []

    class _FSDP:
        def __init__(self):
            self.child = _Child()

        def named_children(self):
            return [("child", self.child)]

        def named_parameters(self):
            return [("_fsdp_wrapped_module.root", SimpleNamespace(data="root"))]

    torch_stub = SimpleNamespace(
        distributed=SimpleNamespace(
            fsdp=SimpleNamespace(FullyShardedDataParallel=_FSDP)
        )
    )
    monkeypatch.setattr(weight_sync, "torch", torch_stub)
    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, _p: pushed.append(name)
    )
    monkeypatch.setattr(helper, "_reset_vllm_cache", lambda: pushed.append("reset"))

    helper._sync_model_params_to_vllm(_FSDP())
    assert pushed == ["child.p_child", "reset"]


def test_sync_model_params_fsdp_second_pass_after_missing_summon(monkeypatch):
    pushed: list[str] = []
    helper = _WeightSyncUnderTest()
    helper._fsdp_cls = None
    monkeypatch.setattr(weight_sync, "torch", SimpleNamespace(distributed=None))
    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, _p: pushed.append(name)
    )
    monkeypatch.setattr(helper, "_reset_vllm_cache", lambda: pushed.append("reset"))

    class _FSDP:
        @staticmethod
        def summon_full_params(*_a, **_k):
            return None

        def __getattribute__(self, name: str):
            if name == "summon_full_params":
                raise RuntimeError("not ready")
            return object.__getattribute__(self, name)

        def named_children(self):
            return []

        def named_parameters(self):
            return [("_fsdp_wrapped_module.param", SimpleNamespace(data="p"))]

    helper._sync_model_params_to_vllm(_FSDP())
    assert pushed == ["param", "reset"]


def test_sync_model_params_walk_branch(monkeypatch):
    pushed: list[str] = []
    helper = _WeightSyncUnderTest()
    helper._fsdp_cls = None
    monkeypatch.setattr(weight_sync, "torch", SimpleNamespace(distributed=None))
    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, _p: pushed.append(name)
    )
    monkeypatch.setattr(helper, "_reset_vllm_cache", lambda: pushed.append("reset"))

    class _Leaf:
        def named_children(self):
            return []

        def named_parameters(self):
            return [("weight", SimpleNamespace(data="leaf"))]

    class _Root:
        summon_full_params = staticmethod(lambda *_a, **_k: None)

        def __init__(self):
            self.leaf = _Leaf()

        def named_children(self):
            return [("leaf", self.leaf)]

        def named_parameters(self):
            return []

    helper._sync_model_params_to_vllm(_Root())
    assert pushed == ["leaf.weight", "reset"]


def test_sync_model_params_walk_respects_existing_visited(monkeypatch):
    pushed: list[str] = []
    helper = _WeightSyncUnderTest()
    helper._fsdp_cls = None
    monkeypatch.setattr(weight_sync, "torch", SimpleNamespace(distributed=None))
    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, _p: pushed.append(name)
    )
    monkeypatch.setattr(helper, "_reset_vllm_cache", lambda: pushed.append("reset"))

    class _Leaf:
        def __init__(self, name: str):
            self._name = name

        def named_children(self):
            return []

        def named_parameters(self):
            return [(self._name, SimpleNamespace(data=self._name))]

    class _Root:
        summon_full_params = staticmethod(lambda *_a, **_k: None)

        def __init__(self):
            self.leaf = _Leaf("p")
            self.other = _Leaf("p")

        def named_children(self):
            return [("leaf", self.leaf), ("other", self.other)]

        def named_parameters(self):
            return []

    visited = {"leaf.p"}
    helper._sync_model_params_to_vllm(_Root(), visited=visited)
    assert pushed == ["other.p", "reset"]
    assert "leaf.p" in visited


def test_sync_model_params_handles_peft(monkeypatch):
    pushed: list[str] = []
    helper = _WeightSyncUnderTest()
    helper._fsdp_cls = None
    helper._is_peft_model_safe = lambda _m: True  # type: ignore[assignment]
    monkeypatch.setattr(weight_sync, "torch", SimpleNamespace(distributed=None))
    monkeypatch.setattr(helper, "_sync_peft_params", lambda model: pushed.append(model))
    monkeypatch.setattr(helper, "_reset_vllm_cache", lambda: pushed.append("reset"))
    helper._sync_model_params_to_vllm(SimpleNamespace(name="peft"))
    assert getattr(pushed[0], "name") == "peft"
    assert pushed[-1] == "reset"


def test_push_and_reset_vllm_cache(monkeypatch):
    helper = _WeightSyncUnderTest()
    calls: list[tuple[str, object]] = []
    helper._vllm_client = SimpleNamespace(
        update_named_param=lambda name, data: calls.append((name, data)),
        reset_prefix_cache=lambda: (_ for _ in ()).throw(ValueError("boom")),
    )
    helper._push_param_to_vllm("p", SimpleNamespace(data="payload"))
    helper._reset_vllm_cache()  # Should swallow the ValueError.
    assert calls == [("p", "payload")]


def test_sync_standard_params_skips_none_and_recurse(monkeypatch):
    pushed: list[str] = []
    helper = _WeightSyncUnderTest()
    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, _p: pushed.append(name)
    )

    class _Child:
        def named_parameters(self):
            return [("_checkpoint_wrapped_module.child", SimpleNamespace(data="c"))]

        def named_children(self):
            return []

    class _Model:
        def parameters(self):
            return []

        def named_parameters(self):
            return [("keep", SimpleNamespace(data="k")), ("skip", None)]

        def named_children(self):
            return [("child", _Child())]

    helper._sync_standard_params(_Model(), prefix="root.")
    assert pushed == ["root.keep", "root.child.child"]


def test_sync_peft_params_skips_original_and_unmerges(monkeypatch):
    pushed: list[str] = []
    flags = {"merged": False, "unmerged": False}
    helper = _WeightSyncUnderTest()
    helper._gather_factory = lambda _p: nullcontext()
    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, _p: pushed.append(name)
    )

    class _Model:
        prefix = None

        def parameters(self):
            return []

        def merge_adapter(self):
            flags["merged"] = True

        def unmerge_adapter(self):
            flags["unmerged"] = True

        def named_parameters(self):
            return [
                ("modules_to_save.default.keep", SimpleNamespace(data=1)),
                ("original_module.skip", SimpleNamespace(data=2)),
            ]

    helper._sync_peft_params(_Model())
    assert flags == {"merged": True, "unmerged": True}
    assert pushed == ["keep"]


def test_sync_fsdp_params_runs_full_pass(monkeypatch):
    pushed: list[str] = []
    helper = _WeightSyncUnderTest()

    class _FSDP:
        @staticmethod
        @contextmanager
        def summon_full_params(_module, recurse=False, writeback=False):
            yield

    helper._fsdp_cls = _FSDP
    helper._gather_factory = lambda _params: nullcontext()
    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, _p: pushed.append(name)
    )

    class _Module(_FSDP):
        def __init__(self):
            self._calls = 0

        def parameters(self):
            return []

        def named_parameters(self):
            self._calls += 1
            if self._calls == 1:
                return [("_fsdp_wrapped_module.first", SimpleNamespace(data="a"))]
            return [("late", SimpleNamespace(data="b"))]

        def named_children(self):
            return []

    helper._sync_fsdp_params(_Module())
    assert pushed == ["first", "late"]
