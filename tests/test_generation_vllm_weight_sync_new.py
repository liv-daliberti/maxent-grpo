"""Targeted coverage for vllm_weight_sync edge paths introduced by refactor."""

from __future__ import annotations

import builtins
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import maxent_grpo.generation.vllm as vllm
from tests.test_generation_vllm_unit import _ctx


def test_sync_model_params_sets_fsdp_cls_from_summon_full_params(monkeypatch):
    """When fsdp_cls is None but summon_full_params exists, it should walk params."""

    pushed: list[str] = []
    helper = vllm.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    helper._fsdp_cls = None
    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, _p: pushed.append(name)
    )

    class _Model:
        def __init__(self):
            self.p = SimpleNamespace(data="p")

        @staticmethod
        @contextmanager
        def summon_full_params(_module, recurse=False, writeback=False):
            yield

        def named_children(self):
            return []

        def named_parameters(self):
            return [("p", self.p)]

    helper._sync_model_params_to_vllm(_Model())
    assert pushed == ["p"]


def test_sync_model_params_walk_branch_with_hasattr_toggle(monkeypatch):
    """Force the fallback _walk branch via toggled hasattr results."""

    pushed: list[str] = []
    helper = vllm.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    helper._fsdp_cls = None
    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, _p: pushed.append(name)
    )

    class _Node:
        def __init__(self, name):
            self.name = name

        def named_children(self):
            return []

        def named_parameters(self):
            return [(self.name, SimpleNamespace(data=self.name))]

        @staticmethod
        @contextmanager
        def summon_full_params(_module, recurse=False, writeback=False):
            yield

    root = _Node("root")

    toggle = {"count": 0}
    original_hasattr = builtins.hasattr

    def _fake_hasattr(obj, attr):
        if attr == "summon_full_params":
            toggle["count"] += 1
            return toggle["count"] > 1
        return original_hasattr(obj, attr)

    monkeypatch.setattr(builtins, "hasattr", _fake_hasattr)
    helper._sync_model_params_to_vllm(root)
    # Second hasattr passes, so the walker should have pushed the param.
    assert pushed == ["root"]


def test_sync_model_params_sets_fsdp_cls_and_children(monkeypatch):
    """Ensure fsdp_cls assignment from summon_full_params triggers child sync."""

    pushed: list[str] = []
    helper = vllm.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    helper._fsdp_cls = None
    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, _p: pushed.append(name)
    )
    monkeypatch.setattr(helper, "_reset_vllm_cache", lambda: pushed.append("reset"))

    class _Child:
        def named_parameters(self):
            return [("child.w", SimpleNamespace(data="cw"))]

        def named_children(self):
            return []

    class _Model:
        def named_children(self):
            return [("child", _Child())]

        def named_parameters(self):
            return [("_checkpoint_wrapped_module.root", SimpleNamespace(data="r"))]

        @staticmethod
        def summon_full_params(*_a, **_k):
            return None

    helper._sync_model_params_to_vllm(_Model())
    assert "child.child.w" in pushed
    assert pushed[-1] == "reset"


def test_sync_standard_params_skips_none_and_cleans_prefix(monkeypatch):
    helper = vllm.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    pushed: list[str] = []
    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, _p: pushed.append(name)
    )

    class _Model:
        def parameters(self):
            return []

        def named_parameters(self):
            return [("keep.me", SimpleNamespace(data="x")), ("drop", None)]

    helper._sync_standard_params(_Model(), prefix="pre.")
    assert pushed == ["pre.keep.me"]


def test_sync_fsdp_params_handles_attr_error_and_summon_full(monkeypatch):
    pushed: list[str] = []
    helper = vllm.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))

    class _FSDP:
        @staticmethod
        @contextmanager
        def summon_full_params(_module, recurse=False, writeback=False):
            yield

    helper._fsdp_cls = _FSDP
    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, _p: pushed.append(name)
    )

    class _Module(_FSDP):
        def parameters(self):
            raise AttributeError("no params")

        def named_parameters(self):
            return [("_fsdp_wrapped_module.a", SimpleNamespace(data="a"))]

        def named_children(self):
            return []

    helper._sync_fsdp_params(_Module())
    # After summon_full_params, cleaned name should be pushed once.
    assert pushed == ["a"]


def test_sync_model_params_walk_branch_after_initial_attr_error(monkeypatch):
    """Trigger the _walk path when summon_full_params exists only after retry."""

    helper = vllm.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    helper._fsdp_cls = None
    helper._is_peft_model_safe = lambda _m: False  # avoid peft path
    helper._gather_factory = lambda _params: nullcontext()
    pushed: list[str] = []
    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, _p: pushed.append(name)
    )
    monkeypatch.setattr(helper, "_reset_vllm_cache", lambda: pushed.append("reset"))

    class _Leaf:
        def __init__(self, name: str):
            self._param = SimpleNamespace(data=name)
            self._name = name

        def named_parameters(self):
            return [(self._name, self._param)]

        def named_children(self):
            return []

    class _Root:
        def __init__(self):
            self._attr_ready = False
            self.child = _Leaf("leaf.weight")

        def __getattr__(self, name):
            if name == "summon_full_params":
                if self._attr_ready:
                    return lambda *_a, **_k: None
                self._attr_ready = True
                raise AttributeError("not ready")
            raise AttributeError(name)

        def named_parameters(self):
            return [("_fsdp_wrapped_module.root", SimpleNamespace(data="root"))]

        def named_children(self):
            return [("child", self.child)]

    helper._sync_model_params_to_vllm(_Root())
    assert pushed == ["child.leaf.weight", "reset"]


def test_sync_fsdp_params_skips_visited_on_full_param_pass(monkeypatch):
    """Ensure the second summon_full_params pass does not duplicate pushes."""

    helper = vllm.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    helper._gather_factory = lambda _params: nullcontext()
    pushed: list[str] = []
    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, _p: pushed.append(name)
    )

    class _FSDP:
        @staticmethod
        @contextmanager
        def summon_full_params(_module, recurse=False, writeback=False):
            yield

    helper._fsdp_cls = _FSDP

    class _Module(_FSDP):
        def parameters(self):
            return []

        def named_parameters(self):
            return [("_checkpoint_wrapped_module.shared", SimpleNamespace(data="x"))]

        def named_children(self):
            return []

    helper._sync_fsdp_params(_Module())
    assert pushed == ["shared"]


def test_sync_model_params_walk_branch_with_builtin_hasattr(monkeypatch):
    """Force the secondary summon_full_params walk via patched hasattr."""

    helper = vllm.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    helper._fsdp_cls = None
    helper._is_peft_model_safe = lambda _m: False
    helper._gather_factory = lambda _params: nullcontext()
    pushed: list[str] = []
    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, _p: pushed.append(name)
    )
    monkeypatch.setattr(helper, "_reset_vllm_cache", lambda: pushed.append("reset"))

    class _Node:
        def __init__(self, name: str, children=None):
            self._name = name
            self._param = SimpleNamespace(data=name)
            self._children = children or []

        @staticmethod
        @contextmanager
        def summon_full_params(_module, recurse=False, writeback=False):
            yield

        def named_parameters(self):
            return [(self._name, self._param)]

        def named_children(self):
            return self._children

    root = _Node("root", [("child", _Node("child"))])
    calls = {"count": 0}
    real_hasattr = builtins.hasattr

    def _fake_hasattr(obj, attr):
        if attr == "summon_full_params":
            calls["count"] += 1
            return calls["count"] > 1
        return real_hasattr(obj, attr)

    monkeypatch.setattr(builtins, "hasattr", _fake_hasattr)
    helper._sync_model_params_to_vllm(root)
    assert pushed == ["child.child", "reset"]


def test_sync_fsdp_params_full_param_pass_pushes_missing_first(monkeypatch):
    """Ensure summon_full_params pass adds params not seen in the first traversal."""

    helper = vllm.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    helper._gather_factory = lambda _params: nullcontext()
    pushed: list[str] = []
    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, _p: pushed.append(name)
    )

    class _FSDP:
        @staticmethod
        @contextmanager
        def summon_full_params(_module, recurse=False, writeback=False):
            yield

    helper._fsdp_cls = _FSDP

    class _Module(_FSDP):
        def __init__(self):
            self._calls = 0

        def parameters(self):
            return []

        def named_parameters(self):
            self._calls += 1
            if self._calls == 1:
                return []
            return [("late_param", SimpleNamespace(data="v"))]

        def named_children(self):
            return []

    helper._sync_fsdp_params(_Module())
    assert pushed == ["late_param"]


def test_sync_model_params_fsdp_branch_resets_cache(monkeypatch):
    """When fsdp_cls matches the model, children and cache reset should be hit."""

    pushed: list[str] = []
    helper = vllm.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))

    class _FSDP:
        pass

    helper._fsdp_cls = _FSDP
    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, _p: pushed.append(name)
    )
    monkeypatch.setattr(helper, "_reset_vllm_cache", lambda: pushed.append("reset"))

    class _Model(_FSDP):
        def named_children(self):
            return [("child", self)]

        def named_parameters(self):
            return [("_fsdp_wrapped_module.w", SimpleNamespace(data="w"))]

    helper._sync_model_params_to_vllm(_Model())
    assert pushed[:1] == ["child.w"]
    assert pushed[-1] == "reset"


def test_push_param_to_vllm_skips_when_missing_callable():
    helper = vllm.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    helper._client_callable = lambda name: None  # type: ignore[assignment]
    helper._push_param_to_vllm("x", SimpleNamespace(data="d"))  # should no-op


def test_reset_vllm_cache_invokes_client(monkeypatch):
    called = {}
    helper = vllm.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    helper._client_callable = lambda name: (lambda: called.setdefault("reset", True))
    helper._reset_vllm_cache()
    assert called.get("reset") is True


def test_sync_standard_params_handles_nameerror_and_children(monkeypatch):
    pushed: list[str] = []
    helper = vllm.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    helper._gather_factory = lambda _p: (_ for _ in ()).throw(NameError("nope"))
    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, _p: pushed.append(name)
    )

    class _Child:
        def named_parameters(self):
            return [("dup", SimpleNamespace(data="c"))]

        def named_children(self):
            return []

    class _Model:
        def parameters(self):
            return []

        def named_parameters(self):
            return [
                ("dup", SimpleNamespace(data="m")),
                ("dup", SimpleNamespace(data="skip_dup")),
            ]

        def named_children(self):
            return [("child", _Child())]

    helper._sync_standard_params(_Model(), prefix="pre.")
    assert pushed == ["pre.dup", "pre.child.dup"]


def test_sync_model_params_sets_fsdp_cls_after_attr_retry(monkeypatch):
    """First hasattr fails, second succeeds, then FSDP branch should run."""

    pushed: list[str] = []
    resets: list[str] = []
    helper = vllm.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    helper._fsdp_cls = None
    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, _p: pushed.append(name)
    )
    monkeypatch.setattr(helper, "_reset_vllm_cache", lambda: resets.append("reset"))

    class _FlakyFSDP:
        def __init__(self):
            self._first = True
            self._param = SimpleNamespace(data="p")

        def __getattribute__(self, name):
            if name == "summon_full_params":
                if object.__getattribute__(self, "_first"):
                    object.__setattr__(self, "_first", False)
                    raise AttributeError("not yet")
            return object.__getattribute__(self, name)

        @staticmethod
        def summon_full_params(*_args, **_kwargs):
            return None

        def named_children(self):
            return []

        def named_parameters(self):
            return [("weight", self._param)]

    helper._sync_model_params_to_vllm(_FlakyFSDP())
    assert helper._fsdp_cls is _FlakyFSDP
    assert pushed == ["weight"]
    assert resets == ["reset"]


def test_sync_model_params_second_fsdp_pass_hits_continues(monkeypatch):
    """Second FSDP branch should skip missing params and visited entries."""

    pushed: list[str] = []
    resets: list[str] = []
    helper = vllm.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    helper._fsdp_cls = None
    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, _p: pushed.append(name)
    )
    monkeypatch.setattr(helper, "_reset_vllm_cache", lambda: resets.append("reset"))

    class _NoParams:
        named_parameters = None

        def named_children(self):
            return []

    class _Child:
        def named_parameters(self):
            return [
                ("dup", SimpleNamespace(data="d")),
                ("fresh", SimpleNamespace(data="f")),
            ]

        def named_children(self):
            return []

    class _Model:
        summon_full_params = staticmethod(lambda *_a, **_k: None)

        def __init__(self):
            self.no_params = _NoParams()
            self.child = _Child()

        def __getattribute__(self, name):
            if name == "summon_full_params":
                raise AttributeError("probe")
            return object.__getattribute__(self, name)

        def named_children(self):
            return [("no", self.no_params), ("child", self.child)]

        def named_parameters(self):
            return []

    helper._sync_model_params_to_vllm(_Model(), visited={"child.dup"})
    assert pushed == ["child.fresh"]
    assert resets == ["reset"]
    assert helper._fsdp_cls is _Model


def test_sync_model_params_fsdp_branch_uses_existing_cls(monkeypatch):
    """When fsdp_cls is preset and summon_full_params is absent, child params sync."""

    pushed: list[str] = []
    resets: list[str] = []
    helper = vllm.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))

    class _PresetFSDP:
        def named_children(self):
            return [
                (
                    "child",
                    SimpleNamespace(
                        named_parameters=lambda: [("p", "v")], named_children=lambda: []
                    ),
                )
            ]

        def named_parameters(self):
            return [("root", "r")]

    helper._fsdp_cls = _PresetFSDP
    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, _p: pushed.append(name)
    )
    monkeypatch.setattr(helper, "_reset_vllm_cache", lambda: resets.append("reset"))

    helper._sync_model_params_to_vllm(_PresetFSDP())
    assert pushed == ["child.p"]
    assert resets == ["reset"]


def test_sync_peft_params_calls_merge_and_unmerge(monkeypatch):
    helper = vllm.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    helper._gather_factory = lambda _p: nullcontext()
    pushed: list[str] = []
    flags = {"merged": False, "unmerged": False}
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
                ("base_model.model.drop", SimpleNamespace(data=2)),
                ("original_module.skip", SimpleNamespace(data=3)),
            ]

    helper._sync_peft_params(_Model())
    assert flags == {"merged": True, "unmerged": True}
    assert pushed == ["keep", "drop"]


def test_sync_fsdp_params_adds_from_summon_full(monkeypatch):
    pushed: list[str] = []
    helper = vllm.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    helper._gather_factory = lambda _params: nullcontext()

    class _FSDP:
        @staticmethod
        @contextmanager
        def summon_full_params(_module, recurse=False, writeback=False):
            yield

    helper._fsdp_cls = _FSDP

    class _Module(_FSDP):
        def __init__(self):
            self._first = True

        def parameters(self):
            return []

        def named_parameters(self):
            if self._first:
                self._first = False
                return [("_checkpoint_wrapped_module.a", SimpleNamespace(data="a"))]
            return [("b", SimpleNamespace(data="b"))]

        def named_children(self):
            return []

    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, _p: pushed.append(name)
    )
    helper._sync_fsdp_params(_Module())
    assert pushed == ["a", "b"]


def test_sync_model_params_handles_hasattr_exception_and_second_pass(monkeypatch):
    """First hasattr raises AttributeError, second returns True to set fsdp_cls."""

    helper = vllm.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    helper._fsdp_cls = None
    helper._is_peft_model_safe = lambda _m: False  # type: ignore[assignment]
    helper._gather_factory = lambda _p: nullcontext()
    pushed: list[str] = []
    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, _p: pushed.append(name)
    )
    monkeypatch.setattr(helper, "_reset_vllm_cache", lambda: pushed.append("reset"))

    class _Child:
        def named_parameters(self):
            return [
                ("p", SimpleNamespace(data="child")),
                ("skip", SimpleNamespace(data="skip")),
            ]

    class _Model:
        summon_full_params = staticmethod(lambda *_a, **_k: None)

        def __init__(self):
            self.child = _Child()

        def named_children(self):
            return [("child", self.child)]

        def named_parameters(self):
            return [("p", SimpleNamespace(data="root"))]

    calls = {"count": 0}
    real_hasattr = builtins.hasattr

    def _flaky_hasattr(obj, attr):
        if attr == "summon_full_params":
            calls["count"] += 1
            if calls["count"] == 1:
                raise AttributeError("boom")
            return True
        return real_hasattr(obj, attr)

    visited = {"child.p"}
    monkeypatch.setattr(builtins, "hasattr", _flaky_hasattr)
    helper._sync_model_params_to_vllm(_Model(), visited=visited)
    # child.p was skipped due to visited; child.skip pushed alongside root and reset.
    assert "child.p" not in pushed
    assert "child.skip" in pushed
    assert pushed[-1] == "reset"
    assert helper._fsdp_cls is not None


def test_sync_model_params_walk_skips_none_and_repushes_root(monkeypatch):
    """Walk branch should skip None params, honor visited, and push new ones."""

    helper = vllm.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    helper._fsdp_cls = None
    helper._is_peft_model_safe = lambda _m: False  # type: ignore[assignment]
    helper._gather_factory = lambda _p: nullcontext()
    pushed: list[str] = []
    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, _p: pushed.append(name)
    )
    monkeypatch.setattr(helper, "_reset_vllm_cache", lambda: pushed.append("reset"))

    class _Model:
        def __getattr__(self, name):
            if name == "summon_full_params":
                return lambda *_a, **_k: None
            raise AttributeError(name)

        def named_children(self):
            return []

        def named_parameters(self):
            return [
                ("none", None),
                ("_fsdp_wrapped_module.seen", SimpleNamespace(data="seen")),
                ("new", SimpleNamespace(data="new")),
                ("extra", SimpleNamespace(data="extra")),
            ]

    helper._sync_model_params_to_vllm(_Model(), visited={"seen"})
    # "new" pushed during walk, "extra" pushed during root pass, cache reset at end.
    assert pushed[:2] == ["new", "extra"]
    assert pushed[-1] == "reset"


def test_sync_model_params_sets_fsdp_cls_after_second_probe(monkeypatch):
    """Second summon_full_params probe should set fsdp_cls and sync parameters."""

    helper = vllm.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    helper._fsdp_cls = None
    helper._is_peft_model_safe = lambda _m: False  # type: ignore[assignment]
    helper._gather_factory = lambda _p: nullcontext()
    pushed: list[str] = []
    monkeypatch.setattr(
        helper, "_push_param_to_vllm", lambda name, _p: pushed.append(name)
    )
    monkeypatch.setattr(helper, "_reset_vllm_cache", lambda: pushed.append("reset"))

    class _Model:
        summon_full_params = staticmethod(lambda *_a, **_k: None)

        def named_children(self):
            return []

        def named_parameters(self):
            return [("_fsdp_wrapped_module.root", SimpleNamespace(data="r"))]

    calls = {"count": 0}
    real_hasattr = builtins.hasattr

    def _flaky_hasattr(obj, attr):
        if attr == "summon_full_params":
            calls["count"] += 1
            return calls["count"] > 1
        return real_hasattr(obj, attr)

    monkeypatch.setattr(builtins, "hasattr", _flaky_hasattr)
    helper._sync_model_params_to_vllm(_Model())

    assert helper._fsdp_cls is _Model
    assert pushed == ["root", "reset"]
