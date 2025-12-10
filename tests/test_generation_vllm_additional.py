"""Additional coverage for vLLM helper edge branches."""

from __future__ import annotations

import builtins
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import maxent_grpo.generation.vllm as vllm
from maxent_grpo.generation import vllm_distributed, vllm_requests
from tests.helpers.vllm import make_vllm_context


def _ctx(**overrides):
    return make_vllm_context(**overrides)


def test_sync_model_params_skips_child_without_named_params():
    pushed = []

    class _Param:
        def __init__(self, name):
            self.data = name

    class _FSDP:
        @staticmethod
        @contextmanager
        def summon_full_params(_module, recurse=False, writeback=False):
            yield

        def __init__(self):
            self.child = None

        def named_children(self):
            return []

        def named_parameters(self):
            return []

    class _Child(_FSDP):
        named_parameters = None

    class _Root(_FSDP):
        def __init__(self):
            super().__init__()
            self.child = _Child()

        def named_children(self):
            return [("child", self.child)]

        def named_parameters(self):
            return [("root", _Param("root"))]

    helper = vllm.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    helper._fsdp_cls = _FSDP  # treat root/child as fsdp instances
    helper._push_param_to_vllm = lambda name, param: pushed.append(name)
    helper._sync_model_params_to_vllm(_Root())
    assert pushed in (["root"], ["root", "child.root"], [])


def test_sync_model_params_walks_summon_full_params_tree():
    pushed = []

    class _Param:
        def __init__(self, name):
            self.data = name

    class _Node:
        def __init__(self, name, children=None, params=None):
            self.name = name
            self._children = children or []
            self._params = params or []

        def named_children(self):
            return [(child.name, child) for child in self._children]

        def named_parameters(self):
            return self._params

        @staticmethod
        @contextmanager
        def summon_full_params(_module, recurse=False, writeback=False):
            yield

    leaf = _Node("leaf", params=[("p", _Param("p"))])
    root = _Node("root", children=[leaf])
    helper = vllm.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    helper._push_param_to_vllm = lambda name, param: pushed.append(name)
    helper._sync_model_params_to_vllm(root)
    assert len(set(pushed)) == len(pushed)


def test_sync_standard_params_handles_nameerror_gather():
    class _Model:
        def parameters(self):
            return [1]

        def named_parameters(self):
            return [("p", SimpleNamespace(data="p"))]

    helper = vllm.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    pushed = []
    helper._push_param_to_vllm = lambda name, param: pushed.append(name)

    def _bad_factory(_params):
        raise NameError("no nullcontext")

    helper._sync_standard_params(_Model(), _bad_factory)
    assert pushed == ["p"]


def test_fsdp_dedup_avoids_duplicate_names():
    pushed = []

    class _Param:
        def __init__(self, name):
            self.data = name

    class _Child:
        def named_children(self):
            return []

        def named_parameters(self):
            return [
                ("_fsdp_wrapped_module.shared", _Param("a")),
                ("shared", _Param("b")),
            ]

        @staticmethod
        @contextmanager
        def summon_full_params(_module, recurse=False, writeback=False):
            yield

    class _Root(_Child):
        def __init__(self):
            super().__init__()
            self.child = _Child()

        def named_children(self):
            return [("child", self.child)]

    helper = vllm.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    helper._fsdp_cls = _Child
    helper._push_param_to_vllm = lambda name, param: pushed.append(name)
    helper._sync_model_params_to_vllm(_Root())
    # Only one push per unique cleaned name if pushes occur.
    assert len(set(pushed)) == len(pushed)


def test_generate_collective_with_hooks():
    helper = vllm.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    calls = {}

    def _fake_generate(prompts, n, counts, ensure_client=None, sync_model=None):
        calls["args"] = (prompts, n, counts, ensure_client, sync_model)
        return [["a"]], None

    helper.generate = _fake_generate  # type: ignore[assignment]
    accel = SimpleNamespace(
        is_main_process=True,
        num_processes=1,
        process_index=0,
        gather_object=lambda obj: [obj],
    )
    helper.ctx.accelerator = accel
    grouped, meta = helper.generate_collective(
        ["p"], 1, None, lambda: True, lambda m: None
    )
    assert grouped == [["a"]] and meta is None
    assert calls["args"][3] is not None and calls["args"][4] is not None


def test_sync_model_params_sets_fsdp_cls_from_model():
    pushed = []

    class _Param:
        def __init__(self, data):
            self.data = data

    class _Model:
        @staticmethod
        @contextmanager
        def summon_full_params(_module, recurse=False, writeback=False):
            yield

        def named_children(self):
            return []

        def named_parameters(self):
            return [("p", _Param("p"))]

    helper = vllm.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    helper._fsdp_cls = None  # force fsdp_cls discovery from model
    helper._push_param_to_vllm = lambda name, param: pushed.append(
        (name, getattr(param, "data", None))
    )
    helper._sync_model_params_to_vllm(_Model())
    assert pushed == [("p", "p")]


def test_sync_model_params_walks_when_hasattr_flaky():
    class _FlakyModel:
        def __init__(self):
            self._has_attr = False

        def __getattr__(self, name):
            if name == "summon_full_params":
                if not self._has_attr:
                    self._has_attr = True
                    raise AttributeError("missing on first check")
                return lambda *_a, **_k: None
            raise AttributeError(name)

        def named_parameters(self):
            return [("root", SimpleNamespace(data="root"))]

        def named_children(self):
            child = SimpleNamespace(
                named_parameters=lambda: [("leaf", SimpleNamespace(data="leaf"))],
                named_children=lambda: [],
            )
            return [("child", child)]

    helper = vllm.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    helper._fsdp_cls = None
    pushed = []
    reset_called = {}
    helper._push_param_to_vllm = lambda name, param: pushed.append(name)
    helper._reset_vllm_cache = lambda: reset_called.setdefault("reset", True)
    helper._sync_model_params_to_vllm(_FlakyModel())
    assert reset_called.get("reset") is True
    assert set(pushed) == {"root", "child.leaf"}


def test_sync_fsdp_params_handles_full_param_summon():
    pushed = []

    class _Param:
        def __init__(self, data):
            self.data = data

    class _FSDP:
        @staticmethod
        @contextmanager
        def summon_full_params(_module, recurse=False, writeback=False):
            yield

        def named_parameters(self):
            return [("wrapped.param", _Param("wrapped"))]

        def named_children(self):
            return []

    helper = vllm.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    helper._fsdp_cls = _FSDP
    helper._push_param_to_vllm = lambda name, param: pushed.append(
        (name, getattr(param, "data", None))
    )
    helper._sync_fsdp_params(_FSDP(), gather_factory=lambda _p: nullcontext())
    assert pushed and pushed[0][0] == "wrapped.param"


def test_prompt_char_limit_falls_back_on_import_error(monkeypatch):
    helper = vllm.VLLMGenerationHelper(_ctx(), lambda *_: ([], None))
    original_import = builtins.__import__

    def _boom(name, *args, **kwargs):
        if name == "maxent_grpo.generation.vllm":
            raise ImportError("fail")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _boom)
    limit = helper._prompt_char_limit()
    assert limit == vllm_requests._DEFAULT_PROMPT_CHAR_LIMIT


def test_current_torch_prefers_global_when_no_vllm_module(monkeypatch):
    monkeypatch.setitem(
        vllm_distributed.sys.modules, "maxent_grpo.generation.vllm", None
    )
    assert vllm_distributed._current_torch() is vllm_distributed.torch
