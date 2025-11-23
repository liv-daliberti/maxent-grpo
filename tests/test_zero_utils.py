"""Tests for utilities that integrate with DeepSpeed ZeRO."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

torch_stub = sys.modules.setdefault("torch", ModuleType("torch"))
torch_stub.__spec__ = getattr(torch_stub, "__spec__", SimpleNamespace())
torch_stub.__path__ = getattr(torch_stub, "__path__", [])
torch_stub.Tensor = getattr(torch_stub, "Tensor", type("Tensor", (), {}))
torch_stub.cuda = getattr(
    torch_stub,
    "cuda",
    SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
)
torch_utils = getattr(torch_stub, "utils", SimpleNamespace())
torch_data = getattr(torch_utils, "data", SimpleNamespace())
if not hasattr(torch_data, "DataLoader"):

    class _DataLoader:  # minimal stub
        pass

    torch_data.DataLoader = _DataLoader
if not hasattr(torch_data, "Sampler"):

    class _Sampler:
        pass

    torch_data.Sampler = _Sampler
torch_utils_module = sys.modules.setdefault("torch.utils", ModuleType("torch.utils"))
torch_utils_module.__spec__ = getattr(torch_utils_module, "__spec__", SimpleNamespace())
torch_utils_module.__path__ = getattr(torch_utils_module, "__path__", [])
torch_utils_module.data = torch_data
torch_data_module = sys.modules.setdefault(
    "torch.utils.data", ModuleType("torch.utils.data")
)
torch_data_module.__spec__ = getattr(torch_data_module, "__spec__", SimpleNamespace())
torch_data_module.__path__ = getattr(torch_data_module, "__path__", [])
torch_data_module.DataLoader = torch_data.DataLoader
torch_data_module.Sampler = torch_data.Sampler
torch_stub.utils = torch_utils_module
torch_utils_module.data = torch_data_module

torch_nn = sys.modules.setdefault("torch.nn", ModuleType("torch.nn"))
torch_nn.__spec__ = getattr(torch_nn, "__spec__", SimpleNamespace())
if not hasattr(torch_nn, "Module"):

    class _Module:
        pass

    torch_nn.Module = _Module
torch_nn_functional = sys.modules.setdefault(
    "torch.nn.functional", ModuleType("torch.nn.functional")
)
torch_nn_functional.__spec__ = getattr(
    torch_nn_functional, "__spec__", SimpleNamespace()
)
if not hasattr(torch_nn_functional, "log_softmax"):

    def _log_softmax(*_args, **_kwargs):
        raise NotImplementedError

    torch_nn_functional.log_softmax = _log_softmax

torch_optim = sys.modules.setdefault("torch.optim", ModuleType("torch.optim"))
torch_optim.__spec__ = getattr(torch_optim, "__spec__", SimpleNamespace())
if not hasattr(torch_optim, "Optimizer"):

    class _Optimizer:
        pass

    torch_optim.Optimizer = _Optimizer

accelerate_mod = sys.modules.setdefault("accelerate", ModuleType("accelerate"))
accelerate_mod.__spec__ = getattr(accelerate_mod, "__spec__", SimpleNamespace())
if not hasattr(accelerate_mod, "Accelerator"):

    class _Accel:
        def __init__(self, **_kwargs):
            pass

    accelerate_mod.Accelerator = _Accel

transformers_mod = sys.modules.setdefault("transformers", ModuleType("transformers"))
transformers_mod.__spec__ = getattr(transformers_mod, "__spec__", SimpleNamespace())
transformers_mod.PreTrainedModel = getattr(transformers_mod, "PreTrainedModel", object)
transformers_mod.PreTrainedTokenizer = getattr(
    transformers_mod,
    "PreTrainedTokenizer",
    object,
)

zero_utils = importlib.import_module("training.zero_utils")
zero_utils._DEEPSPEED_READY = True
zero_utils.ZeroParamStatus = object()


class _FakeGather:
    """Lightweight context manager that records gathered params."""

    def __init__(
        self, params, modifier_rank=None
    ):  # noqa: D401 - signature mirrors DS API
        self.params = list(params)
        self.entered = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyParam:
    def __init__(self, name: str):
        self.name = name


class _DummyModel:
    def __init__(self):
        self._params = [_DummyParam("w"), _DummyParam("b")]
        self.zero_optimization_stage = 2

    def parameters(self):
        return iter(self._params)


class _Recorder:
    def __init__(self):
        self.messages = []

    def warning(self, msg, *args):
        if args:
            msg = msg % args
        self.messages.append(msg)


class _Zero3Engine:
    def __init__(self):
        self.zero_optimization_stage = 3
        self._partition = True
        self.original_calls = 0

    def zero_optimization_partition_gradients(self):
        return self._partition

    class _OrigCtx:
        def __init__(self, outer):
            self.outer = outer

        def __enter__(self):
            self.outer.original_calls += 1
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def no_sync(self):
        return self._OrigCtx(self)


def test_zero_stage_gathers_full_parameter_list(monkeypatch):
    """When running with ZeRO>0 we gather the entire parameter list."""

    model = _DummyModel()

    monkeypatch.setattr(
        zero_utils,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
        ),
        raising=False,
    )

    captured = {}

    def _fake_gathered(params, modifier_rank=None):
        handle = _FakeGather(params, modifier_rank)
        captured["params"] = handle.params
        captured["ctx"] = handle
        return handle

    monkeypatch.setattr(
        zero_utils,
        "ds_zero",
        SimpleNamespace(GatheredParameters=_fake_gathered),
    )

    with zero_utils._maybe_zero_gather_params(model, enabled=True):
        assert captured["ctx"].entered is True

    assert captured["params"] == list(model.parameters())


def test_zero_stage_callable_property(monkeypatch):
    """Callable zero_optimization_stage attributes are handled gracefully."""

    class _MethodModel(_DummyModel):
        def zero_optimization_stage(self):
            return 1

    model = _MethodModel()
    monkeypatch.setattr(
        zero_utils,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
        ),
        raising=False,
    )
    captured = {}

    def _fake_gathered(params, modifier_rank=None):
        handle = _FakeGather(params, modifier_rank)
        captured["params"] = handle.params
        captured["ctx"] = handle
        return handle

    monkeypatch.setattr(
        zero_utils,
        "ds_zero",
        SimpleNamespace(GatheredParameters=_fake_gathered),
    )

    with zero_utils._maybe_zero_gather_params(model, enabled=True):
        assert captured["ctx"].entered is True


def test_zero_no_sync_patch_replaces_context(monkeypatch):
    """ZeRO-3 engines get their no_sync patched to avoid asserts."""

    engine = _Zero3Engine()
    recorder = _Recorder()
    monkeypatch.setattr(zero_utils, "LOG", recorder, raising=False)

    patched = zero_utils._maybe_patch_zero_no_sync(engine)
    assert patched is True
    assert getattr(engine, "_maxent_zero_no_sync_patched", False) is True

    with engine.no_sync():
        pass
    with engine.no_sync():
        pass
    assert (
        engine.original_calls == 0
    ), "Patched context should skip original while partitioning"
    assert len(recorder.messages) == 1

    engine._partition = False
    with engine.no_sync():
        pass
    assert (
        engine.original_calls == 1
    ), "Once partitioning stops we delegate to the real context"
    assert zero_utils._maybe_patch_zero_no_sync(engine) is False
