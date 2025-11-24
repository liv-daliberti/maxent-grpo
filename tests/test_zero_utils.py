"""
Copyright 2025 Liv d'Aliberti

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Behavioral tests for zero_utils helpers with stubbed dependencies.
"""

from __future__ import annotations

import builtins
import importlib
import logging
import sys
from types import SimpleNamespace
import types
from contextlib import contextmanager

import pytest

# Seed lightweight stubs for optional runtime deps before importing training.*
from maxent_grpo.training.run_helpers import _build_torch_stub

torch_stub = _build_torch_stub()
torch_stub.__spec__ = types.SimpleNamespace(name="torch")
sys.modules["torch"] = torch_stub

torch_utils = sys.modules.setdefault("torch.utils", types.ModuleType("torch.utils"))
torch_utils.__spec__ = getattr(torch_utils, "__spec__", types.SimpleNamespace())
torch_data = sys.modules.setdefault(
    "torch.utils.data", types.ModuleType("torch.utils.data")
)
torch_data.__spec__ = getattr(torch_data, "__spec__", types.SimpleNamespace())
torch_data.DataLoader = getattr(torch_data, "DataLoader", type("DataLoader", (), {}))
torch_data.Sampler = getattr(torch_data, "Sampler", type("Sampler", (), {}))
torch_utils.data = torch_data
torch_optim = sys.modules.setdefault("torch.optim", types.ModuleType("torch.optim"))
torch_optim.__spec__ = getattr(torch_optim, "__spec__", types.SimpleNamespace())
torch_optim.Optimizer = getattr(torch_optim, "Optimizer", type("Optimizer", (), {}))
transformers_stub = sys.modules.setdefault(
    "transformers", types.ModuleType("transformers")
)
transformers_stub.__spec__ = getattr(
    transformers_stub, "__spec__", types.SimpleNamespace()
)
transformers_stub.PreTrainedModel = getattr(
    transformers_stub, "PreTrainedModel", type("PreTrainedModel", (), {})
)
transformers_stub.PreTrainedTokenizer = getattr(
    transformers_stub, "PreTrainedTokenizer", type("PreTrainedTokenizer", (), {})
)
accelerate_stub = sys.modules.setdefault("accelerate", types.ModuleType("accelerate"))
accelerate_stub.__spec__ = getattr(accelerate_stub, "__spec__", types.SimpleNamespace())
accelerate_stub.Accelerator = getattr(
    accelerate_stub, "Accelerator", type("Accelerator", (), {})
)
accelerate_utils = sys.modules.setdefault(
    "accelerate.utils", types.ModuleType("accelerate.utils")
)
accelerate_utils.__spec__ = getattr(
    accelerate_utils, "__spec__", types.SimpleNamespace()
)
accelerate_utils.DeepSpeedPlugin = getattr(
    accelerate_utils, "DeepSpeedPlugin", type("DeepSpeedPlugin", (), {})
)


def _import_zero_utils():
    import maxent_grpo.training.zero_utils as zero_utils

    return zero_utils


zero_utils = _import_zero_utils()


@pytest.fixture(autouse=True)
def _reload_zero_utils():
    """Reload zero_utils after each test to restore globals."""
    sys.modules.setdefault("training.zero_utils", zero_utils)
    yield
    sys.modules.setdefault("training.zero_utils", zero_utils)
    if getattr(zero_utils, "__spec__", None) is None:
        zero_utils.__spec__ = SimpleNamespace(name="training.zero_utils")
    importlib.reload(zero_utils)


def test_zero_utils_uses_cuda_fallback_when_torch_missing(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("torch"):
            raise ModuleNotFoundError("torch unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    mod = importlib.reload(zero_utils)
    assert mod.torch.cuda.is_available() is False
    assert hasattr(mod.nn, "Module")


def test_zero_utils_adds_cuda_fallback_when_missing_attr(monkeypatch):
    real_import = builtins.__import__
    torch_stub = types.SimpleNamespace()
    torch_stub.nn = types.SimpleNamespace(Module=object, Parameter=object)

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            return torch_stub
        if name == "torch.nn":
            return torch_stub.nn
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    mod = importlib.reload(zero_utils)
    assert mod.torch.cuda.is_available() is False


def test_cuda_fallback_empty_cache_returns_none():
    fallback = zero_utils._ensure_cuda_fallback()
    assert fallback.is_available() is False
    assert fallback.empty_cache() is None


def test_zero_partitioning_gradients_defaults_false():
    """Models without ZeRO markers should report no partitioning."""
    assert zero_utils._zero_partitioning_gradients(SimpleNamespace()) is False


def test_maybe_patch_zero_no_sync_skips_already_patched(monkeypatch):
    model = SimpleNamespace()
    setattr(model, zero_utils._NO_SYNC_PATCH_ATTR, True)
    monkeypatch.setattr(zero_utils, "_zero_stage", lambda _m: 3)
    assert zero_utils._maybe_patch_zero_no_sync(model) is False


def test_maybe_patch_zero_no_sync_requires_callable(monkeypatch):
    model = SimpleNamespace(no_sync=None)
    monkeypatch.setattr(zero_utils, "_zero_stage", lambda _m: 3)
    assert zero_utils._maybe_patch_zero_no_sync(model) is False


def test_maybe_patch_zero_no_sync_calls_original_when_partitioning_clears(monkeypatch):
    model = SimpleNamespace()
    calls = []

    @contextmanager
    def _orig_no_sync(*_a, **_k):
        calls.append("orig")
        yield

    seq = [True, False]
    monkeypatch.setattr(zero_utils, "_zero_stage", lambda _m: 3)
    monkeypatch.setattr(
        zero_utils, "_zero_partitioning_gradients", lambda _m: seq.pop(0)
    )
    model.no_sync = _orig_no_sync

    assert zero_utils._maybe_patch_zero_no_sync(model) is True
    with model.no_sync():
        pass
    assert calls == ["orig"]


def test_ensure_deepspeed_ready_handles_failure(monkeypatch):
    monkeypatch.setattr(zero_utils, "_DEEPSPEED_READY", None)
    monkeypatch.setattr(
        zero_utils,
        "require_deepspeed",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("missing")),
    )
    assert zero_utils._ensure_deepspeed_ready() is False
    assert zero_utils._DEEPSPEED_READY is False


def test_ensure_deepspeed_ready_populates_globals(monkeypatch):
    monkeypatch.setattr(zero_utils, "_DEEPSPEED_READY", None)

    @contextmanager
    def gathered_params(*_args, **_kwargs):
        yield

    status_cls = type("Status", (), {"NOT_AVAILABLE": "na"})
    ds_mod = types.SimpleNamespace(
        zero=types.SimpleNamespace(GatheredParameters=gathered_params)
    )
    partition_mod = types.SimpleNamespace(ZeroParamStatus=status_cls)
    modules = [ds_mod, partition_mod]

    def require_stub(*_args, **_kwargs):
        return modules.pop(0)

    monkeypatch.setattr(zero_utils, "require_deepspeed", require_stub)
    assert zero_utils._ensure_deepspeed_ready() is True
    assert zero_utils.ds_zero is ds_mod.zero
    assert zero_utils.ZeroParamStatus is status_cls


def test_ensure_deepspeed_ready_respects_cache(monkeypatch):
    monkeypatch.setattr(zero_utils, "_DEEPSPEED_READY", True)
    assert zero_utils._ensure_deepspeed_ready() is True
    monkeypatch.setattr(zero_utils, "_DEEPSPEED_READY", False)
    assert zero_utils._ensure_deepspeed_ready() is False


def test_zero_stage_and_partitioning_helpers_handle_edge_cases():
    class StageModel:
        def __init__(self, ret):
            self.ret = ret

        def zero_optimization_stage(self, arg=None):
            if arg is None:
                raise TypeError("needs target")
            return self.ret

    assert zero_utils._zero_stage(None) == 0
    assert zero_utils._zero_stage(StageModel(3)) == 3
    assert zero_utils._zero_stage(StageModel("bad")) == 0

    class PartitionModel:
        def zero_optimization_partition_gradients(self, arg=None):
            if arg is None:
                raise TypeError("arg missing")
            return True

    class AttrPartitionModel:
        partition_gradients = True

    assert zero_utils._zero_partitioning_gradients(PartitionModel()) is True
    assert zero_utils._zero_partitioning_gradients(AttrPartitionModel()) is True
    assert zero_utils._zero_partitioning_gradients(None) is False

    class CallableAttr:
        def partition_gradients(self, arg=None):
            if arg is None:
                raise TypeError("need arg")
            return False

    assert zero_utils._zero_partitioning_gradients(CallableAttr()) is False


def test_maybe_patch_zero_no_sync_warns(monkeypatch, caplog):
    class Model:
        def __init__(self):
            self.zero_optimization_stage = lambda: 3
            self.zero_optimization_partition_gradients = lambda: True

        @contextmanager
        def no_sync(self):
            yield "orig"

    model = Model()
    patched = zero_utils._maybe_patch_zero_no_sync(model)
    assert patched is True
    with caplog.at_level(logging.WARNING):
        with model.no_sync():
            pass
    assert getattr(model, zero_utils._NO_SYNC_PATCH_ATTR)


def test_maybe_patch_zero_no_sync_bypasses_when_stage_low():
    class Model:
        def __init__(self):
            self.zero_optimization_stage = lambda: 2

    assert zero_utils._maybe_patch_zero_no_sync(Model()) is False


def test_maybe_patch_zero_no_sync_delegates_when_no_partition(monkeypatch):
    class Model:
        def __init__(self):
            self.zero_optimization_stage = lambda: 3
            self.zero_optimization_partition_gradients = lambda: False

        @contextmanager
        def no_sync(self):
            yield "orig"

    monkeypatch.setattr(zero_utils, "_zero_partitioning_gradients", lambda _m: False)
    model = Model()
    patched = zero_utils._maybe_patch_zero_no_sync(model)
    assert patched is False
    with model.no_sync() as val:
        assert val == "orig"


def test_embedding_weight_needing_gather_respects_status(monkeypatch):
    monkeypatch.setattr(zero_utils, "_ensure_deepspeed_ready", lambda: True)
    zero_utils.ZeroParamStatus = type("Status", (), {"NOT_AVAILABLE": "na"})

    class Weight:
        def __init__(self, status):
            self.ndim = 3
            self.ds_status = status

    class Emb:
        def __init__(self, weight):
            self.weight = weight

    class Model:
        def __init__(self, weight):
            self.module = self
            self._emb = Emb(weight)

        def get_input_embeddings(self):
            return self._emb

    assert (
        zero_utils._embedding_weight_needing_gather(Model(Weight("available"))) is None
    )
    weight_ok = Weight("na")
    assert zero_utils._embedding_weight_needing_gather(Model(weight_ok)) is weight_ok


def test_embedding_weight_needing_gather_handles_missing_cases(monkeypatch):
    monkeypatch.setattr(zero_utils, "_ensure_deepspeed_ready", lambda: False)
    assert zero_utils._embedding_weight_needing_gather(object()) is None
    monkeypatch.setattr(zero_utils, "_ensure_deepspeed_ready", lambda: True)
    zero_utils.ZeroParamStatus = type("Status", (), {"NOT_AVAILABLE": "na"})

    class Model:
        def __init__(self, weight):
            self.module = self
            self._emb = types.SimpleNamespace(weight=weight)

        def get_input_embeddings(self):
            return self._emb

    assert zero_utils._embedding_weight_needing_gather(Model(None)) is None
    ndim2_weight = types.SimpleNamespace(ndim=2, ds_status="na")
    assert zero_utils._embedding_weight_needing_gather(Model(ndim2_weight)) is None
    blocked = types.SimpleNamespace(ndim=3, ds_status="blocked")
    assert zero_utils._embedding_weight_needing_gather(Model(blocked)) is None


def test_maybe_zero_gather_embedding_invokes_gather(monkeypatch):
    calls: dict[str, object] = {}

    @contextmanager
    def gathered(params, modifier_rank=None):
        calls["params"] = list(params)
        yield

    zero_utils.ds_zero = types.SimpleNamespace(GatheredParameters=gathered)
    monkeypatch.setattr(zero_utils, "_ensure_deepspeed_ready", lambda: True)
    zero_utils.ZeroParamStatus = type("Status", (), {"NOT_AVAILABLE": "na"})
    weight = types.SimpleNamespace(ndim=3, ds_status="na")
    embedder = types.SimpleNamespace(weight=weight)
    model = types.SimpleNamespace(
        module=types.SimpleNamespace(get_input_embeddings=lambda: embedder)
    )

    with zero_utils._maybe_zero_gather_embedding(model):
        calls["entered"] = True

    assert calls["params"] == [weight]
    assert calls["entered"] is True


def test_maybe_zero_gather_embedding_handles_missing_components(monkeypatch):
    monkeypatch.setattr(zero_utils, "_ensure_deepspeed_ready", lambda: False)
    with zero_utils._maybe_zero_gather_embedding(object()):
        pass
    zero_utils.ds_zero = None
    monkeypatch.setattr(zero_utils, "_ensure_deepspeed_ready", lambda: True)
    with zero_utils._maybe_zero_gather_embedding(object()):
        pass
    zero_utils.ds_zero = types.SimpleNamespace(GatheredParameters=None)
    with zero_utils._maybe_zero_gather_embedding(object()):
        pass
    zero_utils.ds_zero = types.SimpleNamespace(
        GatheredParameters=lambda *_a, **_k: (_ for _ in ()).throw(TypeError)
    )
    weightless = types.SimpleNamespace(
        module=types.SimpleNamespace(get_input_embeddings=lambda: None)
    )
    with zero_utils._maybe_zero_gather_embedding(weightless):
        pass


def test_zero_param_list_and_gather_params(monkeypatch):
    assert zero_utils._zero_param_list(None) == []

    class NoParams:
        pass

    assert zero_utils._zero_param_list(NoParams()) == []

    class BadParams:
        def parameters(self):
            raise TypeError("bad")

    assert zero_utils._zero_param_list(BadParams()) == []

    class WithParams:
        def __init__(self):
            self.p = object()

        def parameters(self):
            return [self.p]

    with_params = WithParams()
    assert zero_utils._zero_param_list(with_params) == [with_params.p]

    calls: dict[str, object] = {}

    @contextmanager
    def gathered(params, modifier_rank=None):
        calls["params"] = list(params)
        yield

    zero_utils.ds_zero = types.SimpleNamespace(GatheredParameters=gathered)
    monkeypatch.setattr(zero_utils, "_ensure_deepspeed_ready", lambda: True)
    monkeypatch.setattr(zero_utils, "_zero_stage", lambda _m: 0)
    cuda_calls: list[str] = []
    torch_stub = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True, empty_cache=lambda: cuda_calls.append("cleared")
        )
    )
    monkeypatch.setattr(zero_utils, "torch", torch_stub)

    class Model:
        def __init__(self):
            self._params = [types.SimpleNamespace(ds_id=1), object()]

        def parameters(self):
            return list(self._params)

    with zero_utils._maybe_zero_gather_params(Model(), enabled=True):
        calls["entered"] = True

    assert calls["params"][0].ds_id == 1
    assert "cleared" in cuda_calls


def test_maybe_zero_gather_params_no_gather_class(monkeypatch):
    zero_utils.ds_zero = types.SimpleNamespace(GatheredParameters=None)
    monkeypatch.setattr(zero_utils, "_ensure_deepspeed_ready", lambda: True)
    with zero_utils._maybe_zero_gather_params(types.SimpleNamespace(), enabled=True):
        pass


def test_maybe_zero_gather_params_handles_disabled_and_stage(monkeypatch):
    with zero_utils._maybe_zero_gather_params(types.SimpleNamespace(), enabled=False):
        pass
    zero_utils.ds_zero = types.SimpleNamespace(
        GatheredParameters=lambda *_a, **_k: (_ for _ in ()).throw(TypeError)
    )
    monkeypatch.setattr(zero_utils, "_ensure_deepspeed_ready", lambda: True)
    model = types.SimpleNamespace(parameters=lambda: [])
    with zero_utils._maybe_zero_gather_params(model, enabled=True):
        pass

    class Model:
        def __init__(self):
            self.p = object()

        def parameters(self):
            return [self.p]

    calls: dict[str, object] = {}

    @contextmanager
    def gathered(params, modifier_rank=None):
        calls["params"] = list(params)
        yield

    zero_utils.ds_zero = types.SimpleNamespace(GatheredParameters=gathered)
    monkeypatch.setattr(zero_utils, "_zero_stage", lambda _m: 1)
    monkeypatch.setattr(
        zero_utils,
        "torch",
        types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                is_available=lambda: False, empty_cache=lambda: None
            )
        ),
    )
    model_instance = Model()
    with zero_utils._maybe_zero_gather_params(model_instance, enabled=True):
        pass
    assert calls["params"][0] is model_instance.p