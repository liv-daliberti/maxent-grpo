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

Unit tests for helpers in :mod:`training.generation.helpers`.
"""

import sys  # noqa: E402  # module-level stubs set up first in some environments
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

torch_module = sys.modules.setdefault("torch", ModuleType("torch"))
torch_module.__spec__ = getattr(torch_module, "__spec__", SimpleNamespace())
torch_module.__path__ = getattr(torch_module, "__path__", [])
if not hasattr(torch_module, "Tensor"):
    torch_module.Tensor = type("Tensor", (), {})
torch_utils_module = sys.modules.setdefault("torch.utils", ModuleType("torch.utils"))
torch_utils_module.__spec__ = getattr(torch_utils_module, "__spec__", SimpleNamespace())
torch_utils_module.__path__ = getattr(torch_utils_module, "__path__", [])
torch_data_module = sys.modules.setdefault(
    "torch.utils.data", ModuleType("torch.utils.data")
)
torch_data_module.__spec__ = getattr(torch_data_module, "__spec__", SimpleNamespace())
torch_data_module.__path__ = getattr(torch_data_module, "__path__", [])
if not hasattr(torch_data_module, "DataLoader"):

    class _DataLoader:  # minimal stub
        pass

    torch_data_module.DataLoader = _DataLoader
if not hasattr(torch_data_module, "Sampler"):

    class _Sampler:
        pass

    torch_data_module.Sampler = _Sampler
torch_utils_module.data = torch_data_module
torch_module.utils = torch_utils_module
torch_optim_module = sys.modules.setdefault("torch.optim", ModuleType("torch.optim"))
torch_optim_module.__spec__ = getattr(torch_optim_module, "__spec__", SimpleNamespace())
torch_optim_module.Optimizer = type("Optimizer", (), {})
torch_module.optim = torch_optim_module
torch_nn_module = sys.modules.setdefault("torch.nn", ModuleType("torch.nn"))
torch_nn_module.__spec__ = getattr(torch_nn_module, "__spec__", SimpleNamespace())
torch_nn_functional = sys.modules.setdefault(
    "torch.nn.functional",
    ModuleType("torch.nn.functional"),
)
torch_nn_functional.__spec__ = getattr(
    torch_nn_functional, "__spec__", SimpleNamespace()
)
if not hasattr(torch_nn_functional, "log_softmax"):

    def _log_softmax(*_args, **_kwargs):
        raise NotImplementedError  # should never run in these unit tests

    torch_nn_functional.log_softmax = _log_softmax

accelerate_module = sys.modules.setdefault("accelerate", ModuleType("accelerate"))
accelerate_module.__spec__ = getattr(accelerate_module, "__spec__", SimpleNamespace())
if not hasattr(accelerate_module, "Accelerator"):

    class _Accel:
        def __init__(self, **_kwargs):
            self.is_main_process = True
            self.process_index = 0

    accelerate_module.Accelerator = _Accel

transformers_module = sys.modules.setdefault("transformers", ModuleType("transformers"))
transformers_module.__spec__ = getattr(
    transformers_module, "__spec__", SimpleNamespace()
)
if not hasattr(transformers_module, "PreTrainedModel"):

    class _PreTrainedModel:
        pass

    class _PreTrainedTokenizer:
        pass

    transformers_module.PreTrainedModel = _PreTrainedModel
    transformers_module.PreTrainedTokenizer = _PreTrainedTokenizer

from maxent_grpo.training.generation.helpers import (  # noqa: E402  import after torch stub
    CompletionGenerator,
    _broadcast_object_list,
    _gather_object_list,
)


class _DummyAccelerator:
    """Accelerator stub that records gather/broadcast invocations."""

    def __init__(self) -> None:
        self.gather_calls = []
        self.broadcast_calls = []

    def gather_object(self, value):
        """Mimic HF accelerator gather_object."""
        self.gather_calls.append(value)
        # Pretend there are two ranks.
        return [value, ["peer"]]

    def broadcast_object_list(self, payload, src=0):
        """Mimic HF accelerator broadcast_object_list."""
        # Record a serialized copy to ensure mutation in place happened.
        self.broadcast_calls.append((list(payload), src))


def test_gather_object_list_with_accelerator_api():
    accelerator = _DummyAccelerator()
    value = ["prompt"]
    gathered = _gather_object_list(accelerator, value)
    assert gathered == [value, ["peer"]]
    assert accelerator.gather_calls == [value]


def test_broadcast_object_list_with_accelerator_api():
    accelerator = _DummyAccelerator()
    payload = [["a"], ["b"]]
    _broadcast_object_list(accelerator, payload, src=1)
    assert accelerator.broadcast_calls == [([["a"], ["b"]], 1)]


@pytest.fixture(name="dist_stub")
def _dist_stub():
    """Return a simple torch.distributed stub for monkeypatching."""

    class _Dist:
        def __init__(self):
            self.gathered = None
            self.broadcasted = None

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def get_world_size():
            return 2

        def all_gather_object(self, output_list, input_obj):
            self.gathered = input_obj
            output_list[0] = input_obj
            output_list[1] = ["peer"]

        def broadcast_object_list(self, payload, src=0):
            self.broadcasted = (list(payload), src)

    return _Dist()


def test_gather_object_list_falls_back_to_dist(monkeypatch, dist_stub):
    accelerator = SimpleNamespace(gather_object=None)  # lacks gather_object
    monkeypatch.setattr(
        "maxent_grpo.training.generation.helpers.dist",
        dist_stub,
        raising=False,
    )
    gathered = _gather_object_list(accelerator, ["local"])
    assert gathered and gathered[0] == ["local"]
    if len(gathered) > 1:
        assert gathered[1] == ["peer"]
    assert dist_stub.gathered in (["local"], None)


def test_broadcast_object_list_falls_back_to_dist(monkeypatch, dist_stub):
    accelerator = SimpleNamespace(broadcast_object_list=None)
    monkeypatch.setattr(
        "maxent_grpo.training.generation.helpers.dist",
        dist_stub,
        raising=False,
    )
    payload = [["x"], ["y"]]
    _broadcast_object_list(accelerator, payload, src=0)
    if dist_stub.broadcasted is not None:
        assert dist_stub.broadcasted == (payload, 0)


def test_resolve_local_counts_validates_overrides():
    prompts = ["p1", "p2"]
    overrides = [1, 3]
    assert CompletionGenerator._resolve_local_counts(prompts, 2, overrides) == overrides
    with pytest.raises(ValueError):
        CompletionGenerator._resolve_local_counts(["p1"], 2, [1, 2])


def test_merge_group_chunk_merges_text_and_meta():
    chunk = [["a"], ["b"]]
    meta_chunk = [[SimpleNamespace(score=1.0)], [SimpleNamespace(score=2.0)]]
    merged, merged_meta = CompletionGenerator._merge_group_chunk(
        chunk, meta_chunk, requested_n=2
    )
    assert merged == ["a", "b"]
    assert merged_meta == [SimpleNamespace(score=1.0), SimpleNamespace(score=2.0)]
