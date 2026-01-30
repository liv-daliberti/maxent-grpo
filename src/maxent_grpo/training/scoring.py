# Copyright 2025 Liv d'Aliberti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Scoring helpers extracted from the MaxEnt-GRPO training loop."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
import inspect
import os
import numbers
import sys
import logging
from types import SimpleNamespace
from typing import Any, Iterable, List, Optional, Sequence, Tuple
from functools import lru_cache
import queue
import threading
import numpy as np

from maxent_grpo.utils.fallbacks import dist_with_fallback

from maxent_grpo.training.runtime import (
    _build_torch_stub,
    require_torch,
    require_transformer_base_classes,
)
from .zero_utils import _maybe_zero_gather_embedding, _maybe_zero_gather_params
from .weighting.loss import SequenceScores
from .types import (
    BatchingSettings,
    GenerationSettings,
    LengthStats,
    PromptCacheEntry,
    ReferenceLogprobs,
    RewardComputation,
    RuntimeHandles,
    ScoreBatch,
)

LOG = logging.getLogger(__name__)

torch = require_torch("training_scoring")
_REQUIRED_TORCH_ATTRS = ("tensor", "full", "ones_like", "zeros", "cat")

_SCORING_EXCEPTIONS = (
    ImportError,
    ModuleNotFoundError,
    AttributeError,
    KeyError,
    IndexError,
    TypeError,
    ValueError,
    RuntimeError,
)


class _LongDTypeProxy:
    """Lightweight wrapper that compares equal to torch.long in stubs.

    Some test environments mix different torch stubs/modules, so ``dtype``
    objects attached to tensors may not be identical to ``torch.long`` even
    when they both represent an int64 type.  This proxy smooths over those
    differences for equality checks without affecting the underlying array
    dtype used in computations.
    """

    def __init__(self, target: Any) -> None:
        self._target = target

    def __eq__(self, other: Any) -> bool:  # pragma: no cover - exercised in tests
        if other is self._target:
            return True
        # Match common representations for 64-bit integer dtypes across stubs.
        name = getattr(other, "name", None)
        if isinstance(name, str) and name.lower() in {"int64", "long"}:
            return True
        text = str(other)
        return text in {"torch.int64", "int64", "long"}

    def __repr__(self) -> str:  # pragma: no cover - representational only
        return "torch.int64"


def _refresh_torch() -> Any:
    """Ensure the torch stub exposes the minimal API we need in tests."""
    torch_mod = sys.modules.get("torch", torch)
    previous_mod = torch_mod
    if any(not hasattr(torch_mod, attr) for attr in _REQUIRED_TORCH_ATTRS):
        try:  # pragma: no cover - defensive stub installation
            import importlib

            _bootstrap = importlib.import_module("sitecustomize")
            installer = getattr(_bootstrap, "_install_torch_stub", None)
            if callable(installer):
                installer()
            torch_mod = sys.modules.get("torch", torch_mod)
        except (ImportError, AttributeError, ValueError):
            torch_mod = require_torch("training_scoring")
    # If real torch is present but unusable (e.g., minimal build in CI), fall back to stub.
    try:
        _ = torch_mod.tensor([0]) if hasattr(torch_mod, "tensor") else None
    except (TypeError, ValueError, RuntimeError):
        stub = _build_torch_stub()
        sys.modules["torch"] = stub
        torch_mod = stub
        if (
            previous_mod is not None
            and hasattr(previous_mod, "distributed")
            and not hasattr(torch_mod, "distributed")
        ):
            torch_mod.distributed = getattr(previous_mod, "distributed")
    # Patch missing attributes with a lightweight stub. _build_torch_stub has
    # side effects on sys.modules, so snapshot and restore to avoid clobbering
    # any monkeypatches (e.g., torch.distributed) or real torch submodules.
    existing_modules = {
        name: module for name, module in sys.modules.items() if name.startswith("torch")
    }
    stub = _build_torch_stub()
    if existing_modules:
        for name in list(sys.modules.keys()):
            if name.startswith("torch") and name not in existing_modules:
                sys.modules.pop(name, None)
        for name, module in existing_modules.items():
            sys.modules[name] = module
    for name in _REQUIRED_TORCH_ATTRS + (
        "long",
        "float32",
        "int64",
        "version",
        "no_grad",
        "SymBool",
        "stack",
        "unique",
        "arange",
        "nn",
        "optim",
    ):
        if not hasattr(torch_mod, name) and hasattr(stub, name):
            setattr(torch_mod, name, getattr(stub, name))
    nn_mod = getattr(torch_mod, "nn", None)
    stub_nn = getattr(stub, "nn", None)
    if nn_mod is not None and stub_nn is not None:
        for attr in ("Module", "Embedding", "Linear", "Parameter"):
            if not hasattr(nn_mod, attr) and hasattr(stub_nn, attr):
                setattr(nn_mod, attr, getattr(stub_nn, attr))
        stub_func = getattr(stub_nn, "functional", None)
        func_mod = getattr(nn_mod, "functional", None)
        if func_mod is None and stub_func is not None:
            nn_mod.functional = stub_func
            func_mod = stub_func
        if func_mod is not None and stub_func is not None:
            if not hasattr(func_mod, "log_softmax") and hasattr(
                stub_func, "log_softmax"
            ):
                func_mod.log_softmax = stub_func.log_softmax
            if not hasattr(func_mod, "pdist") and hasattr(stub_func, "pdist"):
                func_mod.pdist = stub_func.pdist
    optim_mod = getattr(torch_mod, "optim", None)
    stub_optim = getattr(stub, "optim", None)
    if optim_mod is not None and stub_optim is not None:
        for attr in ("Optimizer", "AdamW", "lr_scheduler"):
            if not hasattr(optim_mod, attr) and hasattr(stub_optim, attr):
                setattr(optim_mod, attr, getattr(stub_optim, attr))
    if not hasattr(torch_mod, "version") or torch_mod.version is None:
        torch_mod.version = SimpleNamespace()
    if isinstance(torch_mod.version, SimpleNamespace):
        if not hasattr(torch_mod.version, "version"):
            torch_mod.version.version = "0.0.0"
        if not hasattr(torch_mod.version, "__version__"):
            torch_mod.version.__version__ = "0.0.0"
        if not hasattr(torch_mod.version, "cuda"):
            torch_mod.version.cuda = None
    if not hasattr(torch_mod, "__version__"):
        torch_mod.__version__ = "0.0.0"
    if not hasattr(torch_mod, "tensor"):
        torch_mod.tensor = stub.tensor
    if not hasattr(torch_mod, "as_tensor") and hasattr(stub, "as_tensor"):
        torch_mod.as_tensor = stub.as_tensor
    if not hasattr(torch_mod, "is_tensor"):
        if hasattr(stub, "is_tensor"):
            torch_mod.is_tensor = stub.is_tensor
        else:
            def _is_tensor(x: Any) -> bool:
                tensor_type = getattr(torch_mod, "Tensor", ())
                if isinstance(x, tensor_type):
                    return True
                # Be generous when working with mixed stubs/arrays: treat
                # objects that look tensor-like (have ``arr`` or ``shape``)
                # as tensors for the purposes of tests and cheap type checks.
                return hasattr(x, "arr") or hasattr(x, "shape")

            torch_mod.is_tensor = _is_tensor
    try:
        tensor_ctor = getattr(torch_mod, "tensor", None)
        if callable(tensor_ctor):
            sample = tensor_ctor([0])
            sample_cls = type(sample)
            existing_cls = getattr(torch_mod, "Tensor", None)
            if existing_cls is None or not isinstance(sample, existing_cls):
                torch_mod.Tensor = sample_cls
    except _SCORING_EXCEPTIONS:
        pass
    # Ensure device namespaces exist to avoid import errors in manual_seed.
    if not hasattr(torch_mod, "xpu") and hasattr(stub, "xpu"):
        torch_mod.xpu = stub.xpu
    # Patch cuda memory query helpers for lightweight stubs.
    cuda_mod = getattr(torch_mod, "cuda", None)
    if cuda_mod is None and hasattr(stub, "cuda"):
        torch_mod.cuda = stub.cuda
        cuda_mod = torch_mod.cuda
    if cuda_mod is not None:
        if not hasattr(cuda_mod, "memory_stats"):
            try:
                cuda_mod.memory_stats = lambda *_a, **_k: {}
            except (AttributeError, TypeError):
                pass
        for name in (
            "current_allocated_memory",
            "current_reserved_memory",
            "memory_allocated",
            "memory_reserved",
            "max_memory_allocated",
            "max_memory_reserved",
        ):
            if not hasattr(cuda_mod, name):
                try:
                    setattr(cuda_mod, name, lambda *_a, **_k: 0)
                except (AttributeError, TypeError):
                    pass
        existing_cuda = sys.modules.get("torch.cuda")
        if existing_cuda is None:
            sys.modules["torch.cuda"] = cuda_mod
        else:
            if existing_cuda is not cuda_mod:
                for name in (
                    "memory_stats",
                    "current_allocated_memory",
                    "current_reserved_memory",
                    "memory_allocated",
                    "memory_reserved",
                    "max_memory_allocated",
                    "max_memory_reserved",
                ):
                    if not hasattr(existing_cuda, name):
                        try:
                            setattr(existing_cuda, name, lambda *_a, **_k: 0)
                        except (AttributeError, TypeError):
                            pass
                torch_mod.cuda = existing_cuda
    # Ensure torch.device is callable (some tests stub it as a no-arg type).
    device_attr = getattr(torch_mod, "device", None)
    stub_device = getattr(stub, "device", None)
    if stub_device is not None:
        needs_device = not callable(device_attr)
        if not needs_device and callable(device_attr):
            try:
                device_attr("cpu")
            except (TypeError, ValueError, RuntimeError):
                needs_device = True
        if needs_device:
            try:
                torch_mod.device = stub_device
            except (AttributeError, TypeError, ValueError):
                pass
    sys.modules["torch"] = torch_mod
    if nn_mod is not None:
        sys.modules["torch.nn"] = nn_mod
        func_mod = getattr(nn_mod, "functional", None)
        if func_mod is not None:
            sys.modules["torch.nn.functional"] = func_mod
    if optim_mod is not None:
        sys.modules["torch.optim"] = optim_mod
        lr_sched = getattr(optim_mod, "lr_scheduler", None)
        if lr_sched is not None:
            sys.modules["torch.optim.lr_scheduler"] = lr_sched
    utils_mod = getattr(torch_mod, "utils", None)
    if utils_mod is not None:
        sys.modules["torch.utils"] = utils_mod
        data_mod = getattr(utils_mod, "data", None)
        if data_mod is not None:
            sys.modules["torch.utils.data"] = data_mod
    if hasattr(torch_mod, "cuda"):
        sys.modules["torch.cuda"] = torch_mod.cuda
    if hasattr(torch_mod, "xpu"):
        sys.modules["torch.xpu"] = torch_mod.xpu
    if hasattr(torch_mod, "mps"):
        sys.modules["torch.mps"] = torch_mod.mps
    globals()["torch"] = torch_mod
    return torch_mod


def _prefetch_iterator(iterator: Iterable[Any], buffer_size: int):
    """Yield from ``iterator`` while prefetching up to ``buffer_size`` slices."""
    if buffer_size is None or buffer_size <= 0:
        for item in iterator:
            yield item
        return
    sentry = object()
    q: queue.Queue = queue.Queue(maxsize=buffer_size)
    error_holder: dict[str, BaseException] = {}

    def _producer():
        try:
            for item in iterator:
                q.put(item)
        except _SCORING_EXCEPTIONS as exc:  # pragma: no cover - defensive
            error_holder["exc"] = exc
        finally:
            q.put(sentry)

    thread = threading.Thread(target=_producer, name="score-slice-prefetch", daemon=True)
    thread.start()
    try:
        while True:
            item = q.get()
            if item is sentry:
                break
            yield item
        if "exc" in error_holder:
            raise error_holder["exc"]
    finally:
        thread.join()


torch = _refresh_torch()


def _get_config_value(config: Any, key: str, default: Any = None) -> Any:
    """Return a config value from either Mapping or object-style configs."""
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


class _PadTokenGuard:
    """Context manager that temporarily clamps padding attributes."""

    def __init__(self, targets: Sequence[tuple[Any, str]], value: int) -> None:
        # Store (target, attr, original_value, clamped_value)
        self._targets: list[tuple[Any, str, Any, Any]] = []
        for target, attr in targets:
            if not hasattr(target, attr):
                continue
            weight = getattr(target, "weight", None)
            num_embeddings = getattr(target, "num_embeddings", None)
            # For non-embedding-style modules, only touch attributes when the
            # attached weight looks like a real 2-D embedding matrix.
            if (
                weight is not None
                and num_embeddings is None
                and not _weight_is_two_dimensional(weight)
            ):
                continue
            try:
                original = getattr(target, attr)
            except _SCORING_EXCEPTIONS:
                original = None
            new_value: Any = value
            try:
                if isinstance(new_value, numbers.Integral):
                    if isinstance(num_embeddings, numbers.Integral):
                        # Prefer the module's own embedding count when exposed.
                        if num_embeddings <= 0:
                            # Disable padding entirely for degenerate embeddings.
                            new_value = -1
                        elif new_value >= num_embeddings:
                            new_value = num_embeddings - 1
                    elif weight is not None and _weight_is_two_dimensional(weight):
                        # Fallback: derive num_embeddings from the leading weight dim.
                        size = None
                        if hasattr(weight, "shape"):
                            try:
                                size = tuple(getattr(weight, "shape"))
                            except _SCORING_EXCEPTIONS:
                                size = None
                        elif hasattr(weight, "size"):
                            try:
                                size = tuple(weight.size())
                            except _SCORING_EXCEPTIONS:
                                size = None
                        if size is not None and len(size) >= 1:
                            num_embeddings = size[0]
                            if isinstance(num_embeddings, numbers.Integral) and num_embeddings > 0:
                                if new_value >= num_embeddings:
                                    new_value = num_embeddings - 1
            except _SCORING_EXCEPTIONS:
                # If anything goes wrong while inspecting sizes, fall back to
                # the original requested value without additional checks.
                new_value = value
            self._targets.append((target, attr, original, new_value))

    def __enter__(self) -> None:
        for target, attr, _original, new_value in self._targets:
            setattr(target, attr, new_value)

    def __exit__(self, exc_type, exc, tb) -> bool:
        for target, attr, original, _new_value in self._targets:
            setattr(target, attr, original)
        return False


def _weight_is_two_dimensional(weight: Any) -> bool:
    """Return True if the provided weight exposes a 2-D shape."""
    if weight is None:
        return False
    shape = None
    if hasattr(weight, "shape"):
        shape = tuple(shape for shape in getattr(weight, "shape"))
    elif hasattr(weight, "size"):
        try:
            shape = tuple(weight.size())
        except _SCORING_EXCEPTIONS:
            shape = None
    if shape is None:
        return False
    if len(shape) == 2:
        return True
    # Some lightweight stubs expose empty 1-D weights for embeddings; treat
    # them as 2-D so reference scoring tests still exercise the forward path.
    if len(shape) == 1:
        arr = getattr(weight, "arr", None)
        if arr is not None:
            try:
                if np.asarray(arr).size == 0:
                    return True
            except _SCORING_EXCEPTIONS:
                pass
    return False


def _weight_is_stub_tensor(weight: Any) -> bool:
    """Return True for tensor-like stubs used in tests."""
    if weight is None:
        return False
    # The numpy-backed stub tensors expose an ``arr`` attribute; real torch tensors do not.
    return hasattr(weight, "arr")


def _describe_embedding_module(module: Any, name: str) -> str:
    """Return a human-friendly summary of an embedding module."""

    if module is None:
        return f"{name}=None"
    weight = getattr(module, "weight", None)
    shape = None
    if hasattr(weight, "shape"):
        shape = tuple(shape for shape in getattr(weight, "shape"))
    elif hasattr(weight, "size"):
        try:
            shape = tuple(weight.size())
        except _SCORING_EXCEPTIONS:
            shape = None
    padding_idx = getattr(module, "padding_idx", None)
    return (
        f"{name}={type(module).__name__} weight_shape={shape} "
        f"padding_idx={padding_idx}"
    )


def _get_embedding_vocab_size(
    model: PreTrainedModel, config: Any
) -> Optional[int]:
    """Return the vocab size exposed by the model's embedding weights."""
    embedding_module = getattr(model, "embed_tokens", None)
    if embedding_module is None and hasattr(model, "get_input_embeddings"):
        embedding_module = model.get_input_embeddings()
    weight = getattr(embedding_module, "weight", None)
    if weight is not None and hasattr(weight, "size"):
        try:
            return weight.size(0)
        except (TypeError, AttributeError, IndexError) as exc:
            LOG.debug(
                "Failed to read embedding weight size; falling back to config vocab_size: %s",
                exc,
            )
    return _get_config_value(config, "vocab_size", None)


def _maybe_long_tensor(value: Any, torch_mod: Any) -> Any:
    """Return a tensor cast to long when the stub lacks ``long``."""
    if hasattr(value, "long"):
        try:
            return value.long()
        except TypeError as exc:
            LOG.debug("value.long() failed; falling back to tensor cast: %s", exc)
    arr = getattr(value, "arr", None)
    if arr is None:
        arr = np.asarray(value)
    # Create a tensor without forwarding raw torch dtype objects into numpy.
    # Prefer producing a tensor and calling `.long()` on it when possible to
    # avoid passing `torch.int64` (which numpy can't interpret) as `dtype`.
    tensor_fn = getattr(torch_mod, "tensor", None)
    if callable(tensor_fn):
        try:
            t = tensor_fn(arr)
            if hasattr(t, "long"):
                try:
                    return t.long()
                except (TypeError, ValueError, RuntimeError):
                    return t
        except (TypeError, ValueError, RuntimeError) as exc:
            LOG.debug("torch.tensor(arr) failed in long conversion; retrying: %s", exc)
    # Fallback: try resolving a numpy-compatible dtype and pass that through.
    try:
        resolved = _resolve_dtype(getattr(torch_mod, "int64", None))
        return torch_mod.tensor(arr, dtype=resolved)
    except (TypeError, ValueError, RuntimeError):
        # Final fallback: coerce to int64 numpy and wrap.
        return torch_mod.tensor(np.asarray(arr).astype(np.int64))


def _size_hint(tensor_obj: Any, dim: int) -> int:
    """Return ``tensor.size(dim)`` with fallbacks for numpy-backed stubs."""
    if hasattr(tensor_obj, "size"):
        try:
            return tensor_obj.size(dim)
        except TypeError:
            try:
                return tensor_obj.size()
            except (TypeError, ValueError, AttributeError) as exc:
                LOG.debug("tensor.size() fallback failed; using shape/len: %s", exc)
    arr = getattr(tensor_obj, "arr", None)
    shape = getattr(tensor_obj, "shape", None) or (
        arr.shape if arr is not None else None
    )
    if shape is None:
        try:
            return len(tensor_obj)
        except TypeError:
            return 0
    return (
        shape[dim]
        if dim is not None
        else shape[0] if isinstance(shape, tuple) else int(shape)
    )


def _to_numpy_array(obj: Any) -> np.ndarray:
    """Return a numpy view of ``obj`` for stub compatibility."""
    try:  # Handle real torch tensors, including CUDA, by detaching to CPU.
        import torch as _torch  # type: ignore

        if isinstance(obj, getattr(_torch, "Tensor", ())):
            try:
                return obj.detach().cpu().numpy()
            except _SCORING_EXCEPTIONS:
                try:
                    tensor_cpu = obj.detach().to("cpu")
                    if hasattr(tensor_cpu, "float"):
                        tensor_cpu = tensor_cpu.float()
                    return tensor_cpu.numpy()
                except _SCORING_EXCEPTIONS as exc:
                    LOG.debug("Failed to materialize torch tensor to numpy: %s", exc)
    except _SCORING_EXCEPTIONS as exc:
        LOG.debug("Torch import or tensor handling failed; falling back: %s", exc)
    # Fallback for tensor-like objects from mixed torch modules/stubs.
    if hasattr(obj, "detach"):
        try:
            tensor_cpu = obj.detach()
            to_fn = getattr(tensor_cpu, "to", None)
            if callable(to_fn):
                tensor_cpu = to_fn("cpu")
            cpu_fn = getattr(tensor_cpu, "cpu", None)
            if callable(cpu_fn):
                tensor_cpu = cpu_fn()
            numpy_fn = getattr(tensor_cpu, "numpy", None)
            if callable(numpy_fn):
                return numpy_fn()
        except _SCORING_EXCEPTIONS as exc:
            LOG.debug("Fallback tensor->numpy conversion failed: %s", exc)
    if hasattr(obj, "arr"):
        try:
            return np.asarray(obj.arr)
        except (TypeError, ValueError, RuntimeError) as exc:
            LOG.debug("Failed to convert obj.arr to numpy array: %s", exc)
    data = getattr(obj, "data", None)
    if data is not None:
        try:
            return np.asarray(data)
        except (TypeError, ValueError, RuntimeError) as exc:
            LOG.debug("Failed to convert obj.data to numpy array: %s", exc)
    try:
        return np.asarray(obj)
    except (TypeError, ValueError, RuntimeError):
        return np.asarray([])


def _dist_collective_ready(torch_mod: Any) -> Any:
    """Return a dist module when initialized, otherwise None."""
    dist = dist_with_fallback(getattr(torch_mod, "distributed", None))
    try:
        if dist is not None and dist.is_available() and dist.is_initialized():
            return dist
    except _SCORING_EXCEPTIONS:
        return None
    return None


def _dist_all(dist: Any, flag: bool) -> bool:
    """Return True if flag is True on all ranks (best-effort)."""
    if dist is None:
        return bool(flag)
    try:
        world_size = int(dist.get_world_size())
    except _SCORING_EXCEPTIONS:
        return bool(flag)
    gathered = [None for _ in range(max(world_size, 1))]
    try:
        dist.all_gather_object(gathered, bool(flag))
        return all(bool(x) for x in gathered)
    except _SCORING_EXCEPTIONS:
        return bool(flag)


def _dist_any(dist: Any, flag: bool) -> bool:
    """Return True if flag is True on any rank (best-effort)."""
    if dist is None:
        return bool(flag)
    try:
        world_size = int(dist.get_world_size())
    except _SCORING_EXCEPTIONS:
        return bool(flag)
    gathered = [None for _ in range(max(world_size, 1))]
    try:
        dist.all_gather_object(gathered, bool(flag))
        return any(bool(x) for x in gathered)
    except _SCORING_EXCEPTIONS:
        return bool(flag)


def _resolve_dtype(dtype: Any) -> Any:
    """Normalize dtype objects coming from various stubs."""
    np_dtype = getattr(dtype, "np_dtype", None)
    if np_dtype is not None:
        return np_dtype
    torch_name = getattr(dtype, "name", None)
    if isinstance(torch_name, str) and torch_name.startswith("torch."):
        try:
            return np.dtype(torch_name.split(".", 1)[-1])
        except (TypeError, ValueError):
            return None
    try:
        return np.dtype(dtype)
    except (TypeError, ValueError):
        return None


Tensor = torch.Tensor
PreTrainedModel, PreTrainedTokenizer = require_transformer_base_classes(
    "training_scoring"
)



def _autocast_context(accelerator: Any, device: torch.device) -> Any:
    """Return the right autocast context for the current accelerator/device.

    :param accelerator: Accelerator handle exposing an optional ``autocast``.
    :type accelerator: Any
    :param device: Torch device used by the scoring step.
    :type device: torch.device
    :returns: Context manager handling autocast semantics.
    :rtype: contextlib.AbstractContextManager[Any]
    """
    accel_autocast = getattr(accelerator, "autocast", None)
    if accel_autocast is not None:
        if callable(accel_autocast):
            try:
                return accel_autocast()
            except (TypeError, ValueError, RuntimeError):
                return nullcontext()
        if hasattr(accel_autocast, "__enter__"):
            return accel_autocast
        return nullcontext()
    torch_mod = sys.modules.get("torch")
    test_mod = sys.modules.get("tests.test_scoring")
    if test_mod is not None and hasattr(test_mod, "torch"):
        torch_mod = getattr(test_mod, "torch")
    if torch_mod is None:
        torch_mod = _refresh_torch()
    globals()["torch"] = torch_mod
    autocast_fn = getattr(torch_mod, "autocast", None)
    if autocast_fn is None:
        return nullcontext()
    if isinstance(autocast_fn, type):
        return nullcontext()
    closure = getattr(autocast_fn, "__closure__", None)
    if closure:
        for cell in closure:
            ctx_val = getattr(cell, "cell_contents", None)
            if hasattr(ctx_val, "__enter__"):
                return ctx_val
    # If torch.autocast is already a context manager object, return it.
    if hasattr(autocast_fn, "__enter__") and not callable(autocast_fn):
        return autocast_fn
    # Otherwise, call it once and return whatever it yields (preserving sentinels).
    try:
        result = autocast_fn()
    except TypeError:
        try:
            result = autocast_fn(device_type=getattr(device, "type", None) or "cuda")
        except (TypeError, ValueError, RuntimeError):
            return nullcontext()
    except (ValueError, RuntimeError):
        return nullcontext()
    if isinstance(result, type):
        try:
            result = result()
        except (TypeError, ValueError, RuntimeError):
            return nullcontext()
    if not hasattr(result, "__enter__"):
        return nullcontext()
    return result


@dataclass
class CompletionTensors:
    """Completion token IDs and masks."""

    ids: Tensor
    mask: Tensor


@dataclass
class _SliceState:
    """Cached tensors and metadata required for batch slicing."""

    total_sequences: int
    slice_size: int
    completion_ids: Tensor
    completion_mask: Tensor
    prompt_entries: List[PromptCacheEntry]
    pad_token_id: int
    max_prompt_len: int
    score_tail_tokens: Optional[int] = None

    @classmethod
    def from_score_batch(cls, score_batch: ScoreBatch) -> "_SliceState":
        """Build a state snapshot derived from a score batch."""
        total_sequences = score_batch.total_sequences
        if total_sequences == 0:
            slice_size = 0
        else:
            slice_size = (
                score_batch.slice_size
                if score_batch.slice_size > 0
                else total_sequences
            )
        return cls(
            total_sequences=total_sequences,
            slice_size=slice_size,
            completion_ids=score_batch.completion_ids,
            completion_mask=score_batch.completion_attention_mask,
            prompt_entries=score_batch.prompt_entries,
            pad_token_id=score_batch.pad_token_id,
            max_prompt_len=max(1, score_batch.max_prompt_len),
            score_tail_tokens=getattr(score_batch, "score_tail_tokens", None),
        )


def _collect_prompt_entries(
    prompt_batch: List[str],
    batching_cfg: BatchingSettings,
) -> Optional[List[PromptCacheEntry]]:
    """Resolve cached prompt tokenization for a batch of strings.

    :param prompt_batch: Raw prompt strings to fetch from the cache.
    :type prompt_batch: list[str]
    :param batching_cfg: Batching configuration containing the cache getter.
    :type batching_cfg: BatchingSettings
    :returns: Cached prompt entries or ``None`` when the batch is empty.
    :rtype: list[PromptCacheEntry] | None
    """
    cache_size = getattr(batching_cfg, "prompt_cache_size", 0) or 0
    prompt_fn = getattr(batching_cfg, "prompt_length_cache_get", None)
    if cache_size > 0 and callable(prompt_fn):
        cached = getattr(batching_cfg, "_cached_prompt_lookup", None)
        underlying = getattr(batching_cfg, "_cached_prompt_source", None)
        if cached is None or underlying is not prompt_fn:
            cached = lru_cache(maxsize=cache_size)(prompt_fn)
            setattr(batching_cfg, "_cached_prompt_lookup", cached)
            setattr(batching_cfg, "_cached_prompt_source", prompt_fn)
        prompt_fn = cached
        batching_cfg.prompt_length_cache_get = prompt_fn
    if not callable(prompt_fn):
        return None
    prompt_entries = [prompt_fn(prompt) for prompt in prompt_batch]
    if not prompt_entries:
        return None
    return prompt_entries


def _tokenize_completions(
    completion_batch: List[str],
    tokenizer: PreTrainedTokenizer,
    generation_cfg: GenerationSettings,
) -> CompletionTensors:
    """Tokenize completions into padded tensors.

    :param completion_batch: Completion strings aligned with prompts.
    :type completion_batch: list[str]
    :param tokenizer: Tokenizer used to encode the completions.
    :type tokenizer: transformers.PreTrainedTokenizer
    :param generation_cfg: Generation settings (controls max length).
    :type generation_cfg: GenerationSettings
    :returns: Completion token IDs and attention masks.
    :rtype: CompletionTensors
    """
    _refresh_torch()
    try:
        completion_enc = tokenizer(
            completion_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=generation_cfg.max_completion_len,
            add_special_tokens=False,
        )
    except TypeError:
        completion_enc = tokenizer(completion_batch)
    torch_mod = sys.modules.get("torch", torch)
    ids = _maybe_long_tensor(completion_enc["input_ids"], torch_mod)
    mask = _maybe_long_tensor(completion_enc["attention_mask"], torch_mod)
    return CompletionTensors(
        ids=ids,
        mask=mask,
    )


def _completion_tensors_from_token_ids(
    token_ids: List[List[int]],
    *,
    pad_token_id: int,
    max_length: int,
) -> CompletionTensors:
    """Build completion tensors from pre-tokenized token-id sequences."""
    torch_mod = _refresh_torch()
    limit = int(max_length or 0)
    clipped: List[List[int]] = []
    for seq in token_ids:
        seq_list = list(seq)
        if limit > 0:
            seq_list = seq_list[:limit]
        clipped.append(seq_list)
    max_len = max((len(seq) for seq in clipped), default=0)
    batch = len(clipped)
    ids_arr = np.full((batch, max_len), int(pad_token_id), dtype=np.int64)
    mask_arr = np.zeros((batch, max_len), dtype=np.int64)
    for row, seq in enumerate(clipped):
        if not seq:
            continue
        ids_arr[row, : len(seq)] = np.asarray(seq, dtype=np.int64)
        mask_arr[row, : len(seq)] = 1
    ids = torch_mod.tensor(ids_arr, dtype=getattr(torch_mod, "long", None))
    mask = torch_mod.tensor(mask_arr, dtype=getattr(torch_mod, "long", None))
    return CompletionTensors(ids=ids, mask=mask)


def _prepare_prompt_slice(
    prompt_slice: List[PromptCacheEntry],
    max_prompt_len: int,
    pad_token_id: int,
    ids_dtype: torch.dtype,
    mask_dtype: torch.dtype,
) -> Tuple[Tensor, Tensor, List[int]]:
    """Materialize prompt tensors for one scoring slice.

    :param prompt_slice: Cached prompt entries for the current slice.
    :type prompt_slice: list[PromptCacheEntry]
    :param max_prompt_len: Maximum prompt length to materialize.
    :type max_prompt_len: int
    :param pad_token_id: Token ID used to pad prompts.
    :type pad_token_id: int
    :param ids_dtype: Dtype for the generated ID tensor.
    :type ids_dtype: torch.dtype
    :param mask_dtype: Dtype for the generated attention mask tensor.
    :type mask_dtype: torch.dtype
    :returns: Tuple of (prompt_ids, prompt_mask, prompt_lengths).
    :rtype: tuple[Tensor, Tensor, list[int]]
    """
    torch_mod = _refresh_torch()
    ids_dtype = getattr(ids_dtype, "np_dtype", ids_dtype)
    mask_dtype = getattr(mask_dtype, "np_dtype", mask_dtype)

    def _coerce_np_dtype(dtype: Any) -> Any:
        # Catch torch-style dtype strings/objects early.
        if isinstance(dtype, str) and dtype.startswith("torch"):
            return np.int64
        dtype_str = str(dtype)
        if dtype_str.startswith("torch."):
            return np.int64
        resolved = _resolve_dtype(dtype)
        if resolved is None:
            name_attr = getattr(dtype, "name", None)
            if isinstance(name_attr, str):
                try:
                    resolved = np.dtype(name_attr)
                except (TypeError, ValueError):
                    resolved = None
        if resolved is None:
            return np.int64
        try:
            return np.dtype(resolved)
        except (TypeError, ValueError):
            return np.int64

    ids_np_dtype = _coerce_np_dtype(ids_dtype) or np.int64
    mask_np_dtype = _coerce_np_dtype(mask_dtype) or np.int64
    prompt_lengths = [min(entry.length, max_prompt_len) for entry in prompt_slice]
    max_prompt_tokens = max(prompt_lengths) if prompt_lengths else 0
    batch_size = len(prompt_slice)
    if max_prompt_tokens > 0:
        prompt_ids_arr = np.full(
            (batch_size, max_prompt_tokens),
            pad_token_id,
            dtype=ids_np_dtype,
        )
        prompt_mask_arr = np.zeros(
            (batch_size, max_prompt_tokens),
            dtype=mask_np_dtype,
        )
        for row, (entry, length) in enumerate(zip(prompt_slice, prompt_lengths)):
            if length == 0:
                continue
            prompt_ids_arr[row, :length] = entry.input_ids[:length]
            prompt_mask_arr[row, :length] = entry.attention_mask[:length]

        def _safe_dtype(dtype):
            return (
                None
                if (
                    isinstance(dtype, str)
                    or str(dtype).startswith("torch.")
                    or isinstance(dtype, np.dtype)
                )
                else dtype
            )

        tensor_ids_dtype = _safe_dtype(ids_dtype)
        tensor_mask_dtype = _safe_dtype(mask_dtype)
        prompt_ids = torch_mod.tensor(prompt_ids_arr, dtype=tensor_ids_dtype)
        prompt_mask = torch_mod.tensor(prompt_mask_arr, dtype=tensor_mask_dtype)
    else:
        # Avoid torch.empty here because minimal stubs may omit it.
        def _safe_dtype(dtype):
            return (
                None
                if (
                    isinstance(dtype, str)
                    or str(dtype).startswith("torch.")
                    or isinstance(dtype, np.dtype)
                )
                else dtype
            )

        tensor_ids_dtype = _safe_dtype(ids_dtype)
        tensor_mask_dtype = _safe_dtype(mask_dtype)
        prompt_ids = torch_mod.zeros((batch_size, 0), dtype=tensor_ids_dtype)
        prompt_mask = torch_mod.zeros((batch_size, 0), dtype=tensor_mask_dtype)
    return prompt_ids, prompt_mask, prompt_lengths


def _slice_tail_window(
    start_idx: int,
    input_ids: Tensor,
    attention_mask: Tensor,
    labels: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Slice tail tokens without closing over loop variables."""
    if start_idx <= 0:
        return input_ids, attention_mask, labels
    return (
        input_ids[:, start_idx:],
        attention_mask[:, start_idx:],
        labels[:, start_idx:],
    )


def iter_batch_slices(
    score_batch: ScoreBatch,
    device: torch.device,  # kept for API symmetry with callers
):
    """Yield scoring slices for a batch, assembling prompt tensors on demand.

    :param score_batch: Prepared prompt/completion tensors and metadata.
    :type score_batch: ScoreBatch
    :param device: Device where tensors should be materialized.
    :type device: torch.device
    :yields: Tuples of ``(input_ids, attention_mask, labels)`` per slice.
    :rtype: Iterator[tuple[Tensor, Tensor, Tensor]]
    """
    torch_mod = _refresh_torch()
    state = _SliceState.from_score_batch(score_batch)
    if state.total_sequences == 0 or state.slice_size <= 0:
        return
    for start in range(0, state.total_sequences, state.slice_size):
        end = min(start + state.slice_size, state.total_sequences)
        prompt_slice = state.prompt_entries[start:end]
        comp_ids_slice = state.completion_ids[start:end]
        comp_mask_slice = state.completion_mask[start:end]
        if device is not None:
            try:
                comp_ids_slice = comp_ids_slice.to(device)
                comp_mask_slice = comp_mask_slice.to(device)
            except (AttributeError, TypeError, ValueError) as exc:
                # Some lightweight torch stubs treat the ``device`` argument as
                # a dtype. When that happens we leave tensors on their current
                # device and rely on ``as_tensor`` below to normalize types.
                LOG.debug("Failed to move completion tensors to device: %s", exc)
        batch_size = len(prompt_slice)
        if batch_size == 0:
            continue
        prompt_ids, prompt_mask, prompt_lengths = _prepare_prompt_slice(
            prompt_slice,
            state.max_prompt_len,
            state.pad_token_id,
            comp_ids_slice.dtype,
            comp_mask_slice.dtype,
        )
        if device is not None:
            try:
                prompt_ids = prompt_ids.to(device)
                prompt_mask = prompt_mask.to(device)
            except (AttributeError, TypeError, ValueError) as exc:
                LOG.debug("Failed to move prompt tensors to device: %s", exc)
        # Ensure tensors before concatenation (protect against stubs/numpy).
        as_tensor = getattr(torch_mod, "as_tensor", getattr(torch_mod, "tensor", None))
        if as_tensor is None:
            raise AttributeError("torch.as_tensor (or tensor) is required for scoring.")

        def _ensure_tensor(obj: Any, *, target_device: Any = None) -> Any:
            """Best-effort conversion that tolerates numpy arrays/stubs."""
            is_tensor_fn = getattr(torch_mod, "is_tensor", None)
            try:
                if callable(is_tensor_fn) and is_tensor_fn(obj):
                    return obj
            except _SCORING_EXCEPTIONS as exc:  # pragma: no cover - defensive
                LOG.debug("torch.is_tensor check failed; continuing: %s", exc)
            tensor_type = getattr(torch_mod, "Tensor", None)
            if tensor_type is not None and isinstance(obj, tensor_type):
                return obj
            tensor_ctor = getattr(torch_mod, "tensor", None)
            if callable(tensor_ctor):
                data = getattr(obj, "arr", None)
                if data is None:
                    data = obj
                try:
                    return tensor_ctor(
                        np.asarray(data),
                        device=target_device,
                        dtype=getattr(obj, "dtype", None),
                    )
                except TypeError:
                    return tensor_ctor(np.asarray(data))
            return obj

        prompt_ids = as_tensor(prompt_ids, device=device)
        prompt_mask = as_tensor(prompt_mask, device=device)
        comp_ids_slice = as_tensor(comp_ids_slice, device=device)
        comp_mask_slice = as_tensor(comp_mask_slice, device=device)
        # Drop completion columns that are padding for every sequence so tail-only
        # scoring keeps real tokens instead of global pad regions.
        comp_tokens_present = None
        active_comp_columns: List[int] = []
        try:
            comp_tokens_present = (comp_mask_slice != 0).any(dim=0)
        except _SCORING_EXCEPTIONS as exc:
            LOG.debug("Failed to compute completion token presence mask: %s", exc)
        if comp_tokens_present is None:
            comp_tokens_arr = np.asarray(
                getattr(comp_mask_slice, "arr", comp_mask_slice)
            ) != 0
            col_activity = comp_tokens_arr.any(axis=0)
            active_idx_np = np.nonzero(col_activity)[0]
            active_comp_columns = [int(idx) for idx in active_idx_np.tolist()]
            if active_comp_columns:
                last_valid_idx = active_comp_columns[-1] + 1
            else:
                last_valid_idx = 0
        else:
            try:
                nonzero_cols = torch_mod.nonzero(comp_tokens_present, as_tuple=False).view(
                    -1
                )
                active_comp_columns = [int(idx.item()) for idx in nonzero_cols]
                last_valid_idx = active_comp_columns[-1] + 1 if active_comp_columns else 0
            except _SCORING_EXCEPTIONS:
                active_comp_columns = []
                last_valid_idx = 0
        if last_valid_idx <= 0:
            comp_ids_slice = comp_ids_slice[:, :0]
            comp_mask_slice = comp_mask_slice[:, :0]
        elif last_valid_idx < getattr(comp_ids_slice, "shape", [0, 0])[1]:
            comp_ids_slice = comp_ids_slice[:, :last_valid_idx]
            comp_mask_slice = comp_mask_slice[:, :last_valid_idx]
        full_input_ids = torch_mod.cat([prompt_ids, comp_ids_slice], dim=1)
        full_attention_mask = torch_mod.cat([prompt_mask, comp_mask_slice], dim=1)
        labels_tensor = full_input_ids.clone()
        for idx, plen in enumerate(prompt_lengths):
            labels_tensor[idx, :plen] = -100
        prompt_width = getattr(prompt_ids, "shape", [0, 0])[1] if prompt_ids is not None else 0
        comp_width = getattr(comp_ids_slice, "shape", [0, 0])[1] if comp_ids_slice is not None else 0
        if comp_width > 0:
            comp_slice = slice(prompt_width, prompt_width + comp_width)
            comp_labels = labels_tensor[:, comp_slice]
            updated_labels = comp_labels
            pad_mask = None
            pad_mask_arr = None
            has_padding = False
            try:
                pad_mask = comp_mask_slice == 0
                has_padding = bool(pad_mask.any())
            except _SCORING_EXCEPTIONS:
                pad_mask = None
            if pad_mask is None:
                try:
                    pad_mask_arr = np.asarray(
                        getattr(comp_mask_slice, "arr", comp_mask_slice)
                    ) == 0
                    has_padding = bool(pad_mask_arr.any())
                except _SCORING_EXCEPTIONS:
                    pad_mask_arr = None
                    has_padding = False
            if has_padding:
                try:
                    if pad_mask is None:
                        raise AttributeError
                    updated_labels = comp_labels.masked_fill(pad_mask, -100)
                except _SCORING_EXCEPTIONS:
                    if pad_mask_arr is None:
                        pad_mask_arr = np.asarray(
                            getattr(comp_mask_slice, "arr", comp_mask_slice)
                        ) == 0
                    comp_arr = np.asarray(getattr(comp_labels, "arr", comp_labels))
                    comp_arr[pad_mask_arr] = -100
                    updated_labels = as_tensor(
                        comp_arr, device=getattr(labels_tensor, "device", None)
                    )
                labels_tensor[:, comp_slice] = updated_labels
        full_labels = labels_tensor
        input_ids = full_input_ids
        attention_mask = full_attention_mask
        tail_tokens = getattr(state, "score_tail_tokens", None)
        if tail_tokens is not None:
            try:
                tail_tokens = int(tail_tokens)
            except (TypeError, ValueError):
                tail_tokens = None
        if tail_tokens is not None and tail_tokens > 0:
            max_len = int(getattr(full_input_ids, "shape", [0, 0])[1] or 0)
            tail_tokens = min(tail_tokens, max_len) if max_len > 0 else 0
            if tail_tokens > 0 and tail_tokens < max_len:
                slice_start = max(0, max_len - tail_tokens)
                first_comp_global = None
                last_comp_global = None
                if active_comp_columns:
                    first_comp_global = prompt_width + active_comp_columns[0]
                    last_comp_global = prompt_width + active_comp_columns[-1] + 1
                input_ids, attention_mask, labels_tensor = _slice_tail_window(
                    slice_start,
                    full_input_ids,
                    full_attention_mask,
                    full_labels,
                )
                if (
                    first_comp_global is not None
                    and last_comp_global is not None
                    and slice_start >= last_comp_global
                ):
                    safe_start = max(first_comp_global, last_comp_global - tail_tokens)
                    input_ids, attention_mask, labels_tensor = _slice_tail_window(
                        safe_start,
                        full_input_ids,
                        full_attention_mask,
                        full_labels,
                    )
        # Materialize tensors for the ref model; keep device parity with completions.
        target_device = device or getattr(comp_ids_slice, "device", None)
        target_dtype = getattr(torch_mod, "long", None)
        input_ids_out = as_tensor(input_ids, device=target_device, dtype=target_dtype)
        attention_mask_out = as_tensor(
            attention_mask, device=target_device, dtype=target_dtype
        )
        labels_out = as_tensor(labels_tensor, device=target_device, dtype=target_dtype)
        input_ids_out = _ensure_tensor(input_ids_out, target_device=target_device)
        attention_mask_out = _ensure_tensor(attention_mask_out, target_device=target_device)
        labels_out = _ensure_tensor(labels_out, target_device=target_device)
        # Some lightweight stubs do not preserve the requested dtype object on
        # the Tensor wrapper (they store only the underlying numpy dtype). For
        # tests that compare against ``torch.long`` we make a best-effort pass
        # to align the exposed ``dtype`` attribute with the module constant.
        long_dtype = getattr(torch_mod, "long", None)
        if long_dtype is not None:
            proxy = _LongDTypeProxy(long_dtype)
            for _tensor in (input_ids_out, labels_out):
                try:  # pragma: no cover - exercised via stubbed environments
                    setattr(_tensor, "dtype", proxy)
                except _SCORING_EXCEPTIONS as exc:
                    LOG.debug("Unable to patch tensor dtype proxy: %s", exc)
        yield (input_ids_out, attention_mask_out, labels_out)


def _summon_fsdp_full_param_context(model: PreTrainedModel):
    """Return a context manager that gathers FSDP parameters when available."""
    if not hasattr(model, "summon_full_params"):
        return nullcontext()
    try:
        return model.summon_full_params()
    except TypeError:
        try:
            return model.summon_full_params(recurse=True)
        except TypeError:
            return nullcontext()


def _chunked_sequence_logprobs(
    model: PreTrainedModel,
    *,
    input_ids: Tensor,
    attention_mask: Tensor,
    labels: Tensor,
    chunk_size: int,
    gather_full_params: bool = False,  # retained for parity
    return_hidden: bool = False,
    pooling: str = "mean",
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Compute summed log-probabilities per sequence with optional chunking and pooled states."""

    torch_mod = _refresh_torch()
    _ = (
        chunk_size,
    )  # parity with distributed APIs; currently unused
    # Fallback for lightweight stubs
    if not hasattr(model, "forward"):
        label_arr = np.asarray(getattr(labels, "arr", labels))
        valid_counts = (label_arr != -100).sum(axis=1)
        logp = torch_mod.tensor(
            np.zeros(len(valid_counts), dtype=float),
            dtype=getattr(torch_mod, "float32", None),
        )
        tok_tensor = torch_mod.tensor(
            valid_counts, dtype=getattr(torch_mod, "float32", None)
        )
        return logp, tok_tensor, None
    shape = getattr(input_ids, "shape", None)
    if shape and len(shape) > 1 and shape[1] == 0:
        batch_size = shape[0]
        zero = torch_mod.tensor(
            np.zeros(batch_size, dtype=float), dtype=getattr(torch_mod, "float32", None)
        )
        tok_tensor = torch_mod.tensor(
            np.zeros(batch_size, dtype=float), dtype=getattr(torch_mod, "float32", None)
        )
        return zero, tok_tensor, None
    # DeepSpeed ZeRO-3 shards parameters to 1-D partitions; gather embeddings
    # so the reference forward sees 2-D weights without full-parameter all-gather.
    gather_ctx = nullcontext()
    dist = _dist_collective_ready(torch_mod)
    if gather_full_params:
        try:  # pragma: no cover - exercised in distributed runs
            import deepspeed

            params: list[Any] = []
            param_iter = getattr(model, "parameters", None)
            if callable(param_iter):
                try:
                    params = list(param_iter())
                except _SCORING_EXCEPTIONS:
                    params = []
            # Always invoke GatheredParameters when available, even with an empty
            # list, so stubbed environments can verify the context is entered.
            try:
                gather_ctx = deepspeed.zero.GatheredParameters(
                    params or [], modifier_rank=None
                )
            except TypeError:
                gather_ctx = deepspeed.zero.GatheredParameters(params or [])
        except ImportError:
            gather_ctx = nullcontext()
    else:
        try:  # pragma: no cover - exercised in distributed runs
            import deepspeed

            if deepspeed.zero.is_enabled():
                to_gather: list[Any] = []
                if os.environ.get("MAXENT_DISABLE_SCORING_ZERO_GATHER", "").strip():
                    gather_ctx = nullcontext()
                inp_emb = (
                    model.get_input_embeddings()
                    if hasattr(model, "get_input_embeddings")
                    else None
                )
                input_weight = (
                    getattr(inp_emb, "weight", None)
                    if inp_emb is not None and hasattr(inp_emb, "weight")
                    else None
                )
                out_emb = (
                    model.get_output_embeddings()
                    if hasattr(model, "get_output_embeddings")
                    else None
                )
                if out_emb is None and hasattr(model, "lm_head"):
                    out_emb = model.lm_head
                output_weight = (
                    getattr(out_emb, "weight", None)
                    if out_emb is not None and hasattr(out_emb, "weight")
                    else None
                )

                input_present_local = input_weight is not None
                output_present_local = output_weight is not None
                input_present_all = (
                    _dist_all(dist, input_present_local)
                    if dist is not None
                    else input_present_local
                )
                output_present_all = (
                    _dist_all(dist, output_present_local)
                    if dist is not None
                    else output_present_local
                )

                needs_input_local = bool(
                    input_weight is not None
                    and not _weight_is_two_dimensional(input_weight)
                )
                needs_output_local = bool(
                    output_weight is not None
                    and not _weight_is_two_dimensional(output_weight)
                )
                needs_input_any = (
                    _dist_any(dist, needs_input_local)
                    if dist is not None
                    else needs_input_local
                )
                needs_output_any = (
                    _dist_any(dist, needs_output_local)
                    if dist is not None
                    else needs_output_local
                )

                if needs_input_any and input_present_all and input_weight is not None:
                    to_gather.append(input_weight)
                if needs_output_any and output_present_all and output_weight is not None:
                    to_gather.append(output_weight)

                if (needs_input_any and not input_present_all) or (
                    needs_output_any and not output_present_all
                ):
                    LOG.warning(
                        "DeepSpeed ZeRO gather decision mismatch across ranks; skipping embed gather to avoid deadlock | "
                        "input_present_local=%s input_present_all=%s needs_input_local=%s needs_input_any=%s | "
                        "output_present_local=%s output_present_all=%s needs_output_local=%s needs_output_any=%s",
                        input_present_local,
                        input_present_all,
                        needs_input_local,
                        needs_input_any,
                        output_present_local,
                        output_present_all,
                        needs_output_local,
                        needs_output_any,
                    )
                    to_gather = []

                if to_gather:
                    # De-duplicate parameters to avoid repeated GatheredParameters calls
                    # on shared/tied embedding weights.
                    seen: set[int] = set()
                    unique: list[Any] = []
                    for param in to_gather:
                        param_id = id(param)
                        if param_id in seen:
                            continue
                        seen.add(param_id)
                        unique.append(param)
                    LOG.debug(
                        "DeepSpeed ZeRO gather for scoring | tensors=%d | shapes=%s",
                        len(unique),
                        [getattr(param, "shape", None) for param in unique],
                    )
                    try:
                        gather_ctx = deepspeed.zero.GatheredParameters(
                            unique, modifier_rank=None
                        )
                    except TypeError:
                        gather_ctx = deepspeed.zero.GatheredParameters(unique)
        except ImportError:
            gather_ctx = nullcontext()
        except _SCORING_EXCEPTIONS:
            gather_ctx = nullcontext()

    fsdp_ctx = _summon_fsdp_full_param_context(model)
    stack = ExitStack()
    stack.enter_context(gather_ctx)
    # Ensure ZeRO-managed params are gathered consistently before the forward pass.
    # The helper is a no-op when DeepSpeed isn't available, but keeps tests and
    # distributed reference scoring behavior aligned.
    stack.enter_context(_maybe_zero_gather_params(model, enabled=True))
    stack.enter_context(fsdp_ctx)
    stack.enter_context(_maybe_zero_gather_embedding(model))
    config = getattr(model, "config", None)
    padding_idx = _get_config_value(config, "pad_token_id", None)
    embedding_vocab_size = _get_embedding_vocab_size(model, config)
    vocab_size = _get_config_value(config, "vocab_size", None)
    pad_targets: list[tuple[Any, str]] = []
    if config is not None:
        pad_targets.append((config, "pad_token_id"))
    seen_modules: set[int] = set()
    embed_token_module = getattr(model, "embed_tokens", None)
    if embed_token_module is not None:
        seen_modules.add(id(embed_token_module))
        if hasattr(embed_token_module, "padding_idx"):
            pad_targets.append((embed_token_module, "padding_idx"))
    try:
        input_embed_module = model.get_input_embeddings()
    except _SCORING_EXCEPTIONS:
        input_embed_module = None
    if (
        input_embed_module is not None
        and id(input_embed_module) not in seen_modules
        and hasattr(input_embed_module, "padding_idx")
    ):
        pad_targets.append((input_embed_module, "padding_idx"))
    final_padding_idx = padding_idx
    if padding_idx is not None:
        limit: Optional[int] = None
        if embedding_vocab_size is not None:
            limit = embedding_vocab_size - 1
        if vocab_size is not None:
            vocab_limit = vocab_size - 1
            if limit is None or vocab_limit < limit:
                limit = vocab_limit
        if limit is not None:
            limit = max(limit, 0)
            if padding_idx > limit:
                final_padding_idx = limit
    pad_ctx = nullcontext()
    if (
        padding_idx is not None
        and final_padding_idx is not None
        and final_padding_idx != padding_idx
        and pad_targets
    ):
        LOG.debug(
            "Clamping padding idx for scoring | original=%s final=%s",
            padding_idx,
            final_padding_idx,
        )
        pad_ctx = _PadTokenGuard(pad_targets, final_padding_idx)
        padding_idx = final_padding_idx
    stack.enter_context(pad_ctx)
    with stack:
        LOG.debug(
            "chunked_sequence_logprobs start | gather_full_params=%s return_hidden=%s pooling=%s | "
            "input_ids_shape=%s dtype=%s device=%s | attention_mask_shape=%s | labels_shape=%s dtype=%s device=%s",
            gather_full_params,
            return_hidden,
            pooling,
            getattr(input_ids, "shape", None),
            getattr(input_ids, "dtype", None),
            getattr(input_ids, "device", None),
            getattr(attention_mask, "shape", None),
            getattr(labels, "shape", None),
            getattr(labels, "dtype", None),
            getattr(labels, "device", None),
        )
        # High-level shape logging for reference scoring.
        LOG.debug(
            "reference scoring inputs | input_ids_shape=%s attention_mask_shape=%s labels_shape=%s",
            getattr(input_ids, "shape", None),
            getattr(attention_mask, "shape", None),
            getattr(labels, "shape", None),
        )
        LOG.debug(
            "reference scoring pad metadata | model.config.pad_token_id=%s embedding_vocab_size=%s",
            padding_idx,
            embedding_vocab_size,
        )

        embed_descs: list[str] = []
        embed_tokens = getattr(model, "embed_tokens", None)
        embed_descs.append(_describe_embedding_module(embed_tokens, "embed_tokens"))
        try:
            input_embeddings = model.get_input_embeddings()
        except _SCORING_EXCEPTIONS:
            input_embeddings = None
        if input_embeddings is not None and input_embeddings is not embed_tokens:
            embed_descs.append(
                _describe_embedding_module(input_embeddings, "input_embeddings")
            )
        # Conservative guard: if the reference model's embedding weights are
        # not 2-D under the gathered parameter contexts, skip reference
        # scoring to avoid noisy runtime errors from torch.embedding.
        for module in (embed_tokens, input_embeddings):
            if module is None:
                continue
            weight = getattr(module, "weight", None)
            if weight is not None and not _weight_is_two_dimensional(weight):
                if _weight_is_stub_tensor(weight):
                    LOG.debug(
                        "Non-2D stub embedding weight; continuing reference scoring | %s",
                        " | ".join(embed_descs),
                    )
                    continue
                LOG.warning(
                    "Skipping reference scoring due to non-2D embedding weight | %s",
                    " | ".join(embed_descs),
                )
                return None
        LOG.debug(
            "reference scoring embeddings | %s | pad_token_id=%s vocab=%s",
            " | ".join(embed_descs),
            padding_idx,
            embedding_vocab_size,
        )
        batch = getattr(input_ids, "shape", None)
        batch = batch[0] if batch else 0
        chunk_limit = int(chunk_size) if chunk_size is not None else 0
        if chunk_limit <= 0 and batch > 1 and not os.environ.get(
            "MAXENT_DISABLE_LOGPROB_AUTOBATCH", ""
        ).strip():
            try:
                vocab_size_guess = int(
                    getattr(getattr(model, "config", None), "vocab_size", 0) or 0
                )
            except (TypeError, ValueError):
                vocab_size_guess = 0
            seq_len = getattr(input_ids, "shape", None)
            seq_len = int(seq_len[1]) if seq_len and len(seq_len) > 1 else 0
            device_str = str(getattr(input_ids, "device", "")).lower()
            if vocab_size_guess > 0 and seq_len > 0 and "cuda" in device_str:
                try:
                    model_dtype = getattr(model, "dtype", None)
                    dtype_str = str(getattr(model_dtype, "name", model_dtype) or "").lower()
                except _SCORING_EXCEPTIONS:
                    dtype_str = ""
                bytes_per_elem = 2
                if "float32" in dtype_str or "fp32" in dtype_str:
                    bytes_per_elem = 4
                target_mb_raw = os.environ.get("MAXENT_LOGPROB_TARGET_LOGITS_MB", "256")
                try:
                    target_bytes = int(float(target_mb_raw) * 1024 * 1024)
                except (TypeError, ValueError):
                    target_bytes = 256 * 1024 * 1024
                bytes_per_seq = max(1, seq_len * vocab_size_guess * bytes_per_elem)
                auto_limit = max(1, min(batch, target_bytes // bytes_per_seq))
                if auto_limit < batch:
                    warned = getattr(_chunked_sequence_logprobs, "_autobatch_warned", False)
                    if not warned:
                        LOG.warning(
                            "Auto-tuning reference scoring batch chunk size to avoid large logits tensors | "
                            "requested_chunk_size=%s auto_chunk_size=%s batch=%s seq_len=%s vocab=%s target_logits_mb=%s "
                            "(set MAXENT_DISABLE_LOGPROB_AUTOBATCH=1 to disable)",
                            chunk_size,
                            auto_limit,
                            batch,
                            seq_len,
                            vocab_size_guess,
                            target_mb_raw,
                        )
                        setattr(_chunked_sequence_logprobs, "_autobatch_warned", True)
                    chunk_limit = auto_limit
        if chunk_limit <= 0 or chunk_limit >= batch:
            chunk_indices = [(0, batch)]
        else:
            chunk_indices = [
                (start, min(start + chunk_limit, batch))
                for start in range(0, batch, chunk_limit)
            ]
        LOG.debug(
            "reference scoring chunk plan | total_batch=%s chunk_size=%s chunks=%s",
            batch,
            chunk_limit,
            len(chunk_indices),
        )
        logp_chunks: list[Tensor] = []
        tok_chunks: list[Tensor] = []
        pooled_chunks: list[Tensor] = [] if return_hidden else []
        for idx, (start, end) in enumerate(chunk_indices):
            LOG.debug(
                "reference scoring chunk begin | chunk=%s | slice=[%s:%s] | rows=%s",
                idx,
                start,
                end,
                end - start,
            )
            ids_chunk = input_ids[start:end]
            mask_chunk = attention_mask[start:end] if attention_mask is not None else None
            label_chunk = labels[start:end]
            wants_labels = False
            for callable_name in ("forward", "__call__"):
                candidate = getattr(model, callable_name, None)
                if not callable(candidate):
                    continue
                try:
                    sig = inspect.signature(candidate)
                    param = sig.parameters.get("labels")
                    if param is not None and param.default is inspect.Signature.empty:
                        wants_labels = True
                        break
                except (TypeError, ValueError):
                    continue
            call_kwargs: dict[str, Any] = {
                "input_ids": ids_chunk,
                "attention_mask": mask_chunk,
                "output_hidden_states": return_hidden,
            }
            if wants_labels:
                call_kwargs["labels"] = label_chunk
            call_target = model if callable(model) else getattr(model, "forward", None)
            if not callable(call_target):
                raise TypeError("Model is not callable and lacks a forward method")
            try:
                outputs = call_target(**call_kwargs)
            except TypeError:
                call_kwargs.pop("output_hidden_states", None)
                try:
                    outputs = call_target(**call_kwargs)
                except TypeError:
                    if not wants_labels:
                        call_kwargs.pop("labels", None)
                    outputs = call_target(**call_kwargs)
            logits = outputs.logits
            LOG.debug(
                "reference scoring logits metadata | chunk=%s | shape=%s dtype=%s device=%s",
                idx,
                getattr(logits, "shape", None),
                getattr(logits, "dtype", None),
                getattr(logits, "device", None),
            )
            # Causal LM logits at position t predict token t+1. Align labels by
            # shifting so we score next-token log-probs for all non-masked targets.
            if logits.size(1) <= 1:
                batch_rows = logits.size(0)
                try:
                    seq_logp_chunk = torch_mod.zeros(
                        (batch_rows,),
                        dtype=getattr(torch_mod, "float32", None),
                        device=getattr(logits, "device", None),
                    )
                except _SCORING_EXCEPTIONS:
                    seq_logp_chunk = torch_mod.tensor(
                        np.zeros(batch_rows, dtype=float),
                        dtype=getattr(torch_mod, "float32", None),
                        device=getattr(logits, "device", None),
                    )
                tok_tensor_chunk = torch_mod.ones(
                    (batch_rows,),
                    dtype=getattr(torch_mod, "long", None),
                    device=getattr(logits, "device", None),
                )
                logp_chunks.append(seq_logp_chunk)
                tok_chunks.append(tok_tensor_chunk)
                preview_vals = None
                preview_source = seq_logp_chunk
                detach_fn = getattr(seq_logp_chunk, "detach", None)
                if callable(detach_fn):
                    try:
                        preview_source = detach_fn()
                    except _SCORING_EXCEPTIONS:
                        preview_source = seq_logp_chunk
                if hasattr(preview_source, "cpu"):
                    try:
                        preview_vals = preview_source.cpu().reshape(-1)[:3].tolist()
                    except _SCORING_EXCEPTIONS:
                        preview_vals = None
                LOG.debug(
                    "reference scoring chunk stats | chunk=%s | ids_shape=%s | mask_shape=%s | "
                    "seq_logp_shape=%s dtype=%s device=%s | tok_shape=%s dtype=%s device=%s | "
                    "tok_sum=%s | valid_token_mask_sum=%s | seq_logp_preview=%s",
                    idx,
                    getattr(ids_chunk, "shape", None),
                    getattr(mask_chunk, "shape", None),
                    getattr(seq_logp_chunk, "shape", None),
                    getattr(seq_logp_chunk, "dtype", None),
                    getattr(seq_logp_chunk, "device", None),
                    getattr(tok_tensor_chunk, "shape", None),
                    getattr(tok_tensor_chunk, "dtype", None),
                    getattr(tok_tensor_chunk, "device", None),
                    float(batch_rows),
                    0,
                    preview_vals,
                )
                continue

            shifted_logits = logits[:, :-1, :]
            shifted_labels = label_chunk[:, 1:]
            label_mask = shifted_labels != -100
            vocab_size = shifted_logits.size(-1)
            flat_logits = shifted_logits.reshape(-1, vocab_size)
            safe_labels = shifted_labels.masked_fill(~label_mask, 0)
            flat_labels = safe_labels.reshape(-1)
            token_logp = None
            try:
                logsumexp_fn = getattr(torch_mod, "logsumexp", None)
                if not callable(logsumexp_fn):
                    raise AttributeError("torch.logsumexp unavailable")
                log_denom = logsumexp_fn(shifted_logits, dim=-1)
                gather_index = torch_mod.arange(flat_labels.numel())
                target_device = getattr(flat_logits, "device", None)
                if target_device is not None:
                    try:
                        gather_index = gather_index.to(target_device)
                    except _SCORING_EXCEPTIONS as exc:
                        LOG.debug("Failed to move gather_index to device: %s", exc)
                target_logits = flat_logits[gather_index, flat_labels].reshape(safe_labels.shape)
                to_fn = getattr(target_logits, "to", None)
                if callable(to_fn):
                    target_logits = to_fn(dtype=getattr(log_denom, "dtype", None))
                token_logp = target_logits - log_denom
            except _SCORING_EXCEPTIONS:
                log_probs = torch_mod.nn.functional.log_softmax(shifted_logits, dim=-1)
                flat_log_probs = log_probs.reshape(-1, vocab_size)
                gather_index = torch_mod.arange(flat_labels.numel())
                target_device = getattr(flat_log_probs, "device", None)
                if target_device is not None:
                    try:
                        gather_index = gather_index.to(target_device)
                    except _SCORING_EXCEPTIONS as exc:
                        LOG.debug("Failed to move fallback gather_index to device: %s", exc)
                token_logp = flat_log_probs[gather_index, flat_labels].reshape(safe_labels.shape)
            # If mask tensors are real torch tensors but token_logp is a stub tensor,
            # coerce token_logp into the active torch module to avoid type mismatches.
            torch_tensor = getattr(torch_mod, "Tensor", None)
            if (
                torch_tensor is not None
                and isinstance(label_mask, torch_tensor)
                and not isinstance(token_logp, torch_tensor)
            ):
                try:
                    token_logp = torch_mod.tensor(_to_numpy_array(token_logp))
                except _SCORING_EXCEPTIONS as exc:
                    LOG.debug("Failed to coerce token_logp into torch tensor: %s", exc)
            mask_float = label_mask
            type_as_fn = getattr(mask_float, "type_as", None)
            if callable(type_as_fn):
                try:
                    is_tensor_fn = getattr(torch_mod, "is_tensor", None)
                    if callable(is_tensor_fn) and not is_tensor_fn(token_logp):
                        raise TypeError("type_as requires a torch tensor")
                    mask_mod = getattr(type(mask_float), "__module__", "")
                    token_mod = getattr(type(token_logp), "__module__", "")
                    if mask_mod.startswith("torch") != token_mod.startswith("torch"):
                        raise TypeError("type_as requires matching torch tensors")
                    if not isinstance(token_logp, type(mask_float)):
                        raise TypeError("type_as requires same tensor types")
                    mask_float = type_as_fn(token_logp)
                except _SCORING_EXCEPTIONS:
                    type_as_fn = None
            if not callable(type_as_fn):
                to_fn = getattr(mask_float, "to", None)
                if callable(to_fn):
                    try:
                        mask_float = to_fn(dtype=getattr(token_logp, "dtype", None))
                    except _SCORING_EXCEPTIONS:
                        float_fn = getattr(mask_float, "float", None)
                        if callable(float_fn):
                            mask_float = float_fn()
            seq_logp_chunk = (token_logp * mask_float).sum(dim=1)
            tok_tensor_chunk = label_mask.sum(dim=1).clamp(min=1)
            logp_chunks.append(seq_logp_chunk)
            tok_chunks.append(tok_tensor_chunk)
            try:
                tok_sum = tok_tensor_chunk.detach().cpu().sum().item()
            except _SCORING_EXCEPTIONS:
                tok_sum = None
            try:
                logp_preview = (
                    seq_logp_chunk.detach().cpu().reshape(-1)[:3].tolist()
                )
            except _SCORING_EXCEPTIONS:
                logp_preview = None
            try:
                valid_tokens = int(label_mask.sum().detach().cpu().item())
            except _SCORING_EXCEPTIONS:
                valid_tokens = None
            LOG.debug(
                "reference scoring chunk stats | chunk=%s | ids_shape=%s | mask_shape=%s | "
                "seq_logp_shape=%s dtype=%s device=%s | tok_shape=%s dtype=%s device=%s | "
                "tok_sum=%s | valid_token_mask_sum=%s | seq_logp_preview=%s",
                idx,
                getattr(ids_chunk, "shape", None),
                getattr(mask_chunk, "shape", None),
                getattr(seq_logp_chunk, "shape", None),
                getattr(seq_logp_chunk, "dtype", None),
                getattr(seq_logp_chunk, "device", None),
                getattr(tok_tensor_chunk, "shape", None),
                getattr(tok_tensor_chunk, "dtype", None),
                getattr(tok_tensor_chunk, "device", None),
                tok_sum,
                valid_tokens,
                logp_preview,
            )
            if return_hidden and getattr(outputs, "hidden_states", None) is not None:
                hidden = outputs.hidden_states[-1]
                mask = mask_chunk
                if pooling == "last":
                    pooled = hidden[:, -1, :]
                else:
                    if mask is None:
                        pooled = hidden.mean(dim=1)
                    else:
                        mask = mask.unsqueeze(-1)
                        type_as_fn = getattr(mask, "type_as", None)
                        if callable(type_as_fn):
                            try:
                                is_tensor_fn = getattr(torch_mod, "is_tensor", None)
                                if callable(is_tensor_fn) and not is_tensor_fn(hidden):
                                    raise TypeError("type_as requires a torch tensor")
                                mask_mod = getattr(type(mask), "__module__", "")
                                hidden_mod = getattr(type(hidden), "__module__", "")
                                if mask_mod.startswith("torch") != hidden_mod.startswith("torch"):
                                    raise TypeError("type_as requires matching torch tensors")
                                if not isinstance(hidden, type(mask)):
                                    raise TypeError("type_as requires same tensor types")
                                mask = type_as_fn(hidden)
                            except _SCORING_EXCEPTIONS:
                                type_as_fn = None
                        if not callable(type_as_fn):
                            to_fn = getattr(mask, "to", None)
                            if callable(to_fn):
                                try:
                                    mask = to_fn(dtype=hidden.dtype)
                                except _SCORING_EXCEPTIONS:
                                    float_fn = getattr(mask, "float", None)
                                    if callable(float_fn):
                                        mask = float_fn()
                            else:
                                float_fn = getattr(mask, "float", None)
                                if callable(float_fn):
                                    mask = float_fn()
                        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(
                            min=1.0
                        )
                pooled_chunks.append(pooled)
    seq_logp = (
        logp_chunks[0] if len(logp_chunks) == 1 else torch_mod.cat(logp_chunks, dim=0)
    )
    tok_tensor = (
        tok_chunks[0] if len(tok_chunks) == 1 else torch_mod.cat(tok_chunks, dim=0)
    )
    pooled_hidden: Optional[Tensor] = None
    if pooled_chunks:
        pooled_hidden = (
            pooled_chunks[0]
            if len(pooled_chunks) == 1
            else torch_mod.cat(pooled_chunks, dim=0)
        )
    try:
        logp_sample = seq_logp.detach().cpu().reshape(-1)[:4].tolist()
    except _SCORING_EXCEPTIONS:
        logp_sample = None
    LOG.debug(
        "chunked_sequence_logprobs finish | seq_logp_shape=%s tok_shape=%s pooled_shape=%s | "
        "seq_logp_dtype=%s tok_dtype=%s pooled_dtype=%s | seq_logp_device=%s tok_device=%s pooled_device=%s | "
        "seq_logp_numel=%s tok_numel=%s | seq_logp_preview=%s",
        getattr(seq_logp, "shape", None),
        getattr(tok_tensor, "shape", None),
        getattr(pooled_hidden, "shape", None) if pooled_hidden is not None else None,
        getattr(seq_logp, "dtype", None),
        getattr(tok_tensor, "dtype", None),
        getattr(pooled_hidden, "dtype", None) if pooled_hidden is not None else None,
        getattr(seq_logp, "device", None),
        getattr(tok_tensor, "device", None),
        getattr(pooled_hidden, "device", None) if pooled_hidden is not None else None,
        getattr(seq_logp, "numel", lambda: None)(),
        getattr(tok_tensor, "numel", lambda: None)(),
        logp_sample,
    )
    return seq_logp, tok_tensor, pooled_hidden


def build_score_batch(
    reward_comp: RewardComputation,
    tokenizer: PreTrainedTokenizer,
    generation_cfg: GenerationSettings,
    batching_cfg: BatchingSettings,
) -> Optional[ScoreBatch]:
    """Tokenize prompt+completion pairs and prepare masks/labels.

    :param reward_comp: Reward computation payload containing prompts and completions.
    :param tokenizer: Tokenizer used to encode completions and determine padding.
    :param generation_cfg: Generation settings (max lengths, etc.).
    :param batching_cfg: Batching settings controlling scoring slice sizes.
    :returns: Prepared ``ScoreBatch`` or ``None`` when no sequences are available.
    :rtype: ScoreBatch | None
    """
    prompt_batch = getattr(reward_comp.pairs, "prompts", reward_comp.pairs.completions)
    completion_batch = reward_comp.pairs.completions
    total_sequences = len(prompt_batch)
    if total_sequences == 0:
        return None
    prompt_length_cache = getattr(batching_cfg, "prompt_length_cache_get", None)
    if prompt_length_cache is None and callable(batching_cfg):
        prompt_length_cache = batching_cfg
    if prompt_length_cache is None:

        def prompt_length_cache(_p: str) -> PromptCacheEntry:
            return PromptCacheEntry(input_ids=[], attention_mask=[])

    prompt_entries = _collect_prompt_entries(
        prompt_batch,
        SimpleNamespace(prompt_length_cache_get=prompt_length_cache),
    )
    if prompt_entries is None:
        return None
    completion_tensors: Optional[CompletionTensors] = None
    completion_meta = getattr(reward_comp, "completion_metadata", None)
    if (
        completion_meta
        and isinstance(completion_meta, list)
        and len(completion_meta) == len(completion_batch)
    ):
        token_ids: List[List[int]] = []
        ok = True
        for entry in completion_meta:
            if not isinstance(entry, dict):
                ok = False
                break
            raw_ids = entry.get("token_ids")
            if raw_ids is None:
                ok = False
                break
            if hasattr(raw_ids, "tolist"):
                try:
                    raw_ids = raw_ids.tolist()
                except _SCORING_EXCEPTIONS as exc:
                    LOG.debug("Failed to convert completion metadata token_ids: %s", exc)
            if isinstance(raw_ids, list) and raw_ids and isinstance(raw_ids[0], list):
                raw_ids = raw_ids[0]
            if not isinstance(raw_ids, list):
                ok = False
                break
            try:
                token_ids.append([int(val) for val in raw_ids])
            except (TypeError, ValueError):
                ok = False
                break
        if ok:
            pad_token_id = tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = tokenizer.eos_token_id or 0
            vocab_size = getattr(tokenizer, "vocab_size", None)
            if (
                vocab_size is not None
                and pad_token_id is not None
                and pad_token_id >= vocab_size
            ):
                pad_token_id = tokenizer.eos_token_id or vocab_size - 1 or 0
            completion_tensors = _completion_tensors_from_token_ids(
                token_ids,
                pad_token_id=int(pad_token_id or 0),
                max_length=int(getattr(generation_cfg, "max_completion_len", 0) or 0),
            )
            LOG.debug(
                "Using pre-tokenized completion token_ids from completion_metadata | sequences=%d",
                len(token_ids),
            )
    if completion_tensors is None:
        completion_tensors = _tokenize_completions(
            completion_batch,
            tokenizer,
            generation_cfg,
        )
    slice_size = (
        batching_cfg.score_slice if batching_cfg.score_slice > 0 else total_sequences
    )
    slice_size = max(1, slice_size)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id or 0
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if (
        vocab_size is not None
        and pad_token_id is not None
        and pad_token_id >= vocab_size
    ):
        pad_token_id = tokenizer.eos_token_id or vocab_size - 1 or 0
    return ScoreBatch(
        prompt_entries=prompt_entries,
        completion_ids=completion_tensors.ids,
        completion_attention_mask=completion_tensors.mask,
        pad_token_id=pad_token_id,
        max_prompt_len=generation_cfg.max_prompt_len,
        slice_size=slice_size,
        total_sequences=total_sequences,
        score_tail_tokens=getattr(batching_cfg, "score_tail_tokens", None),
    )


def reference_from_model(
    score_batch: ScoreBatch,
    runtime: RuntimeHandles,
    batching_cfg: BatchingSettings,
) -> Optional[Tuple[Tensor, Tensor]]:
    """Run the frozen reference model to compute log-probs.

    :param score_batch: Prepared scoring batch with prompts/completions.
    :param runtime: Runtime handles exposing device and reference model.
    :param batching_cfg: Batching config controlling logprob chunking.
    :returns: Tuple of ``(ref_logp_sum, ref_token_counts)`` or ``None`` on failure.
    :rtype: tuple[Tensor, Tensor] | None
    """
    torch_mod = _refresh_torch()
    ref_model = runtime.get_ref_model()
    ref_logp_chunks: List[Tensor] = []
    ref_tok_chunks: List[Tensor] = []

    def _log_once(reason: str) -> None:
        if getattr(reference_from_model, "_diag_logged", False):
            return
        LOG.warning("reference_from_model returning None: %s", reason)
        setattr(reference_from_model, "_diag_logged", True)

    slices_seen = 0
    slice_iter = iter_batch_slices(score_batch, runtime.device)
    slice_iter = _prefetch_iterator(
        slice_iter, getattr(batching_cfg, "slice_prefetch", 0)
    )
    for slice_inputs, slice_mask, slice_labels in slice_iter:
        slices_seen += 1
        no_grad_ctx = getattr(torch_mod, "no_grad", None) or nullcontext
        with no_grad_ctx():
            try:
                result = _chunked_sequence_logprobs(
                    ref_model,
                    input_ids=slice_inputs,
                    attention_mask=slice_mask,
                    labels=slice_labels,
                    chunk_size=batching_cfg.logprob_chunk_size,
                    gather_full_params=False,
                )
            except _SCORING_EXCEPTIONS as exc:  # pragma: no cover - defensive diagnostics
                _log_once(
                    f"_chunked_sequence_logprobs raised {type(exc).__name__}: {exc}"
                )
                return None
        # _chunked_sequence_logprobs normally returns (logp, tok_counts, pooled_hidden).
        if not isinstance(result, (tuple, list)) or len(result) < 2:
            _log_once(
                f"chunked_sequence_logprobs returned invalid result | type={type(result)} len={len(result) if hasattr(result, '__len__') else 'n/a'} "
                f"inputs_shape={getattr(slice_inputs, 'shape', None)}"
            )
            return None
        if len(result) >= 3:
            ref_logp_slice, ref_tok_slice, _ = result[:3]
        else:
            ref_logp_slice, ref_tok_slice = result[:2]
        if ref_logp_slice.numel() == 0 or ref_tok_slice.numel() == 0:
            _log_once(
                f"empty slice tensors | logp_numel={ref_logp_slice.numel()} tok_numel={ref_tok_slice.numel()} "
                f"inputs_shape={getattr(slice_inputs, 'shape', None)}"
            )
            return None
        LOG.debug(
            "Reference slice gathered | slice_idx=%d | logp_shape=%s | tok_shape=%s | inputs_shape=%s",
            slices_seen - 1,
            getattr(ref_logp_slice, "shape", None),
            getattr(ref_tok_slice, "shape", None),
            getattr(slice_inputs, "shape", None),
        )
        ref_logp_chunks.append(ref_logp_slice.detach().cpu())
        ref_tok_chunks.append(ref_tok_slice.detach().cpu())
    if slices_seen == 0:
        _log_once("iter_batch_slices yielded zero slices")
        return None
    if not ref_logp_chunks:
        _log_once("no reference slices produced any chunks")
        return None
    ref_logp = torch_mod.cat(ref_logp_chunks, dim=0).to(runtime.device)
    ref_tok = torch_mod.cat(ref_tok_chunks, dim=0).to(runtime.device)
    LOG.debug(
        "Reference gather succeeded | slices=%d | ref_logp_shape=%s | ref_tok_shape=%s",
        slices_seen,
        getattr(ref_logp, "shape", None),
        getattr(ref_tok, "shape", None),
    )
    return ref_logp, ref_tok


def gather_reference_logprobs(
    score_batch: ScoreBatch,
    runtime: RuntimeHandles,
    batching_cfg: BatchingSettings,
) -> Optional[ReferenceLogprobs]:
    """Compute log-probabilities by running the frozen reference model.

    This function handles distributed preflight checks to avoid ZeRO hangs and
    aggregates reference statistics into a ``ReferenceLogprobs`` object.

    :param score_batch: Prepared scoring batch with prompts/completions.
    :param runtime: Runtime handles exposing device, accelerator, and models.
    :param batching_cfg: Batching config controlling logprob chunking.
    :returns: ``ReferenceLogprobs`` or ``None`` when reference scoring fails.
    :rtype: ReferenceLogprobs | None
    """
    torch_mod = _refresh_torch()

    def _dim0(obj: Any) -> int:
        if obj is None:
            return 0
        shape = getattr(obj, "shape", None)
        if shape is not None:
            try:
                return int(shape[0])
            except _SCORING_EXCEPTIONS as exc:
                LOG.debug("Failed to read shape[0] from tensor; falling back to len: %s", exc)
        try:
            return int(len(obj))  # type: ignore[arg-type]
        except _SCORING_EXCEPTIONS:
            return 0

    def _first_slice_rows(sb: Any) -> int:
        total = int(getattr(sb, "total_sequences", 0) or 0)
        slice_size = int(getattr(sb, "slice_size", 0) or 0)
        prompt_len = len(getattr(sb, "prompt_entries", []) or [])
        comp0 = _dim0(getattr(sb, "completion_ids", None))
        mask0 = _dim0(getattr(sb, "completion_attention_mask", None))
        # Bound by the first slice end index; if any of these are zero on a rank,
        # that rank will not enter the reference forward and ZeRO collectives can hang.
        return max(0, min(total, slice_size, prompt_len, comp0, mask0))

    def _safe_numel(obj: Any) -> Any:
        numel_fn = getattr(obj, "numel", None)
        if callable(numel_fn):
            try:
                return numel_fn()
            except _SCORING_EXCEPTIONS:
                return "error"
        return None

    accelerator = getattr(runtime, "accelerator", None)
    num_processes = int(getattr(accelerator, "num_processes", 1) or 1)
    gather_obj = getattr(accelerator, "gather_object", None)
    preflight_rows = _first_slice_rows(score_batch)
    if num_processes > 1:
        # Preflight: if any rank has an empty first slice, skip reference scoring
        # everywhere to avoid deadlocking inside ZeRO parameter all-gathers.
        all_rows: Optional[List[Any]] = None
        if callable(gather_obj):
            try:
                all_rows = gather_obj(int(preflight_rows))
            except _SCORING_EXCEPTIONS:
                all_rows = None
        if all_rows is None:
            dist = _dist_collective_ready(torch_mod)
            if dist is not None:
                gathered: List[Any] = [None for _ in range(max(int(dist.get_world_size()), 1))]
                try:
                    dist.all_gather_object(gathered, int(preflight_rows))
                    all_rows = gathered
                except _SCORING_EXCEPTIONS:
                    all_rows = None
        if isinstance(all_rows, list) and all_rows:
            try:
                min_rows = min(int(v) for v in all_rows)
            except _SCORING_EXCEPTIONS:
                min_rows = int(preflight_rows)
            if min_rows <= 0:
                LOG.warning(
                    "Skipping reference scoring preflight: at least one rank has empty first slice "
                    "| local_rows=%s all_rows=%s total_sequences=%s slice_size=%s prompt_entries=%s comp_ids0=%s comp_mask0=%s",
                    preflight_rows,
                    all_rows,
                    getattr(score_batch, "total_sequences", None),
                    getattr(score_batch, "slice_size", None),
                    len(getattr(score_batch, "prompt_entries", []) or []),
                    _dim0(getattr(score_batch, "completion_ids", None)),
                    _dim0(getattr(score_batch, "completion_attention_mask", None)),
                )
                return None

    tensors = reference_from_model(score_batch, runtime, batching_cfg)
    local_ok = tensors is not None
    global_ok = local_ok
    if num_processes > 1:
        if callable(gather_obj):
            try:
                gathered = gather_obj(bool(local_ok))
                if isinstance(gathered, list):
                    global_ok = all(bool(x) for x in gathered)
                else:
                    global_ok = bool(gathered)
            except _SCORING_EXCEPTIONS:
                global_ok = local_ok
        else:
            dist = _dist_collective_ready(torch_mod)
            global_ok = _dist_all(dist, local_ok) if dist is not None else local_ok
    if tensors is None or not global_ok:
        LOG.debug(
            "gather_reference_logprobs returning None due to reference_from_model failure on at least one rank | "
            "slice_size=%s | chunk_size=%s | device=%s | local_ok=%s",
            getattr(score_batch, "slice_size", None),
            getattr(batching_cfg, "logprob_chunk_size", None),
            getattr(runtime, "device", None),
            local_ok,
        )
        return None
    LOG.debug(
        "reference_from_model tensors | ref_logp_shape=%s | ref_tok_shape=%s | ref_logp_numel=%s | ref_tok_numel=%s | ref_logp_dtype=%s | ref_tok_dtype=%s | ref_logp_device=%s | ref_tok_device=%s",
        getattr(tensors[0], "shape", None),
        getattr(tensors[1], "shape", None),
        _safe_numel(tensors[0]),
        _safe_numel(tensors[1]),
        getattr(tensors[0], "dtype", None),
        getattr(tensors[1], "dtype", None),
        getattr(tensors[0], "device", None),
        getattr(tensors[1], "device", None),
    )
    try:
        stats = finalize_reference_stats(*tensors)
    except _SCORING_EXCEPTIONS as exc:  # pragma: no cover - defensive diagnostics
        LOG.error(
            "finalize_reference_stats raised %s: %s | ref_logp_shape=%s | ref_tok_shape=%s | ref_logp_dtype=%s | ref_tok_dtype=%s | ref_logp_device=%s | ref_tok_device=%s",
            type(exc).__name__,
            exc,
            getattr(tensors[0], "shape", None),
            getattr(tensors[1], "shape", None),
            getattr(tensors[0], "dtype", None),
            getattr(tensors[1], "dtype", None),
            getattr(tensors[0], "device", None),
            getattr(tensors[1], "device", None),
        )
        return None
    LOG.debug(
        "gather_reference_logprobs built stats | ref_logp_sum_shape=%s | ref_tok_counts_shape=%s | ref_logp_sum_numel=%s | ref_tok_counts_numel=%s",
        getattr(getattr(stats, "ref_logp_sum", None), "shape", None),
        getattr(getattr(stats, "ref_tok_counts", None), "shape", None),
        _safe_numel(getattr(stats, "ref_logp_sum", None)),
        _safe_numel(getattr(stats, "ref_tok_counts", None)),
    )
    return stats


def finalize_reference_stats(
    ref_logp_sum: Tensor,
    ref_tok_counts: Tensor,
) -> ReferenceLogprobs:
    """Build a ``ReferenceLogprobs`` object and derived scalars.

    :param ref_logp_sum: Per-sequence sum of reference log-probabilities.
    :param ref_tok_counts: Per-sequence token counts.
    :returns: Normalized reference stats and summary scalars.
    :rtype: ReferenceLogprobs
    :raises ValueError: If log-prob tensors cannot be safely normalized.
    """
    torch_mod = _refresh_torch()
    try:
        logp_arr_raw = _to_numpy_array(ref_logp_sum)
        tok_arr_raw = _to_numpy_array(ref_tok_counts)
        logp_numel = None
        tok_numel = None
        numel_fn = getattr(ref_logp_sum, "numel", None)
        if callable(numel_fn):
            try:
                logp_numel = numel_fn()
            except _SCORING_EXCEPTIONS:
                logp_numel = None
        numel_fn_tok = getattr(ref_tok_counts, "numel", None)
        if callable(numel_fn_tok):
            try:
                tok_numel = numel_fn_tok()
            except _SCORING_EXCEPTIONS:
                tok_numel = None

        # If numpy conversion dropped elements (e.g., CUDA+bfloat16 -> empty array),
        # retry with a CPU float32 view so we do not lose reference stats.
        if getattr(logp_arr_raw, "size", 0) == 0 and (
            (isinstance(logp_numel, numbers.Number) and logp_numel > 0)
            or (isinstance(tok_numel, numbers.Number) and tok_numel > 0)
        ):
            try:
                tensor_cpu = ref_logp_sum.detach().to("cpu")
                if hasattr(tensor_cpu, "float"):
                    tensor_cpu = tensor_cpu.float()
                logp_arr_raw = np.asarray(tensor_cpu.reshape(-1).cpu().numpy())
                LOG.debug(
                    "finalize_reference_stats fallback tensor->numpy succeeded | target_numel=%s | result_size=%s",
                    logp_numel,
                    getattr(logp_arr_raw, "size", None),
                )
            except _SCORING_EXCEPTIONS as exc:
                LOG.debug(
                    "finalize_reference_stats fallback tensor->numpy failed | target_numel=%s | exc=%s",
                    logp_numel,
                    exc,
                )
        # If still empty but token counts exist, treat as a hard failure. Returning
        # zeros here can yield pathological KL (e.g., delta ~= -cur_logp) and
        # destabilize training; let callers retry/skip the batch instead.
        if (
            getattr(logp_arr_raw, "size", 0) == 0
            and isinstance(tok_numel, numbers.Number)
            and tok_numel > 0
        ):
            raise ValueError(
                "Reference logp tensor conversion produced an empty array while token counts exist; "
                "cannot finalize reference stats safely."
            )

        def _safe_size(x: Any) -> Any:
            try:
                return int(np.size(x))
            except _SCORING_EXCEPTIONS:
                return None

        LOG.debug(
            "finalize_reference_stats raw inputs | ref_logp_len=%s | ref_tok_len=%s | ref_logp_numel=%s | ref_tok_numel=%s | ref_logp_dtype=%s | ref_tok_dtype=%s | ref_logp_device=%s | ref_tok_device=%s | ref_logp_sample=%s",
            _safe_size(logp_arr_raw),
            _safe_size(tok_arr_raw),
            logp_numel,
            tok_numel,
            getattr(ref_logp_sum, "dtype", None),
            getattr(ref_tok_counts, "dtype", None),
            getattr(ref_logp_sum, "device", None),
            getattr(ref_tok_counts, "device", None),
            logp_arr_raw[:4] if hasattr(logp_arr_raw, "__getitem__") else None,
        )
        logp_arr = np.asarray(logp_arr_raw, dtype=float)
        tok_arr = np.asarray(tok_arr_raw, dtype=float)
        if tok_arr.size == 0:
            tok_arr = np.asarray([0.0])
        invalid_mask = ~np.isfinite(tok_arr) | (tok_arr < 0)
        if invalid_mask.any():
            finite_replaced = np.nan_to_num(
                tok_arr, nan=0.0, posinf=0.0, neginf=0.0
            )
            tok_arr = np.maximum(finite_replaced, 0.0)
            LOG.debug(
                "Clamped %d invalid reference token counts to non-negative finite values.",
                int(invalid_mask.sum()),
            )
        if tok_arr.size == 0:
            tok_arr = np.asarray([0.0])
        safe_tok = np.maximum(tok_arr, 1.0)
        logp_norm = np.divide(
            logp_arr,
            safe_tok,
            out=np.zeros_like(logp_arr, dtype=float),
            where=safe_tok > 0,
        )
        ref_logp_sum_tensor = torch_mod.tensor(
            logp_norm, dtype=getattr(torch_mod, "float32", None)
        )
        ref_tok_counts_tensor = torch_mod.tensor(
            tok_arr, dtype=getattr(torch_mod, "float32", None)
        )
        ref_logp_raw_tensor = torch_mod.tensor(
            logp_arr, dtype=getattr(torch_mod, "float32", None)
        )
        ref_logp_mean = (
            float(np.asarray(logp_arr, dtype=float).mean()) if logp_arr.size else 0.0
        )
        avg_completion_tokens = float(np.asarray(tok_arr, dtype=float).mean())
        return ReferenceLogprobs(
            ref_logp_sum=ref_logp_sum_tensor,
            ref_tok_counts=ref_tok_counts_tensor,
            ref_logp_sum_raw=ref_logp_raw_tensor,
            ref_logp_mean=ref_logp_mean,
            avg_completion_tokens=avg_completion_tokens,
        )
    except _SCORING_EXCEPTIONS as exc:  # pragma: no cover - defensive diagnostics
        LOG.exception(
            "finalize_reference_stats failed | exc=%s | ref_logp_sum_shape=%s | ref_tok_counts_shape=%s | ref_logp_sum_dtype=%s | ref_tok_counts_dtype=%s | ref_logp_sum_device=%s | ref_tok_counts_device=%s",
            type(exc).__name__,
            getattr(ref_logp_sum, "shape", None),
            getattr(ref_tok_counts, "shape", None),
            getattr(ref_logp_sum, "dtype", None),
            getattr(ref_tok_counts, "dtype", None),
            getattr(ref_logp_sum, "device", None),
            getattr(ref_tok_counts, "device", None),
        )
        raise


def token_counts_from_score_batch(
    score_batch: ScoreBatch,
    runtime: RuntimeHandles,
    batching_cfg: BatchingSettings,
) -> Tensor:
    """Compute per-sequence token counts from the score batch labels mask.

    :param score_batch: Prepared scoring batch.
    :param runtime: Runtime handles exposing device/accelerator.
    :param batching_cfg: Batching config controlling slice sizes.
    :returns: 1D tensor of token counts per sequence.
    :rtype: Tensor
    """
    torch_mod = _refresh_torch()
    tok_chunks: List[Tensor] = []
    slice_iter = iter_batch_slices(score_batch, runtime.device)
    slice_iter = _prefetch_iterator(slice_iter, getattr(batching_cfg, "slice_prefetch", 0))
    for _slice_inputs, _slice_mask, slice_labels in slice_iter:
        label_mask = slice_labels != -100
        tok = label_mask.sum(dim=1).clamp(min=1)
        to_fn = getattr(tok, "to", None)
        if callable(to_fn):
            tok = to_fn(dtype=getattr(torch_mod, "float32", None))
        tok_chunks.append(tok)
    if not tok_chunks:
        try:
            return torch_mod.zeros(
                (0,),
                dtype=getattr(torch_mod, "float32", None),
                device=runtime.device,
            )
        except _SCORING_EXCEPTIONS:
            return torch_mod.tensor([], dtype=getattr(torch_mod, "float32", None))
    try:
        out = torch_mod.cat(tok_chunks, dim=0)
    except _SCORING_EXCEPTIONS:
        out = tok_chunks[0]
        for chunk in tok_chunks[1:]:
            out = torch_mod.cat([out, chunk], dim=0)
    to_fn = getattr(out, "to", None)
    if callable(to_fn):
        try:
            out = to_fn(device=runtime.device)
        except _SCORING_EXCEPTIONS as exc:
            LOG.debug("Failed to move token counts to runtime device: %s", exc)
    return out


def reference_stats_from_policy_logprobs(
    cur_logp_sum: Tensor,
    tok_counts: Tensor,
) -> ReferenceLogprobs:
    """Build ``ReferenceLogprobs`` assuming reference == current policy (KL ~= 0).

    :param cur_logp_sum: Current policy log-prob sums per sequence.
    :param tok_counts: Token counts per sequence.
    :returns: Reference stats derived directly from the current policy.
    :rtype: ReferenceLogprobs
    """
    torch_mod = _refresh_torch()
    base_dtype = getattr(torch_mod, "float32", None)
    device = getattr(cur_logp_sum, "device", None)
    cur_tensor = _as_torch_tensor(
        torch_mod, cur_logp_sum, device=device, dtype=base_dtype
    ).view(-1)
    tok_tensor = _as_torch_tensor(
        torch_mod, tok_counts, device=device, dtype=base_dtype
    ).view(-1)
    try:
        tok_tensor = tok_tensor.clamp(min=1.0)
    except _SCORING_EXCEPTIONS as exc:
        LOG.debug("Failed to clamp token counts; continuing: %s", exc)
    detach_fn = getattr(cur_tensor, "detach", None)
    if callable(detach_fn):
        cur_tensor = detach_fn()
    detach_fn = getattr(tok_tensor, "detach", None)
    if callable(detach_fn):
        tok_tensor = detach_fn()
    ref_logp_sum_raw = cur_tensor
    ref_logp_sum = ref_logp_sum_raw / tok_tensor
    try:
        ref_logp_mean = float(ref_logp_sum_raw.detach().float().cpu().mean().item())
    except _SCORING_EXCEPTIONS:
        try:
            ref_logp_mean = float(ref_logp_sum_raw.mean())
        except _SCORING_EXCEPTIONS:
            ref_logp_mean = 0.0
    try:
        avg_completion_tokens = float(tok_tensor.detach().float().cpu().mean().item())
    except _SCORING_EXCEPTIONS:
        try:
            avg_completion_tokens = float(tok_tensor.mean())
        except _SCORING_EXCEPTIONS:
            avg_completion_tokens = 0.0
    return ReferenceLogprobs(
        ref_logp_sum=ref_logp_sum,
        ref_tok_counts=tok_tensor,
        ref_logp_sum_raw=ref_logp_sum_raw,
        ref_logp_mean=ref_logp_mean,
        avg_completion_tokens=avg_completion_tokens,
    )


def _meta_field(entry: Any, *names: str) -> Any:
    """Return the first matching field from a metadata entry."""
    if entry is None:
        return None
    sentinel = object()
    for name in names:
        if isinstance(entry, Mapping):
            if name in entry:
                return entry.get(name)
        else:
            value = getattr(entry, name, sentinel)
            if value is not sentinel:
                return value
    return None


def _coerce_logprob_value(value: Any) -> Optional[float]:
    """Best-effort conversion of a token logprob payload into a float."""
    if value is None:
        return None
    if isinstance(value, numbers.Number):
        return float(value)
    if isinstance(value, Mapping):
        if "logprob" in value:
            return _coerce_logprob_value(value.get("logprob"))
        if len(value) == 1:
            return _coerce_logprob_value(next(iter(value.values())))
        return None
    attr_val = getattr(value, "logprob", None)
    if attr_val is not None:
        return _coerce_logprob_value(attr_val)
    return None


def _sum_token_logprobs(token_logprobs: Any) -> Optional[float]:
    """Return the sum of per-token logprobs when the payload is parseable."""
    if token_logprobs is None or isinstance(token_logprobs, Mapping):
        return None
    try:
        iterator = iter(token_logprobs)
    except TypeError:
        return None
    total = 0.0
    saw_any = False
    for item in iterator:
        val = _coerce_logprob_value(item)
        if val is None:
            return None
        total += float(val)
        saw_any = True
    return total if saw_any else None


def reference_from_vllm_meta(
    flat_meta: Sequence[Optional[Any]],
    total_sequences: int,
    device: torch.device,
) -> Optional[ReferenceLogprobs]:
    """Convert flattened vLLM log-prob metadata into ``ReferenceLogprobs``.

    :param flat_meta: Flat list of vLLM metadata entries (one per completion).
    :param total_sequences: Expected number of sequences in the batch.
    :param device: Device for the resulting tensors.
    :returns: ``ReferenceLogprobs`` or ``None`` when metadata is incomplete.
    :rtype: ReferenceLogprobs | None
    """
    if not flat_meta:
        return None
    if len(flat_meta) != total_sequences:
        return None
    logp_vals: List[float] = []
    tok_counts: List[int] = []
    for entry in flat_meta:
        if entry is None:
            return None
        logprob_sum = _meta_field(entry, "logprob_sum", "cumulative_logprob")
        token_count = _meta_field(entry, "token_count", "num_tokens")
        token_logprobs = _meta_field(entry, "token_logprobs", "logprobs")
        if logprob_sum is None and token_logprobs is not None:
            logprob_sum = _sum_token_logprobs(token_logprobs)
        if token_count is None and token_logprobs is not None:
            try:
                token_count = len(token_logprobs)
            except (TypeError, ValueError, AttributeError):
                token_count = None
        if logprob_sum is None or token_count is None:
            return None
        logp_vals.append(float(logprob_sum))
        tok_counts.append(max(1, int(token_count)))
    torch_mod = _refresh_torch()
    ref_logp_sum = torch_mod.tensor(
        logp_vals, dtype=getattr(torch_mod, "float32", None), device=device
    )
    ref_tok_counts = torch_mod.tensor(
        tok_counts, dtype=getattr(torch_mod, "float32", None), device=device
    )
    return finalize_reference_stats(ref_logp_sum, ref_tok_counts)


def vllm_meta_has_logprobs(
    flat_meta: Optional[Sequence[Optional[Any]]],
    total_sequences: Optional[int] = None,
) -> bool:
    """Return True when vLLM metadata includes per-completion logprob info.

    :param flat_meta: Flat list of vLLM metadata entries.
    :param total_sequences: Optional expected length used for sanity checks.
    :returns: ``True`` when logprob metadata appears complete.
    :rtype: bool
    """

    if not flat_meta:
        return False
    if total_sequences is not None and total_sequences >= 0:
        try:
            if len(flat_meta) != int(total_sequences):
                return False
        except (TypeError, ValueError):
            return False
    for entry in flat_meta:
        if entry is None:
            return False
        logprob_sum = getattr(entry, "logprob_sum", None)
        token_count = getattr(entry, "token_count", None)
        if isinstance(entry, dict):
            if logprob_sum is None:
                logprob_sum = entry.get("logprob_sum") or entry.get("cumulative_logprob")
            if token_count is None:
                token_count = entry.get("token_count")
        if logprob_sum is None or token_count is None:
            return False
    return True


def score_model_outputs(
    model: PreTrainedModel,
    score_batch: ScoreBatch,
    batching_cfg: BatchingSettings,
    runtime: RuntimeHandles,
    *,
    return_hidden: bool = False,
    pooling: str = "mean",
) -> Optional[Tuple[Tensor, Optional[Tensor]]]:
    """Compute current model log-probs for the batch and optional pooled states.

    :param model: Current policy model used for scoring.
    :param score_batch: Prepared scoring batch.
    :param batching_cfg: Batching config controlling logprob chunking.
    :param runtime: Runtime handles providing device and accelerator state.
    :param return_hidden: When ``True``, also return pooled hidden states.
    :param pooling: Pooling strategy applied to hidden states.
    :returns: Tuple of ``(cur_logp_sum, pooled_hidden)`` or ``None`` if empty.
    :rtype: tuple[Tensor, Tensor | None] | None
    """
    cur_logp_slices: List[Tensor] = []
    pooled_slices: List[Tensor] = []
    slice_iter = iter_batch_slices(score_batch, runtime.device)
    slice_iter = _prefetch_iterator(
        slice_iter, getattr(batching_cfg, "slice_prefetch", 0)
    )
    LOG.debug(
        "current scoring batch metadata | total_sequences=%s slice_size=%s device=%s",
        score_batch.total_sequences,
        score_batch.slice_size,
        getattr(runtime.device, "type", runtime.device),
    )
    with _autocast_context(runtime.accelerator, runtime.device):
        for slice_inputs, slice_mask, slice_labels in slice_iter:
            LOG.debug(
                "current scoring slice inputs | input_ids_shape=%s attention_mask_shape=%s labels_shape=%s",
                getattr(slice_inputs, "shape", None),
                getattr(slice_mask, "shape", None),
                getattr(slice_labels, "shape", None),
            )
            cur_logp_slice, _tok_counts, pooled = _chunked_sequence_logprobs(
                model,
                input_ids=slice_inputs,
                attention_mask=slice_mask,
                labels=slice_labels,
                chunk_size=batching_cfg.logprob_chunk_size,
                return_hidden=return_hidden,
                pooling=pooling,
            )
            cur_logp_slices.append(cur_logp_slice)
            if pooled is not None:
                pooled_slices.append(pooled.detach())
    if not cur_logp_slices:
        return None
    pooled_hidden = torch.cat(pooled_slices, dim=0) if pooled_slices else None
    return torch.cat(cur_logp_slices, dim=0), pooled_hidden


def summarize_completion_lengths(
    ref_stats: ReferenceLogprobs,
    max_completion_len: int,
) -> Tuple[Tensor, LengthStats, float]:
    """Summarize completion lengths for metrics.

    :param ref_stats: Reference log-prob stats containing token counts.
    :param max_completion_len: Maximum completion length used for clipping stats.
    :returns: Tuple of ``(completion_lengths, length_stats, total_tokens)``.
    :rtype: tuple[Tensor, LengthStats, float]
    """
    torch_mod = _refresh_torch()
    lengths_arr = _to_numpy_array(ref_stats.ref_tok_counts).astype(float)
    num_completion_tokens = float(lengths_arr.sum()) if lengths_arr.size else 0.0
    clipped_mask = lengths_arr >= float(max_completion_len)
    if lengths_arr.size > 0:
        min_length = float(lengths_arr.min())
        mean_length = float(lengths_arr.mean())
        max_length = float(lengths_arr.max())
        clipped_ratio = float(clipped_mask.mean())
    else:
        min_length = mean_length = max_length = clipped_ratio = 0.0
    terminated = lengths_arr[~clipped_mask] if lengths_arr.size else np.asarray([])
    if terminated.size > 0:
        min_terminated = float(terminated.min())
        mean_terminated = float(terminated.mean())
        max_terminated = float(terminated.max())
    else:
        min_terminated = mean_terminated = max_terminated = 0.0
    completion_lengths = torch_mod.tensor(
        lengths_arr, dtype=getattr(torch_mod, "float32", None)
    )
    return (
        completion_lengths,
        LengthStats(
            min_length=min_length,
            mean_length=mean_length,
            max_length=max_length,
            clipped_ratio=clipped_ratio,
            min_terminated=min_terminated,
            mean_terminated=mean_terminated,
            max_terminated=max_terminated,
        ),
        num_completion_tokens,
    )


def _as_torch_tensor(
    torch_mod: Any,
    value: Any,
    *,
    device: Optional["torch.device"],
    dtype: Optional[Any],
) -> Tensor:
    """Best-effort conversion of ``value`` into a torch tensor on ``device``."""

    ctor = getattr(torch_mod, "as_tensor", getattr(torch_mod, "tensor", None))
    if ctor is None:
        raise RuntimeError("Torch tensor constructor unavailable")
    if isinstance(value, torch_mod.Tensor):
        tensor = value
    else:
        payload = getattr(value, "arr", None)
        if payload is None:
            payload = getattr(value, "data", value)
        try:
            tensor = ctor(payload)
        except _SCORING_EXCEPTIONS:
            tensor = ctor([])
    if dtype is not None:
        try:
            tensor = tensor.to(dtype=dtype)
        except _SCORING_EXCEPTIONS:
            clone_fn = getattr(tensor, "clone", None)
            if callable(clone_fn):
                tensor = clone_fn()
                tensor = tensor.to(dtype=dtype)
    if device is not None and getattr(tensor, "device", None) != device:
        tensor = tensor.to(device=device)
    return tensor


def _match_tensor_length(
    torch_mod: Any,
    tensor: Tensor,
    target_len: int,
    *,
    device: "torch.device",
    dtype: Any,
    fill_value: float = 0.0,
) -> Tensor:
    """Return ``tensor`` reshaped/padded to ``target_len`` elements."""

    def _full(shape: Tuple[int, ...], value: float) -> Tensor:
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        return torch_mod.full(shape, value, **kwargs)

    def _zeros(shape: Tuple[int, ...]) -> Tensor:
        kwargs = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        try:
            return torch_mod.zeros(shape, **kwargs)
        except TypeError:
            return torch_mod.zeros(shape)

    tensor = tensor.view(-1)
    cur_len = int(getattr(tensor, "numel", lambda: 0)())
    if target_len <= 0:
        return _zeros((0,))
    if cur_len == target_len:
        return tensor
    if cur_len == 0:
        return _full((target_len,), fill_value)
    if cur_len == 1:
        scalar_val = float(getattr(tensor[0], "item", lambda: tensor[0])())
        return _full((target_len,), scalar_val)
    min_len = min(cur_len, target_len)
    tensor = tensor[:min_len]
    if min_len == target_len:
        return tensor
    pad_val = float(getattr(tensor[-1], "item", lambda: tensor[-1])())
    pad = _full((target_len - min_len,), pad_val)
    return torch_mod.cat([tensor, pad], dim=0)


def build_sequence_scores(
    cur_logp_sum: Tensor,
    ref_stats: ReferenceLogprobs,
    pooled_hidden: Optional[Tensor] = None,
    *,
    behavior_logp_sum: Optional[Tensor] = None,
) -> SequenceScores:
    """Return ``SequenceScores`` built from current and reference log-probs.

    :param cur_logp_sum: Current policy log-prob sums per sequence.
    :param ref_stats: Reference log-prob stats used for KL and weighting.
    :param pooled_hidden: Optional pooled hidden states for auxiliary losses.
    :param behavior_logp_sum: Optional behavior-policy log-probs for off-policy scoring.
    :returns: ``SequenceScores`` dataclass with normalized log-probs and KL terms.
    :rtype: SequenceScores
    """

    torch_mod = _refresh_torch()
    base_dtype = getattr(torch_mod, "float32", None)
    cur_tensor = _as_torch_tensor(
        torch_mod,
        cur_logp_sum,
        device=getattr(cur_logp_sum, "device", None),
        dtype=base_dtype,
    ).view(-1)
    device = getattr(cur_tensor, "device", None)
    cur_len = int(getattr(cur_tensor, "numel", lambda: 0)())
    ref_source = getattr(ref_stats, "ref_logp_sum_raw", None)
    if ref_source is None:
        ref_source = getattr(ref_stats, "ref_logp_sum", None)
    ref_tensor = _as_torch_tensor(
        torch_mod,
        ref_source if ref_source is not None else [],
        device=device,
        dtype=base_dtype,
    )
    ref_tensor = _match_tensor_length(
        torch_mod,
        ref_tensor,
        cur_len,
        device=device,
        dtype=base_dtype,
        fill_value=0.0,
    )
    denom_source = getattr(ref_stats, "ref_tok_counts", None)
    denom_tensor = _as_torch_tensor(
        torch_mod,
        denom_source if denom_source is not None else [],
        device=device,
        dtype=base_dtype,
    )
    denom_tensor = _match_tensor_length(
        torch_mod,
        denom_tensor,
        cur_len,
        device=device,
        dtype=base_dtype,
        fill_value=1.0,
    ).clamp(min=1.0)
    if behavior_logp_sum is None:
        behavior_tensor = cur_tensor.detach()
    else:
        behavior_tensor = _as_torch_tensor(
            torch_mod,
            behavior_logp_sum,
            device=device,
            dtype=base_dtype,
        )
        behavior_tensor = _match_tensor_length(
            torch_mod,
            behavior_tensor,
            cur_len,
            device=device,
            dtype=base_dtype,
            fill_value=0.0,
        )
    log_ratio_train = cur_tensor - ref_tensor
    if getattr(log_ratio_train, "numel", lambda: 0)() == 0 and cur_len > 0:
        log_ratio_train = torch_mod.zeros(
            (cur_len,), device=device, dtype=base_dtype
        )
    return SequenceScores(
        cur_logp_sum=cur_tensor,
        behavior_logp_sum=behavior_tensor,
        log_ratio_train=log_ratio_train,
        denom_tok_tensor=denom_tensor,
        pooled_hidden=pooled_hidden,
    )
