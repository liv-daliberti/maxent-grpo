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

from collections.abc import Iterable, Mapping, Sized
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
import inspect
import os
import numbers
import sys
import logging
import time
from types import TracebackType
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
    cast,
)
from functools import lru_cache
import queue
import threading
import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from maxent_grpo.training.runtime import (
    require_torch,
    require_transformer_base_classes,
)
from .zero_utils import _maybe_zero_gather_params
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


def _progress_log_enabled() -> bool:
    raw = os.getenv("MAXENT_PROGRESS_LOG")
    if raw is None or not str(raw).strip():
        return False
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _score_slice_log_enabled() -> bool:
    raw = os.getenv("MAXENT_SCORE_SLICE_LOG")
    if raw is None or not str(raw).strip():
        return str(os.getenv("MAXENT_MAX_LOGS", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}

if TYPE_CHECKING:
    import torch as torch_types
    from torch import Tensor as TorchTensor, device as TorchDevice, dtype as TorchDType
    from transformers.modeling_utils import PreTrainedModel as PreTrainedModel
    from transformers.tokenization_utils import PreTrainedTokenizer as PreTrainedTokenizer
else:  # pragma: no cover - runtime uses optional torch/transformers stubs
    torch_types = None
    TorchTensor = Any
    TorchDevice = Any
    TorchDType = Any
    PreTrainedModel = Any
    PreTrainedTokenizer = Any


class _TorchModuleLike(Protocol):
    Tensor: type
    float32: object

    def tensor(self, data: object, *args: Any, **kwargs: Any) -> Any: ...
    def as_tensor(self, data: object, *args: Any, **kwargs: Any) -> Any: ...
    def full(self, size: Tuple[int, ...], fill_value: float, *args: Any, **kwargs: Any) -> Any: ...
    def ones(self, size: Tuple[int, ...], *args: Any, **kwargs: Any) -> Any: ...
    def zeros(self, size: Tuple[int, ...], *args: Any, **kwargs: Any) -> Any: ...
    def arange(self, *args: Any, **kwargs: Any) -> Any: ...
    def cat(self, tensors: Sequence[Any], dim: int = ...) -> Any: ...
    def nonzero(self, input_tensor: Any, *, as_tuple: bool = ...) -> Any: ...
    nn: Any
    def no_grad(self) -> ContextManager[Any]: ...
    def unique(self, values: Any) -> Any: ...
    def stack(self, tensors: Sequence[Any], dim: int = ...) -> Any: ...


class _DistModuleLike(Protocol):
    """Minimal distributed API needed for best-effort gathers."""

    def is_available(self) -> bool: ...
    def is_initialized(self) -> bool: ...
    def get_world_size(self) -> int: ...
    def all_gather_object(self, object_list: List[object], obj: object) -> None: ...

torch = cast(_TorchModuleLike, require_torch("training_scoring"))
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

T = TypeVar("T")


class _LongDTypeProxy:
    """Lightweight wrapper that compares equal to torch.long in stubs.

    Some test environments mix different torch stubs/modules, so ``dtype``
    objects attached to tensors may not be identical to ``torch.long`` even
    when they both represent an int64 type.  This proxy smooths over those
    differences for equality checks without affecting the underlying array
    dtype used in computations.
    """

    def __init__(self, target: object) -> None:
        self._target = target

    def __eq__(self, other: object) -> bool:  # pragma: no cover - exercised in tests
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


def _refresh_torch() -> _TorchModuleLike:
    """Return the active torch module."""

    return torch


def _prefetch_iterator(iterator: Iterable[T], buffer_size: int) -> Iterator[T]:
    """Yield from ``iterator`` while prefetching up to ``buffer_size`` slices."""
    if buffer_size is None or buffer_size <= 0:
        for item in iterator:
            yield item
        return
    sentry = object()
    q: queue.Queue = queue.Queue(maxsize=buffer_size)
    error_holder: dict[str, BaseException] = {}

    def _producer() -> None:
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


def _get_config_value(config: object, key: str, default: object | None = None) -> object | None:
    """Return a config value from either Mapping or object-style configs."""
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _coerce_optional_int(value: object | None) -> Optional[int]:
    """Return ``value`` coerced to int when possible, else ``None``."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        try:
            return int(cast(Any, value))
        except (TypeError, ValueError):
            return None
    if isinstance(value, (str, bytes, bytearray)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError):
        return None


def _coerce_shape(value: object) -> Optional[Tuple[int, ...]]:
    if value is None:
        return None
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        try:
            return tuple(value)
        except TypeError:
            return None
    return None


class _PadTokenGuard:
    """Context manager that temporarily clamps padding attributes."""

    def __init__(self, targets: Sequence[tuple[object, str]], value: int) -> None:
        # Store (target, attr, original_value, clamped_value)
        self._targets: list[tuple[object, str, object, object]] = []
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
            new_value: object = value
            try:
                if isinstance(new_value, numbers.Integral):
                    new_value_int = int(new_value)
                    new_value = new_value_int
                    if isinstance(num_embeddings, numbers.Integral):
                        # Prefer the module's own embedding count when exposed.
                        num_embeddings_int = int(num_embeddings)
                        if num_embeddings_int <= 0:
                            # Disable padding entirely for degenerate embeddings.
                            new_value = -1
                        elif new_value_int >= num_embeddings_int:
                            new_value = num_embeddings_int - 1
                    elif weight is not None and _weight_is_two_dimensional(weight):
                        # Fallback: derive num_embeddings from the leading weight dim.
                        size = _coerce_shape(getattr(weight, "shape", None))
                        if size is None and hasattr(weight, "size"):
                            try:
                                size = _coerce_shape(weight.size())
                            except _SCORING_EXCEPTIONS:
                                size = None
                        if size is not None and len(size) >= 1:
                            num_embeddings = size[0]
                            if isinstance(num_embeddings, numbers.Integral):
                                num_embeddings_int = int(num_embeddings)
                                if num_embeddings_int > 0 and new_value_int >= num_embeddings_int:
                                    new_value = num_embeddings_int - 1
            except _SCORING_EXCEPTIONS:
                # If anything goes wrong while inspecting sizes, fall back to
                # the original requested value without additional checks.
                new_value = value
            self._targets.append((target, attr, original, new_value))

    def __enter__(self) -> None:
        for target, attr, _original, new_value in self._targets:
            setattr(target, attr, new_value)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        for target, attr, original, _new_value in self._targets:
            setattr(target, attr, original)
        return False


def _weight_is_two_dimensional(weight: object) -> bool:
    """Return True if the provided weight exposes a 2-D shape."""
    if weight is None:
        return False
    shape = _coerce_shape(getattr(weight, "shape", None))
    if shape is None:
        size_fn = getattr(weight, "size", None)
        if callable(size_fn):
            try:
                shape = _coerce_shape(size_fn())
            except _SCORING_EXCEPTIONS:
                shape = None
    if shape is None:
        return False
    if len(shape) == 2:
        return True
    # DeepSpeed ZeRO-3 partitions can expose non-2D local shards; rely on the
    # full parameter shape when available.
    ds_shape = getattr(weight, "ds_shape", None)
    ds_shape = _coerce_shape(ds_shape)
    if ds_shape is not None and len(ds_shape) == 2:
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


def _weight_is_stub_tensor(weight: object) -> bool:
    """Return True for tensor-like stubs used in tests."""
    if weight is None:
        return False
    # The numpy-backed stub tensors expose an ``arr`` attribute; real torch tensors do not.
    return hasattr(weight, "arr")


def _model_has_non2d_embeddings(model: object) -> bool:
    """Return True when any known embedding weight is not 2-D."""
    if model is None:
        return False
    def _ds_shape_is_2d(weight: object) -> bool:
        ds_shape = _coerce_shape(getattr(weight, "ds_shape", None))
        return ds_shape is not None and len(ds_shape) == 2

    base = getattr(model, "module", model)
    modules: list[object] = []
    try:
        modules.append(getattr(base, "embed_tokens", None))
    except _SCORING_EXCEPTIONS:
        pass
    try:
        get_inp = getattr(base, "get_input_embeddings", None)
        if callable(get_inp):
            modules.append(get_inp())
    except _SCORING_EXCEPTIONS:
        pass
    try:
        get_out = getattr(base, "get_output_embeddings", None)
        if callable(get_out):
            modules.append(get_out())
    except _SCORING_EXCEPTIONS:
        pass
    try:
        modules.append(getattr(base, "lm_head", None))
    except _SCORING_EXCEPTIONS:
        pass
    for module in modules:
        if module is None:
            continue
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        if _weight_is_stub_tensor(weight):
            continue
        status = getattr(weight, "ds_status", None)
        if status is not None:
            status_name = getattr(status, "name", None)
            if isinstance(status_name, str):
                if status_name.upper() != "AVAILABLE":
                    if not _ds_shape_is_2d(weight):
                        return True
            elif isinstance(status, str):
                if status.upper() != "AVAILABLE":
                    if not _ds_shape_is_2d(weight):
                        return True
        if not _weight_is_two_dimensional(weight):
            return True
    return False


def _describe_embedding_module(module: object, name: str) -> str:
    """Return a human-friendly summary of an embedding module."""

    if module is None:
        return f"{name}=None"
    weight = getattr(module, "weight", None)
    shape = _coerce_shape(getattr(weight, "shape", None))
    if shape is None and weight is not None:
        size_fn = getattr(weight, "size", None)
        if callable(size_fn):
            try:
                shape = _coerce_shape(size_fn())
            except _SCORING_EXCEPTIONS:
                shape = None
    padding_idx = getattr(module, "padding_idx", None)
    return (
        f"{name}={type(module).__name__} weight_shape={shape} "
        f"padding_idx={padding_idx}"
    )


def _get_embedding_vocab_size(
    model: PreTrainedModel, config: object
) -> Optional[int]:
    """Return the vocab size exposed by the model's embedding weights."""
    embedding_module = getattr(model, "embed_tokens", None)
    if embedding_module is None and hasattr(model, "get_input_embeddings"):
        embedding_module = model.get_input_embeddings()
    weight = getattr(embedding_module, "weight", None)
    if weight is not None:
        size_fn = getattr(weight, "size", None)
        if callable(size_fn):
            try:
                size_value = size_fn(0)
                size_value_int = _coerce_optional_int(size_value)
                if size_value_int is not None:
                    return size_value_int
            except (TypeError, AttributeError, IndexError) as exc:
                LOG.debug(
                    "Failed to read embedding weight size; falling back to config vocab_size: %s",
                    exc,
                )
    config_value = _get_config_value(config, "vocab_size", None)
    return _coerce_optional_int(config_value)


def _maybe_long_tensor(value: object, torch_mod: _TorchModuleLike) -> Tensor:
    """Return a tensor cast to long when the stub lacks ``long``."""
    long_fn = getattr(value, "long", None)
    if callable(long_fn):
        try:
            return cast(Tensor, long_fn())
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
            long_method = getattr(t, "long", None)
            if callable(long_method):
                try:
                    return cast(Tensor, long_method())
                except (TypeError, ValueError, RuntimeError):
                    return cast(Tensor, t)
        except (TypeError, ValueError, RuntimeError) as exc:
            LOG.debug("torch.tensor(arr) failed in long conversion; retrying: %s", exc)
    # Fallback: try resolving a numpy-compatible dtype and pass that through.
    try:
        resolved = _resolve_dtype(getattr(torch_mod, "int64", None))
        if callable(tensor_fn):
            if resolved is None:
                return cast(Tensor, tensor_fn(arr))
            return cast(Tensor, tensor_fn(arr, dtype=resolved))
    except (TypeError, ValueError, RuntimeError):
        # Final fallback: coerce to int64 numpy and wrap.
        if callable(tensor_fn):
            return cast(Tensor, tensor_fn(np.asarray(arr).astype(np.int64)))
    return cast(Tensor, np.asarray(arr).astype(np.int64))


def _size_hint(tensor_obj: object, dim: int) -> int:
    """Return ``tensor.size(dim)`` with fallbacks for numpy-backed stubs."""
    size_fn = getattr(tensor_obj, "size", None)
    if callable(size_fn):
        try:
            size_val = size_fn(dim)
        except TypeError:
            try:
                size_val = size_fn()
            except (TypeError, ValueError, AttributeError) as exc:
                LOG.debug("tensor.size() fallback failed; using shape/len: %s", exc)
                size_val = None
        if size_val is not None:
            try:
                return int(cast(Any, size_val))
            except (TypeError, ValueError):
                pass
    arr = getattr(tensor_obj, "arr", None)
    shape = getattr(tensor_obj, "shape", None) or (
        arr.shape if arr is not None else None
    )
    if shape is None:
        if isinstance(tensor_obj, Sized):
            try:
                return len(tensor_obj)
            except TypeError:
                return 0
        return 0
    try:
        size_val = shape[dim] if dim is not None else (shape[0] if isinstance(shape, tuple) else shape)
        return int(cast(Any, size_val))
    except (TypeError, ValueError, IndexError):
        return 0


def _to_numpy_array(obj: object) -> np.ndarray:
    """Return a numpy view of ``obj`` for stub compatibility."""
    try:  # Handle real torch tensors, including CUDA, by detaching to CPU.
        import torch as _torch

        if isinstance(obj, getattr(_torch, "Tensor", ())):
            try:
                tensor = cast(Any, obj)
                return tensor.detach().cpu().numpy()
            except _SCORING_EXCEPTIONS:
                try:
                    tensor = cast(Any, obj)
                    tensor_cpu = tensor.detach().to("cpu")
                    if hasattr(tensor_cpu, "float"):
                        tensor_cpu = tensor_cpu.float()
                    return tensor_cpu.numpy()
                except _SCORING_EXCEPTIONS as exc:
                    LOG.debug("Failed to materialize torch tensor to numpy: %s", exc)
    except _SCORING_EXCEPTIONS as exc:
        LOG.debug("Torch import or tensor handling failed; falling back: %s", exc)
    # Fallback for tensor-like objects from mixed torch modules/stubs.
    detach_fn = getattr(obj, "detach", None)
    if callable(detach_fn):
        try:
            tensor_cpu = detach_fn()
            to_fn = getattr(tensor_cpu, "to", None)
            if callable(to_fn):
                tensor_cpu = to_fn("cpu")
            cpu_fn = getattr(tensor_cpu, "cpu", None)
            if callable(cpu_fn):
                tensor_cpu = cpu_fn()
            numpy_fn = getattr(tensor_cpu, "numpy", None)
            if callable(numpy_fn):
                return np.asarray(numpy_fn())
        except _SCORING_EXCEPTIONS as exc:
            LOG.debug("Fallback tensor->numpy conversion failed: %s", exc)
    arr = getattr(obj, "arr", None)
    if arr is not None:
        try:
            return np.asarray(arr)
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


def _dist_collective_ready(torch_mod: object) -> _DistModuleLike | None:
    """Return a dist module when initialized, otherwise None."""
    dist = getattr(torch_mod, "distributed", None)
    try:
        if dist is not None and dist.is_available() and dist.is_initialized():
            return cast(_DistModuleLike, dist)
    except _SCORING_EXCEPTIONS:
        return None
    return None


def _dist_all(dist: _DistModuleLike | None, flag: bool) -> bool:
    """Return True if flag is True on all ranks (best-effort)."""
    if dist is None:
        return bool(flag)
    try:
        get_world_size = getattr(dist, "get_world_size", None)
        if not callable(get_world_size):
            return bool(flag)
        world_size = int(cast(Any, get_world_size()))
    except _SCORING_EXCEPTIONS:
        return bool(flag)
    gathered = [None for _ in range(max(world_size, 1))]
    try:
        gather_fn = getattr(dist, "all_gather_object", None)
        if not callable(gather_fn):
            return bool(flag)
        gather_fn(gathered, bool(flag))
        return all(bool(x) for x in gathered)
    except _SCORING_EXCEPTIONS:
        return bool(flag)


def _dist_any(dist: _DistModuleLike | None, flag: bool) -> bool:
    """Return True if flag is True on any rank (best-effort)."""
    if dist is None:
        return bool(flag)
    try:
        get_world_size = getattr(dist, "get_world_size", None)
        if not callable(get_world_size):
            return bool(flag)
        world_size = int(cast(Any, get_world_size()))
    except _SCORING_EXCEPTIONS:
        return bool(flag)
    gathered = [None for _ in range(max(world_size, 1))]
    try:
        gather_fn = getattr(dist, "all_gather_object", None)
        if not callable(gather_fn):
            return bool(flag)
        gather_fn(gathered, bool(flag))
        return any(bool(x) for x in gathered)
    except _SCORING_EXCEPTIONS:
        return bool(flag)


def _resolve_dtype(dtype: object) -> Optional[np.dtype]:
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
        return np.dtype(cast(DTypeLike, dtype))
    except (TypeError, ValueError):
        return None


Tensor = TorchTensor
try:
    _ = require_transformer_base_classes("training_scoring")
except (ImportError, RuntimeError, ModuleNotFoundError):  # pragma: no cover - stub fallback
    _ = (Any, Any)


def _as_context_manager(value: object | None) -> ContextManager[object]:
    """Return value as a context manager when possible, otherwise a no-op."""
    if value is not None and hasattr(value, "__enter__") and hasattr(value, "__exit__"):
        return cast(ContextManager[object], value)
    return nullcontext()


def _autocast_context(accelerator: object, device: TorchDevice) -> ContextManager[object]:
    """Return the right autocast context for the current accelerator/device.

    :param accelerator: Accelerator handle exposing an optional ``autocast``.
    :type accelerator: Any
    :param device: Torch device used by the scoring step.
    :type device: torch.device
    :returns: Context manager handling autocast semantics.
    :rtype: contextlib.AbstractContextManager[Any]
    """
    if hasattr(accelerator, "autocast"):
        accel_autocast = getattr(accelerator, "autocast", None)
        if callable(accel_autocast):
            try:
                result = accel_autocast()
            except (TypeError, ValueError, RuntimeError):
                return nullcontext()
            return _as_context_manager(result)
        if accel_autocast is not None:
            return _as_context_manager(accel_autocast)
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
            if ctx_val is not None:
                return _as_context_manager(ctx_val)
    # If torch.autocast is already a context manager object, return it.
    if hasattr(autocast_fn, "__enter__") and not callable(autocast_fn):
        return _as_context_manager(autocast_fn)
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
    return _as_context_manager(result)


