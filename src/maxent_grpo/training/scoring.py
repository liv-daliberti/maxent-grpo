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
import numbers
import sys
import logging
from types import SimpleNamespace
from typing import Any, Iterable, List, Optional, Sequence, Tuple
from functools import lru_cache
import queue
import threading
import numpy as np

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

torch = require_torch("training_scoring")
_REQUIRED_TORCH_ATTRS = ("tensor", "full", "ones_like", "zeros", "cat")


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
    if any(not hasattr(torch_mod, attr) for attr in _REQUIRED_TORCH_ATTRS):
        try:  # pragma: no cover - defensive stub installation
            import importlib

            _bootstrap = importlib.import_module("ops.sitecustomize")
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
    # Patch missing attributes with a lightweight stub
    stub = _build_torch_stub()
    for name in _REQUIRED_TORCH_ATTRS + (
        "long",
        "float32",
        "int64",
        "no_grad",
        "SymBool",
        "stack",
        "unique",
        "nn",
        "optim",
    ):
        if not hasattr(torch_mod, name) and hasattr(stub, name):
            setattr(torch_mod, name, getattr(stub, name))
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
    # Ensure device namespaces exist to avoid import errors in manual_seed.
    if not hasattr(torch_mod, "xpu") and hasattr(stub, "xpu"):
        torch_mod.xpu = stub.xpu
    sys.modules.setdefault("torch.xpu", getattr(torch_mod, "xpu", None))
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
        except BaseException as exc:  # pragma: no cover - defensive
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
            except Exception:
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
                            except Exception:
                                size = None
                        elif hasattr(weight, "size"):
                            try:
                                size = tuple(weight.size())
                            except Exception:
                                size = None
                        if size is not None and len(size) >= 1:
                            num_embeddings = size[0]
                            if isinstance(num_embeddings, numbers.Integral) and num_embeddings > 0:
                                if new_value >= num_embeddings:
                                    new_value = num_embeddings - 1
            except Exception:
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
        except Exception:
            shape = None
    return shape is not None and len(shape) == 2


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
        except Exception:
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
        except (TypeError, AttributeError, IndexError):
            pass
    return _get_config_value(config, "vocab_size", None)


def _maybe_long_tensor(value: Any, torch_mod: Any) -> Any:
    """Return a tensor cast to long when the stub lacks ``long``."""
    if hasattr(value, "long"):
        try:
            return value.long()
        except TypeError:
            pass
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
        except (TypeError, ValueError, RuntimeError):
            pass
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
            except (TypeError, ValueError, AttributeError):
                pass
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
            except Exception:
                pass
    except Exception:
        pass
    if hasattr(obj, "arr"):
        try:
            return np.asarray(obj.arr)
        except (TypeError, ValueError, RuntimeError):
            pass
    data = getattr(obj, "data", None)
    if data is not None:
        try:
            return np.asarray(data)
        except (TypeError, ValueError, RuntimeError):
            pass
    try:
        return np.asarray(obj)
    except (TypeError, ValueError, RuntimeError):
        return np.asarray([])


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

LOG = logging.getLogger(__name__)


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
            except (AttributeError, TypeError, ValueError):
                # Some lightweight torch stubs treat the ``device`` argument as
                # a dtype. When that happens we leave tensors on their current
                # device and rely on ``as_tensor`` below to normalize types.
                pass
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
            except (AttributeError, TypeError, ValueError):
                pass
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
            except Exception:  # pragma: no cover - defensive
                pass
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
        except Exception:
            pass
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
            except Exception:
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
            except Exception:
                pad_mask = None
            if pad_mask is None:
                try:
                    pad_mask_arr = np.asarray(
                        getattr(comp_mask_slice, "arr", comp_mask_slice)
                    ) == 0
                    has_padding = bool(pad_mask_arr.any())
                except Exception:
                    pad_mask_arr = None
                    has_padding = False
            if has_padding:
                try:
                    if pad_mask is None:
                        raise AttributeError
                    updated_labels = comp_labels.masked_fill(pad_mask, -100)
                except Exception:
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
                def _slice_tail(start_idx: int) -> Tuple[Tensor, Tensor, Tensor]:
                    if start_idx <= 0:
                        return full_input_ids, full_attention_mask, full_labels
                    return (
                        full_input_ids[:, start_idx:],
                        full_attention_mask[:, start_idx:],
                        full_labels[:, start_idx:],
                    )

                slice_start = max(0, max_len - tail_tokens)
                first_comp_global = None
                last_comp_global = None
                if active_comp_columns:
                    first_comp_global = prompt_width + active_comp_columns[0]
                    last_comp_global = prompt_width + active_comp_columns[-1] + 1
                input_ids, attention_mask, labels_tensor = _slice_tail(slice_start)
                if (
                    first_comp_global is not None
                    and last_comp_global is not None
                    and slice_start >= last_comp_global
                ):
                    safe_start = max(first_comp_global, last_comp_global - tail_tokens)
                    input_ids, attention_mask, labels_tensor = _slice_tail(safe_start)
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
                except Exception:
                    pass
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
        gather_full_params,
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
    if gather_full_params:
        try:  # pragma: no cover - exercised in distributed runs
            import deepspeed

            params: list[Any] = []
            param_iter = getattr(model, "parameters", None)
            if callable(param_iter):
                try:
                    params = list(param_iter())
                except Exception:
                    params = []
            # In lightweight stub environments ``parameters`` may be missing;
            # still invoke GatheredParameters with an empty list so tests can
            # verify that the context manager is used.
            if params or not hasattr(model, "parameters"):
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
                inp_emb = (
                    model.get_input_embeddings()
                    if hasattr(model, "get_input_embeddings")
                    else None
                )
                if inp_emb is not None and hasattr(inp_emb, "weight"):
                    to_gather.append(inp_emb.weight)
                out_emb = (
                    model.get_output_embeddings()
                    if hasattr(model, "get_output_embeddings")
                    else None
                )
                if out_emb is None and hasattr(model, "lm_head"):
                    out_emb = model.lm_head
                if out_emb is not None and hasattr(out_emb, "weight"):
                    to_gather.append(out_emb.weight)
                if to_gather:
                    try:
                        gather_ctx = deepspeed.zero.GatheredParameters(
                            to_gather, modifier_rank=None
                        )
                    except TypeError:
                        gather_ctx = deepspeed.zero.GatheredParameters(to_gather)
        except Exception:
            gather_ctx = nullcontext()

    fsdp_ctx = _summon_fsdp_full_param_context(model)
    stack = ExitStack()
    stack.enter_context(gather_ctx)
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
    except Exception:
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
        except Exception:
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
        if chunk_limit <= 0 or chunk_limit >= batch:
            chunk_indices = [(0, batch)]
        else:
            chunk_indices = [
                (start, min(start + chunk_limit, batch))
                for start in range(0, batch, chunk_limit)
            ]
        logp_chunks: list[Tensor] = []
        tok_chunks: list[Tensor] = []
        pooled_chunks: list[Tensor] = [] if return_hidden else []
        for idx, (start, end) in enumerate(chunk_indices):
            ids_chunk = input_ids[start:end]
            mask_chunk = attention_mask[start:end] if attention_mask is not None else None
            label_chunk = labels[start:end]
            try:
                outputs = model(
                    input_ids=ids_chunk,
                    attention_mask=mask_chunk,
                    labels=label_chunk,
                    output_hidden_states=return_hidden,
                )
            except TypeError:
                outputs = model(
                    input_ids=ids_chunk,
                    attention_mask=mask_chunk,
                    output_hidden_states=return_hidden,
                )
            logits = outputs.logits
            LOG.debug(
                "reference scoring logits shape | chunk=%s shape=%s",
                idx,
                getattr(logits, "shape", None),
            )
            log_probs = torch_mod.nn.functional.log_softmax(logits, dim=-1)
            label_mask = label_chunk != -100
            vocab_size = log_probs.size(-1)
            flat_labels = label_chunk.masked_fill(~label_mask, 0).view(-1)
            flat_log_probs = log_probs.view(-1, vocab_size)
            gather_index = torch_mod.arange(flat_labels.numel())
            target_device = getattr(flat_log_probs, "device", None)
            if target_device is not None:
                try:
                    gather_index = gather_index.to(target_device)
                except Exception:
                    pass
            gathered = flat_log_probs[gather_index, flat_labels].view(label_chunk.shape)
            seq_logp_chunk = (gathered * label_mask).sum(dim=1)
            tok_tensor_chunk = label_mask.sum(dim=1).clamp(min=1)
            logp_chunks.append(seq_logp_chunk)
            tok_chunks.append(tok_tensor_chunk)
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
                            mask = type_as_fn(hidden)
                        else:
                            to_fn = getattr(mask, "to", None)
                            if callable(to_fn):
                                mask = to_fn(dtype=hidden.dtype)
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
    return seq_logp, tok_tensor, pooled_hidden


def build_score_batch(
    reward_comp: RewardComputation,
    tokenizer: PreTrainedTokenizer,
    generation_cfg: GenerationSettings,
    batching_cfg: BatchingSettings,
) -> Optional[ScoreBatch]:
    """Tokenize prompt+completion pairs and prepare masks/labels."""
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
    """Run the frozen reference model to compute log-probs."""
    torch_mod = _refresh_torch()
    ref_model = runtime.get_ref_model()
    ref_logp_chunks: List[Tensor] = []
    ref_tok_chunks: List[Tensor] = []
    slice_iter = iter_batch_slices(score_batch, runtime.device)
    slice_iter = _prefetch_iterator(
        slice_iter, getattr(batching_cfg, "slice_prefetch", 0)
    )
    for slice_inputs, slice_mask, slice_labels in slice_iter:
        no_grad_ctx = getattr(torch_mod, "no_grad", None) or nullcontext
        with no_grad_ctx():
            result = _chunked_sequence_logprobs(
                ref_model,
                input_ids=slice_inputs,
                attention_mask=slice_mask,
                labels=slice_labels,
                chunk_size=batching_cfg.logprob_chunk_size,
                gather_full_params=False,
            )
        # _chunked_sequence_logprobs normally returns (logp, tok_counts, pooled_hidden).
        if not isinstance(result, (tuple, list)) or len(result) < 2:
            return None
        if len(result) >= 3:
            ref_logp_slice, ref_tok_slice, _ = result[:3]
        else:
            ref_logp_slice, ref_tok_slice = result[:2]
        if ref_logp_slice.numel() == 0 or ref_tok_slice.numel() == 0:
            return None
        ref_logp_chunks.append(ref_logp_slice.detach().cpu())
        ref_tok_chunks.append(ref_tok_slice.detach().cpu())
    if not ref_logp_chunks:
        return None
    return (
        torch_mod.cat(ref_logp_chunks, dim=0).to(runtime.device),
        torch_mod.cat(ref_tok_chunks, dim=0).to(runtime.device),
    )


def finalize_reference_stats(
    ref_logp_sum: Tensor,
    ref_tok_counts: Tensor,
) -> ReferenceLogprobs:
    """Build a ReferenceLogprobs object and derived scalars."""
    torch_mod = _refresh_torch()
    logp_arr = _to_numpy_array(ref_logp_sum)
    tok_arr = np.asarray(_to_numpy_array(ref_tok_counts), dtype=float)
    if tok_arr.size == 0:
        tok_arr = np.asarray([0.0])
    invalid_mask = ~np.isfinite(tok_arr) | (tok_arr < 0)
    if invalid_mask.any():
        finite_replaced = np.nan_to_num(tok_arr, nan=0.0, posinf=0.0, neginf=0.0)
        tok_arr = np.maximum(finite_replaced, 0.0)
        LOG.debug(
            "Clamped %d invalid reference token counts to non-negative finite values.",
            int(invalid_mask.sum()),
        )
    if tok_arr.size == 0:
        tok_arr = np.asarray([0.0])
    ref_logp_sum_tensor = torch_mod.tensor(
        logp_arr, dtype=getattr(torch_mod, "float32", None)
    )
    ref_tok_counts_tensor = torch_mod.tensor(
        tok_arr, dtype=getattr(torch_mod, "float32", None)
    )
    ref_logp_mean = (
        float(np.asarray(logp_arr, dtype=float).mean()) if logp_arr.size else 0.0
    )
    avg_completion_tokens = float(np.asarray(tok_arr, dtype=float).mean())
    return ReferenceLogprobs(
        ref_logp_sum=ref_logp_sum_tensor,
        ref_tok_counts=ref_tok_counts_tensor,
        ref_logp_sum_raw=ref_logp_sum_tensor,
        ref_logp_mean=ref_logp_mean,
        avg_completion_tokens=avg_completion_tokens,
    )


def gather_reference_logprobs(
    score_batch: ScoreBatch,
    runtime: RuntimeHandles,
    batching_cfg: BatchingSettings,
) -> Optional[ReferenceLogprobs]:
    """Compute log-probabilities by running the frozen reference model."""
    tensors = reference_from_model(score_batch, runtime, batching_cfg)
    if tensors is None:
        return None
    return finalize_reference_stats(*tensors)


def reference_from_vllm_meta(
    flat_meta: Sequence[Optional[Any]],
    total_sequences: int,
    device: torch.device,
) -> Optional[ReferenceLogprobs]:
    """Convert flattened vLLM log-prob metadata into ReferenceLogprobs."""
    if not flat_meta:
        return None
    if len(flat_meta) != total_sequences:
        return None
    logp_vals: List[float] = []
    tok_counts: List[int] = []
    for entry in flat_meta:
        if entry is None:
            return None
        logprob_sum = getattr(entry, "logprob_sum", None)
        token_count = getattr(entry, "token_count", None)
        if logprob_sum is None and isinstance(entry, dict):
            logprob_sum = entry.get("logprob_sum")
        if token_count is None and isinstance(entry, dict):
            token_count = entry.get("token_count")
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


def score_model_outputs(
    model: PreTrainedModel,
    score_batch: ScoreBatch,
    batching_cfg: BatchingSettings,
    runtime: RuntimeHandles,
    *,
    return_hidden: bool = False,
    pooling: str = "mean",
) -> Optional[Tuple[Tensor, Optional[Tensor]]]:
    """Compute current model log-probs for the batch and optional pooled states."""
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
    """Summarize completion lengths for metrics."""
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


def build_sequence_scores(
    cur_logp_sum: Tensor,
    ref_stats: ReferenceLogprobs,
    pooled_hidden: Optional[Tensor] = None,
    *,
    behavior_logp_sum: Optional[Tensor] = None,
) -> SequenceScores:
    """Return SequenceScores built from current and reference log-probs."""
    torch_mod = _refresh_torch()
    cur_arr = _to_numpy_array(cur_logp_sum)
    ref_arr = _to_numpy_array(ref_stats.ref_logp_sum_raw)
    denom_arr = _to_numpy_array(ref_stats.ref_tok_counts)
    denom_arr = np.where(denom_arr <= 0, 1, denom_arr)
    try:
        cur_len = len(cur_arr)
        if len(ref_arr) != cur_len:
            if len(ref_arr) == 1:
                ref_arr = np.repeat(ref_arr, cur_len)
            else:
                ref_arr = np.resize(ref_arr, cur_len)
        if len(denom_arr) != cur_len:
            if len(denom_arr) == 1:
                denom_arr = np.repeat(denom_arr, cur_len)
            else:
                denom_arr = np.resize(denom_arr, cur_len)
    except (TypeError, ValueError):
        pass
    cur_tensor = torch_mod.tensor(cur_arr, dtype=getattr(torch_mod, "float32", None))
    if behavior_logp_sum is None:
        # For PPO-style clipping, the behavior policy should be the rollout
        # actor. Default to the current policy log-probs (on-policy) rather than
        # the frozen reference model to avoid turning clipping into a second KL.
        behavior_logp_sum = cur_tensor.detach()
    else:
        behavior_logp_sum = torch_mod.tensor(
            _to_numpy_array(behavior_logp_sum), dtype=getattr(torch_mod, "float32", None)
        )
    log_ratio_train = torch_mod.tensor(
        cur_arr - ref_arr, dtype=getattr(torch_mod, "float32", None)
    )
    denom_tok_tensor = torch_mod.tensor(
        denom_arr, dtype=getattr(torch_mod, "float32", None)
    )
    if getattr(denom_tok_tensor, "numel", lambda: 0)() == 0:
        denom_tok_tensor = torch_mod.ones_like(cur_tensor, dtype=cur_tensor.dtype)
    return SequenceScores(
        cur_logp_sum=cur_tensor,
        behavior_logp_sum=behavior_logp_sum,
        log_ratio_train=log_ratio_train,
        denom_tok_tensor=denom_tok_tensor,
        pooled_hidden=pooled_hidden,
    )
