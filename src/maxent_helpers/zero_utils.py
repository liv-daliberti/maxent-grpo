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

"""Utilities to safely integrate DeepSpeed ZeRO with optional dependencies."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, List, Optional, cast

try:  # Optional dependency when running under DeepSpeed ZeRO
    import torch
    from torch import nn
except (ImportError, ModuleNotFoundError):  # pragma: no cover - environment dependent
    def _cuda_is_available() -> bool:
        """Gracefully report CUDA as unavailable when PyTorch is missing."""
        return False

    def _cuda_empty_cache() -> None:
        """No-op placeholder that mimics torch.cuda.empty_cache."""
        return None

    torch = cast(
        Any,
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=_cuda_is_available, empty_cache=_cuda_empty_cache)
        ),
    )
    nn = cast(Any, SimpleNamespace(Module=object, Parameter=object))

try:
    from transformers import PreTrainedModel
except (ImportError, ModuleNotFoundError):  # pragma: no cover - environment dependent
    PreTrainedModel = Any  # type: ignore[assignment]

try:  # Optional dependency when running under DeepSpeed ZeRO
    from deepspeed import zero as ds_zero  # type: ignore
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus  # type: ignore
except (ImportError, ModuleNotFoundError):  # pragma: no cover - environment dependent
    ds_zero = None  # type: ignore
    ZeroParamStatus = None  # type: ignore


LOG = logging.getLogger(__name__)
_NO_SYNC_PATCH_ATTR = "_maxent_zero_no_sync_patched"
_NO_SYNC_WARN_ATTR = "_maxent_zero_no_sync_warned"


def _zero_stage(model: Optional[nn.Module]) -> int:
    """Return the DeepSpeed ZeRO stage for a model when available."""
    if model is None:
        return 0
    stage_attr = getattr(model, "zero_optimization_stage", 0)
    if callable(stage_attr):
        try:
            stage_attr = stage_attr()
        except TypeError:
            stage_attr = stage_attr(model)  # type: ignore[misc]
    try:
        return int(stage_attr or 0)
    except (TypeError, ValueError):
        return 0


def _zero_partitioning_gradients(model: Optional[nn.Module]) -> bool:
    """Return True when the model partitions gradients (ZeRO-3)."""
    if model is None:
        return False
    partition_fn = getattr(model, "zero_optimization_partition_gradients", None)
    if callable(partition_fn):
        try:
            return bool(partition_fn())
        except TypeError:
            return bool(partition_fn(model))  # type: ignore[misc]
    partition_attr = getattr(model, "partition_gradients", None)
    if callable(partition_attr):
        try:
            return bool(partition_attr())
        except TypeError:
            return bool(partition_attr(model))  # type: ignore[misc]
    if partition_attr is not None:
        return bool(partition_attr)
    return False


def _maybe_patch_zero_no_sync(model: Optional[nn.Module]) -> bool:
    """Patch DeepSpeedEngine.no_sync to a no-op when gradients are partitioned."""
    if model is None or getattr(model, _NO_SYNC_PATCH_ATTR, False):
        return False
    if _zero_stage(model) < 3:
        return False
    no_sync = getattr(model, "no_sync", None)
    if not callable(no_sync):
        return False
    if not _zero_partitioning_gradients(model):
        return False

    original_no_sync = no_sync

    @contextmanager
    def _patched_no_sync(*args: Any, **kwargs: Any):
        if _zero_partitioning_gradients(model):
            if not getattr(model, _NO_SYNC_WARN_ATTR, False):
                LOG.warning(
                    "DeepSpeed ZeRO-3 does not support no_sync; gradients will sync each step."
                )
                setattr(model, _NO_SYNC_WARN_ATTR, True)
            yield
            return
        with original_no_sync(*args, **kwargs):
            yield

    setattr(model, "no_sync", _patched_no_sync)
    setattr(model, _NO_SYNC_PATCH_ATTR, True)
    return True


def _embedding_weight_needing_gather(model: Optional[PreTrainedModel]) -> Optional[Any]:
    """Return the embedding weight tensor when ZeRO gathering is required."""
    if ds_zero is None or ZeroParamStatus is None or model is None:
        return None
    base_model = getattr(model, "module", model)
    embedder = getattr(base_model, "get_input_embeddings", lambda: None)()
    weight = getattr(embedder, "weight", None) if embedder is not None else None
    if weight is None:
        return None
    if getattr(weight, "ndim", 2) == 2:
        return None
    status = getattr(weight, "ds_status", None)
    if status is not None and status != ZeroParamStatus.NOT_AVAILABLE:
        return None
    return weight


@contextmanager
def _maybe_zero_gather_embedding(model: Optional[PreTrainedModel]):
    """Gather ZeRO-sharded embedding weights before a forward pass."""
    weight = _embedding_weight_needing_gather(model)
    if weight is None:
        yield
        return
    with ds_zero.GatheredParameters([weight], modifier_rank=None):  # type: ignore[union-attr]
        yield


def _zero_param_list(model: Optional[nn.Module]) -> List[nn.Parameter]:
    """Return a parameter list for ZeRO-gather contexts, unwrapping DeepSpeed engines."""
    if model is None:
        return []
    base_model = getattr(model, "module", model)
    if not hasattr(base_model, "parameters"):
        return []
    try:
        return list(base_model.parameters())
    except TypeError:
        return []


@contextmanager
def _maybe_zero_gather_params(model: Optional[nn.Module], enabled: bool):
    """Context manager that gathers ZeRO-partitioned params only when needed."""
    if not enabled or ds_zero is None or model is None:
        yield
        return
    params = _zero_param_list(model)
    zero_stage = _zero_stage(model)
    if zero_stage > 0:
        gather_params = params
    else:
        gather_params = [p for p in params if hasattr(p, "ds_id")]
    if not gather_params:
        yield
        return
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    with ds_zero.GatheredParameters(gather_params, modifier_rank=0):  # type: ignore[union-attr]
        yield


__all__ = [
    "_maybe_zero_gather_embedding",
    "_maybe_zero_gather_params",
    "_maybe_patch_zero_no_sync",
]
