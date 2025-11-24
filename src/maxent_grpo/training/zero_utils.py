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
import importlib.machinery
import importlib.util
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Callable, ContextManager, List, Optional, cast

from .run_helpers import require_deepspeed

# Rehydrate the module spec when running under stubbed environments

if (
    __spec__ is None or getattr(__spec__, "loader", None) is None
):  # pragma: no cover - defensive reload support
    __spec__ = importlib.util.spec_from_loader(
        __name__, importlib.machinery.SourceFileLoader(__name__, __file__)
    )
sys.modules.setdefault(__name__, sys.modules.get(__name__, None))
if __spec__ is not None:
    sys.modules.setdefault(__spec__.name, sys.modules[__name__])

try:  # Optional dependency when running under DeepSpeed ZeRO
    import torch
    from torch import nn
except (ImportError, ModuleNotFoundError, RuntimeError):  # pragma: no cover - environment dependent
    torch = None
    nn = None


def _ensure_cuda_fallback() -> Any:
    """Return a cuda namespace exposing ``is_available`` and ``empty_cache``.

    :returns: Namespace exposing ``is_available`` and ``empty_cache`` placeholders.
    :rtype: types.SimpleNamespace
    """

    def _cuda_is_available() -> bool:
        return False

    def _cuda_empty_cache() -> None:
        return None

    return SimpleNamespace(
        is_available=_cuda_is_available, empty_cache=_cuda_empty_cache
    )


if torch is None:
    torch = cast(Any, SimpleNamespace(cuda=_ensure_cuda_fallback()))
    nn = cast(Any, SimpleNamespace(Module=object, Parameter=object))
else:
    if not hasattr(torch, "cuda") or not hasattr(torch.cuda, "is_available"):
        setattr(torch, "cuda", _ensure_cuda_fallback())

try:
    from transformers import PreTrainedModel
except (ImportError, ModuleNotFoundError):  # pragma: no cover - environment dependent
    PreTrainedModel = Any

ds_zero = None
ZeroParamStatus = None
_DEEPSPEED_READY: Optional[bool] = None


def _ensure_deepspeed_ready() -> bool:
    """Best-effort initialization of DeepSpeed helpers when installed.

    :returns: ``True`` when DeepSpeed zero helpers are available; ``False`` otherwise.
    :rtype: bool
    """
    global ds_zero, ZeroParamStatus, _DEEPSPEED_READY
    if _DEEPSPEED_READY is True:
        return True
    if _DEEPSPEED_READY is False:
        return False
    try:
        ds_module = require_deepspeed("ZeRO utilities")
        partition_module = require_deepspeed(
            "ZeRO utilities",
            "deepspeed.runtime.zero.partition_parameters",
        )
    except RuntimeError:
        _DEEPSPEED_READY = False
        ds_zero = None
        ZeroParamStatus = None
        return False
    ds_zero = getattr(ds_module, "zero", None)
    ZeroParamStatus = getattr(partition_module, "ZeroParamStatus", None)
    _DEEPSPEED_READY = bool(ds_zero is not None and ZeroParamStatus is not None)
    return bool(_DEEPSPEED_READY)


LOG = logging.getLogger(__name__)
_NO_SYNC_PATCH_ATTR = "_maxent_zero_no_sync_patched"
_NO_SYNC_WARN_ATTR = "_maxent_zero_no_sync_warned"


def _zero_stage(model: Optional[nn.Module]) -> int:
    """Return the DeepSpeed ZeRO stage for a model when available.

    :param model: Model or engine exposing ``zero_optimization_stage``.
    :type model: torch.nn.Module | None
    :returns: ZeRO stage (0 when unavailable).
    :rtype: int
    """
    if model is None:
        return 0
    stage_attr = getattr(model, "zero_optimization_stage", 0)
    if callable(stage_attr):
        try:
            stage_attr = stage_attr()
        except TypeError:
            stage_attr = stage_attr(model)
    try:
        return int(stage_attr or 0)
    except (TypeError, ValueError):
        return 0


def _zero_partitioning_gradients(model: Optional[nn.Module]) -> bool:
    """Return whether the model partitions gradients (ZeRO-3).

    :param model: Model or engine potentially partitioning gradients.
    :type model: torch.nn.Module | None
    :returns: ``True`` when gradients are partitioned, else ``False``.
    :rtype: bool
    """
    if model is None:
        return False
    partition_fn = getattr(model, "zero_optimization_partition_gradients", None)
    if callable(partition_fn):
        try:
            return bool(partition_fn())
        except TypeError:
            return bool(partition_fn(model))
    partition_attr = getattr(model, "partition_gradients", None)
    if callable(partition_attr):
        try:
            return bool(partition_attr())
        except TypeError:
            return bool(partition_attr(model))
    if partition_attr is not None:
        return bool(partition_attr)
    return False


def _maybe_patch_zero_no_sync(model: Optional[nn.Module]) -> bool:
    """Patch DeepSpeedEngine.no_sync to a no-op when gradients are partitioned.

    :param model: DeepSpeed engine or wrapped model.
    :type model: torch.nn.Module | None
    :returns: ``True`` if the patch was applied; ``False`` otherwise.
    :rtype: bool
    """
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
        """Shim that emits a warning and bypasses ``no_sync`` under ZeRO-3."""
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
    """Return the embedding weight tensor when ZeRO gathering is required.

    :param model: Model potentially wrapping ZeRO-managed embeddings.
    :type model: transformers.PreTrainedModel | None
    :returns: Embedding weight requiring gather, or ``None``.
    :rtype: torch.Tensor | None
    """
    if not _ensure_deepspeed_ready() or model is None:
        return None
    base_model = getattr(model, "module", model)
    embedder = getattr(base_model, "get_input_embeddings", lambda: None)()
    weight = getattr(embedder, "weight", None) if embedder is not None else None
    if weight is None:
        return None
    if getattr(weight, "ndim", 2) == 2:
        return None
    status = getattr(weight, "ds_status", None)
    if (
        status is not None
        and ZeroParamStatus is not None
        and status != ZeroParamStatus.NOT_AVAILABLE
    ):
        return None
    return weight


@contextmanager
def _maybe_zero_gather_embedding(model: Optional[PreTrainedModel]):
    """Gather ZeRO-sharded embedding weights before a forward pass.

    :param model: Model potentially wrapping ZeRO-managed embeddings.
    :type model: transformers.PreTrainedModel | None
    :returns: Context manager that gathers the embedding tensor when needed.
    :rtype: contextlib.AbstractContextManager[None]
    """
    if not _ensure_deepspeed_ready() or ds_zero is None:
        yield
        return
    gather_cls = getattr(ds_zero, "GatheredParameters", None)
    if not callable(gather_cls):
        yield
        return
    gather_ctx = cast(Callable[..., ContextManager[Any]], gather_cls)
    weight = _embedding_weight_needing_gather(model)
    if weight is None:
        yield
        return
    with gather_ctx([weight], modifier_rank=None):
        yield


def _zero_param_list(model: Optional[nn.Module]) -> List[nn.Parameter]:
    """Return a parameter list for ZeRO-gather contexts, unwrapping engines.

    :param model: Module potentially wrapped by DeepSpeed.
    :type model: torch.nn.Module | None
    :returns: Parameters to feed into ``GatheredParameters`` calls.
    :rtype: list[torch.nn.Parameter]
    """
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
    """Gather ZeRO-partitioned params only when needed.

    :param model: Module whose parameters might be partitioned.
    :type model: torch.nn.Module | None
    :param enabled: Whether gathering should be attempted.
    :type enabled: bool
    :returns: Context manager yielding with parameters gathered when necessary.
    :rtype: contextlib.AbstractContextManager[None]
    """
    if not enabled or model is None or not _ensure_deepspeed_ready() or ds_zero is None:
        yield
        return
    gather_cls = getattr(ds_zero, "GatheredParameters", None)
    if not callable(gather_cls):
        yield
        return
    gather_ctx = cast(Callable[..., ContextManager[Any]], gather_cls)
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
    with gather_ctx(gather_params, modifier_rank=0):
        yield


__all__ = [
    "_maybe_zero_gather_embedding",
    "_maybe_zero_gather_params",
    "_maybe_patch_zero_no_sync",
]
