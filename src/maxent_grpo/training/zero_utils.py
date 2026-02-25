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

import importlib.machinery
import importlib.util
import logging
import numbers
import sys
from contextlib import contextmanager, nullcontext
import importlib
from threading import RLock
from types import SimpleNamespace
from typing import (
    Any,
    ContextManager,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    TYPE_CHECKING,
    TypeAlias,
    cast,
)

from maxent_grpo.training.runtime import require_deepspeed

# Rehydrate the module spec when running under stubbed environments
_current_spec = getattr(sys.modules.get(__name__), "__spec__", None)
if (
    _current_spec is None or getattr(_current_spec, "loader", None) is None
):  # pragma: no cover - defensive reload support
    _current_spec = importlib.util.spec_from_loader(
        __name__, importlib.machinery.SourceFileLoader(__name__, __file__)
    )
    setattr(sys.modules[__name__], "__spec__", _current_spec)
_self_module = sys.modules.get(__name__)
if _self_module is not None:
    sys.modules.setdefault(__name__, _self_module)
    if _current_spec is not None:
        sys.modules.setdefault(_current_spec.name, _self_module)

try:  # Optional dependency when running under DeepSpeed ZeRO
    import torch
except (
    ImportError,
    ModuleNotFoundError,
    RuntimeError,
):  # pragma: no cover - environment dependent
    torch = None


def _ensure_cuda_fallback() -> SimpleNamespace:
    """Return a cuda namespace exposing ``is_available`` and ``empty_cache``.

    :returns: Namespace exposing ``is_available`` and ``empty_cache`` placeholders.
    :rtype: types.SimpleNamespace
    """

    def _cuda_is_available() -> bool:
        return False

    def _cuda_empty_cache() -> None:
        return None

    return SimpleNamespace(
        is_available=_cuda_is_available,
        empty_cache=_cuda_empty_cache,
        current_allocated_memory=lambda: 0,
        current_reserved_memory=lambda: 0,
        memory_allocated=lambda *_a, **_k: 0,
        memory_reserved=lambda *_a, **_k: 0,
        max_memory_allocated=lambda *_a, **_k: 0,
        max_memory_reserved=lambda *_a, **_k: 0,
        memory_stats=lambda *_a, **_k: {},
    )


if torch is None:
    torch = cast(object, SimpleNamespace(cuda=_ensure_cuda_fallback()))
else:
    if not hasattr(torch, "cuda") or not hasattr(torch.cuda, "is_available"):
        setattr(torch, "cuda", _ensure_cuda_fallback())

if TYPE_CHECKING:
    from transformers.modeling_utils import (
        PreTrainedModel as _PreTrainedModel,
    )
    from torch import nn as torch_nn

    PreTrainedModel: TypeAlias = _PreTrainedModel
    TorchModule = torch_nn.Module
    TorchParameter = torch_nn.Parameter
else:  # pragma: no cover - runtime uses optional stubs
    PreTrainedModel: TypeAlias = object
    TorchModule = object
    TorchParameter = object

ds_zero = None
ZeroParamStatus = None
_DEEPSPEED_READY: Optional[bool] = None
_DEEPSPEED_ENGINE_CLS: Optional[type] = None
_ZERO_GATHER_LOCK = RLock()
_ZERO_GATHER_ACTIVE_PARAM_IDS: set[int] = set()


def _ensure_deepspeed_ready() -> bool:
    """Best-effort initialization of DeepSpeed helpers when installed.

    :returns: ``True`` when DeepSpeed zero helpers are available; ``False`` otherwise.
    :rtype: bool
    """
    module = sys.modules[__name__]
    ready_state = getattr(module, "_DEEPSPEED_READY", None)
    if ready_state is True:
        return True
    if ready_state is False:
        return False
    try:
        ds_module = require_deepspeed("ZeRO utilities")
        partition_module = require_deepspeed(
            "ZeRO utilities",
            "deepspeed.runtime.zero.partition_parameters",
        )
    except RuntimeError:
        setattr(module, "_DEEPSPEED_READY", False)
        setattr(module, "ds_zero", None)
        setattr(module, "ZeroParamStatus", None)
        return False
    ds_zero_mod = getattr(ds_module, "zero", None)
    zero_status = getattr(partition_module, "ZeroParamStatus", None)
    setattr(module, "ds_zero", ds_zero_mod)
    setattr(module, "ZeroParamStatus", zero_status)
    is_ready = bool(ds_zero_mod is not None and zero_status is not None)
    setattr(module, "_DEEPSPEED_READY", is_ready)
    return is_ready


def _deepspeed_engine_cls() -> Optional[type]:
    """Return the DeepSpeedEngine class when available."""
    module = sys.modules[__name__]
    cached = getattr(module, "_DEEPSPEED_ENGINE_CLS", None)
    if cached is not None:
        return cached
    try:
        from deepspeed.runtime.engine import DeepSpeedEngine
    except (ImportError, ModuleNotFoundError):
        setattr(module, "_DEEPSPEED_ENGINE_CLS", None)
        return None
    setattr(module, "_DEEPSPEED_ENGINE_CLS", DeepSpeedEngine)
    return DeepSpeedEngine


def _is_deepspeed_engine(model: Optional[object]) -> bool:
    """Return True when the provided model is a DeepSpeed engine."""
    if model is None:
        return False
    engine_cls = _deepspeed_engine_cls()
    if engine_cls is not None and isinstance(model, engine_cls):
        return True
    # Heuristic fallback: DeepSpeed engine exposes zero_optimization_stage()
    zero_stage = getattr(model, "zero_optimization_stage", None)
    if callable(zero_stage):
        return True
    return False


LOG = logging.getLogger(__name__)
_NO_SYNC_PATCH_ATTR = "_maxent_zero_no_sync_patched"
_NO_SYNC_WARN_ATTR = "_maxent_zero_no_sync_warned"


class GatherCallable(Protocol):
    """Callable signature exposed by DeepSpeed GatheredParameters."""

    def __call__(
        self, params: Sequence[object], *args: object, **kwargs: object
    ) -> ContextManager[None]: ...


def _zero_stage(model: Optional[TorchModule]) -> int:
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
    if isinstance(stage_attr, numbers.Real):
        try:
            return int(float(stage_attr))
        except (TypeError, ValueError):
            return 0
    return 0


def _zero_partitioning_gradients(model: Optional[TorchModule]) -> bool:
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


def _zero_status_name(param: object) -> Optional[str]:
    """Best-effort extraction of the DeepSpeed ZeRO status name."""
    status = getattr(param, "ds_status", None)
    if status is None:
        return None
    available = getattr(ZeroParamStatus, "AVAILABLE", None)
    if available is not None and status == available:
        return "AVAILABLE"
    status_name = getattr(status, "name", None)
    if isinstance(status_name, str) and status_name:
        return status_name.upper()
    if isinstance(status, str) and status:
        return status.upper()
    try:
        status_text = str(status)
    except (TypeError, ValueError, RuntimeError):
        return None
    if not status_text:
        return None
    if "." in status_text:
        status_text = status_text.rsplit(".", 1)[-1]
    status_text = status_text.strip().upper()
    return status_text or None


def _zero_param_ready_without_gather(param: object) -> bool:
    """Return ``True`` when a ZeRO parameter is already materialized."""
    if _zero_status_name(param) == "AVAILABLE":
        return True
    active = getattr(param, "ds_active_sub_modules", None)
    try:
        return bool(active)
    except (TypeError, ValueError, RuntimeError):
        return False


@contextmanager
def _reserve_zero_gather_params(
    params: Sequence[TorchParameter],
) -> Iterator[List[TorchParameter]]:
    """Reserve parameter ids for a ZeRO gather region.

    Prevents nested ``GatheredParameters`` calls from re-gathering the same
    parameter while another gather context still owns it.

    :param params: Candidate parameters for a gather region.
    :type params: Sequence[torch.nn.Parameter]
    :returns: Reserved subset that is safe to gather in this region.
    :rtype: Iterator[list[torch.nn.Parameter]]
    """

    reserved: List[TorchParameter] = []
    reserved_ids: list[int] = []
    with _ZERO_GATHER_LOCK:
        for param in params:
            param_id = id(param)
            if param_id in _ZERO_GATHER_ACTIVE_PARAM_IDS:
                continue
            _ZERO_GATHER_ACTIVE_PARAM_IDS.add(param_id)
            reserved.append(param)
            reserved_ids.append(param_id)
    try:
        yield reserved
    finally:
        if not reserved_ids:
            return
        with _ZERO_GATHER_LOCK:
            for param_id in reserved_ids:
                _ZERO_GATHER_ACTIVE_PARAM_IDS.discard(param_id)


def _maybe_patch_zero_no_sync(model: Optional[TorchModule]) -> bool:
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
    def _patched_no_sync(*args: object, **kwargs: object) -> Iterator[None]:
        """Shim that emits a warning and bypasses ``no_sync`` under ZeRO-3."""
        if _zero_partitioning_gradients(model):
            if not getattr(model, _NO_SYNC_WARN_ATTR, False):
                LOG.warning(
                    "DeepSpeed ZeRO-3 does not support no_sync; gradients will sync each step."
                )
                setattr(model, _NO_SYNC_WARN_ATTR, True)
            yield
            return
        ctx = original_no_sync(*args, **kwargs)
        with cast(ContextManager[None], ctx):
            yield

    setattr(model, "no_sync", _patched_no_sync)
    setattr(model, _NO_SYNC_PATCH_ATTR, True)
    return True


def _embedding_weight_needing_gather(
    model: Optional[PreTrainedModel],
) -> Optional[TorchParameter]:
    """Return the embedding weight tensor when ZeRO gathering is required.

    :param model: Model potentially wrapping ZeRO-managed embeddings.
    :type model: PreTrainedModel | None
    :returns: Embedding weight requiring gather, or ``None``.
    :rtype: torch.Tensor | None
    """
    if not _ensure_deepspeed_ready() or model is None:
        return None
    if _is_deepspeed_engine(model):
        return None
    base_model = getattr(model, "module", model)
    embedder = getattr(base_model, "get_input_embeddings", lambda: None)()
    weight = getattr(embedder, "weight", None) if embedder is not None else None
    if weight is None:
        return None
    if _zero_param_ready_without_gather(weight):
        return None
    if _zero_status_name(weight) is not None:
        # Any explicit ZeRO status other than AVAILABLE indicates gather need.
        return weight
    if getattr(weight, "ndim", 2) == 2:
        return None
    return weight


def _embedding_weights_needing_gather(
    model: Optional[PreTrainedModel],
) -> List[TorchParameter]:
    """Return all embedding-like weights requiring ZeRO gathering."""
    if model is None:
        return []
    if _is_deepspeed_engine(model):
        return []
    base_model = getattr(model, "module", model)
    modules: list[object] = []
    try:
        modules.append(getattr(base_model, "embed_tokens", None))
    except Exception:
        pass
    try:
        get_inp = getattr(base_model, "get_input_embeddings", None)
        if callable(get_inp):
            modules.append(get_inp())
    except Exception:
        pass
    try:
        get_out = getattr(base_model, "get_output_embeddings", None)
        if callable(get_out):
            modules.append(get_out())
    except Exception:
        pass
    try:
        modules.append(getattr(base_model, "lm_head", None))
    except Exception:
        pass

    weights: List[TorchParameter] = []
    seen: set[int] = set()
    for module in modules:
        if module is None:
            continue
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        weight_id = id(weight)
        if weight_id in seen:
            continue
        seen.add(weight_id)
        if _zero_param_ready_without_gather(weight):
            continue
        if _zero_status_name(weight) is not None:
            # Any explicit ZeRO status other than AVAILABLE indicates gather need.
            weights.append(weight)
            continue
        if getattr(weight, "ndim", 2) != 2:
            weights.append(weight)
    return weights


def _gather_callable() -> Optional[GatherCallable]:
    """Return the callable GatheredParameters helper when available."""
    if ds_zero is None:
        return None
    gather_obj = getattr(ds_zero, "GatheredParameters", None)
    if not callable(gather_obj):
        return None
    return cast(GatherCallable, gather_obj)


def _call_gather_fn(
    gather_fn: GatherCallable, params: Sequence[object], modifier_rank: Optional[int]
) -> ContextManager[None]:
    """Invoke GatheredParameters handling pre/post modifier_rank support."""
    if not callable(gather_fn):
        return nullcontext()
    if modifier_rank is None:
        return gather_fn(params)
    try:
        return gather_fn(params, modifier_rank=modifier_rank)
    except TypeError:
        return gather_fn(params)


@contextmanager
def _disable_hf_deepspeed_zero3_init() -> Iterator[None]:
    """Temporarily disable HF DeepSpeed ZeRO-3 init for model loading.

    When Accelerate enables DeepSpeed, ``transformers`` installs a global
    ``hf_deepspeed_config`` that causes subsequent ``from_pretrained`` calls
    to create partitioned (1-D) parameters. Reference models loaded under
    that global config are not wrapped by the DeepSpeed engine and cannot run
    forward passes. Clearing the global config inside this context forces
    full-parameter initialization for the reference model.
    """

    modules: list[tuple[Any, str, Any]] = []
    for module_name in (
        "transformers.deepspeed",
        "transformers.integrations.deepspeed",
    ):
        try:
            mod = importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError):
            continue
        for attr in ("hf_deepspeed_config", "_hf_deepspeed_config_weak_ref"):
            if hasattr(mod, attr):
                modules.append((mod, attr, getattr(mod, attr)))
        unset_fn = getattr(mod, "unset_hf_deepspeed_config", None)
        if callable(unset_fn):
            try:
                unset_fn()
            except (AttributeError, TypeError, RuntimeError, ValueError):
                pass
    if not modules:
        yield
        return
    try:
        for mod, attr, _value in modules:
            try:
                setattr(mod, attr, None)
            except (AttributeError, TypeError):
                pass
        yield
    finally:
        for mod, attr, value in modules:
            try:
                setattr(mod, attr, value)
            except (AttributeError, TypeError):
                pass


@contextmanager
def _maybe_zero_gather_embedding(model: Optional[PreTrainedModel]) -> Iterator[None]:
    """Gather ZeRO-sharded embedding weights before a forward pass.

    :param model: Model potentially wrapping ZeRO-managed embeddings.
    :type model: PreTrainedModel | None
    :returns: Context manager that gathers the embedding tensor when needed.
    :rtype: contextlib.AbstractContextManager[None]
    """
    if not _ensure_deepspeed_ready() or ds_zero is None:
        yield
        return
    if _is_deepspeed_engine(model):
        yield
        return
    maybe_gather = _gather_callable()
    if maybe_gather is None:
        yield
        return
    gather_fn: GatherCallable = maybe_gather
    weights = _embedding_weights_needing_gather(model)
    if not weights:
        yield
        return
    with _reserve_zero_gather_params(weights) as reserved:
        if not reserved:
            yield
            return
        gather_ctx = _call_gather_fn(gather_fn, reserved, modifier_rank=None)
        with gather_ctx:
            yield


def _zero_param_list(model: Optional[TorchModule]) -> List[TorchParameter]:
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
def _maybe_zero_gather_params(
    model: Optional[TorchModule],
    enabled: bool,
    gather_all_ranks: bool = False,
) -> Iterator[None]:
    """Gather ZeRO-partitioned params only when needed.

    :param model: Module whose parameters might be partitioned.
    :type model: torch.nn.Module | None
    :param enabled: Whether gathering should be attempted.
    :type enabled: bool
    :param gather_all_ranks: When True, gather parameters on every rank (avoid
        modifier-rank-only gathers).
    :type gather_all_ranks: bool
    :returns: Context manager yielding with parameters gathered when necessary.
    :rtype: contextlib.AbstractContextManager[None]
    """
    if not enabled or model is None or not _ensure_deepspeed_ready() or ds_zero is None:
        yield
        return
    if _is_deepspeed_engine(model):
        # DeepSpeed engines already manage parameter partitioning; avoid nested GatheredParameters.
        yield
        return
    maybe_gather = _gather_callable()
    if maybe_gather is None:
        yield
        return
    gather_fn: GatherCallable = maybe_gather
    params = _zero_param_list(model)
    zero_stage = _zero_stage(model)
    if zero_stage > 0:
        candidate_params = params
    else:
        candidate_params = [p for p in params if hasattr(p, "ds_id")]
    gather_params: List[TorchParameter] = []
    seen_param_ids: set[int] = set()
    for param in candidate_params:
        param_id = id(param)
        if param_id in seen_param_ids:
            continue
        seen_param_ids.add(param_id)
        # Do not nest a gather over parameters already materialized by another
        # active context; DeepSpeed asserts during repartition when active
        # sub-modules still hold references.
        if _zero_param_ready_without_gather(param):
            continue
        gather_params.append(param)
    if not gather_params:
        yield
        return
    if torch is not None:
        torch_mod = cast(Any, torch)
        cuda_mod = getattr(torch_mod, "cuda", None)
        if cuda_mod is not None:
            try:
                is_available = getattr(cuda_mod, "is_available", None)
                if callable(is_available) and is_available():
                    empty_cache = getattr(cuda_mod, "empty_cache", None)
                    if callable(empty_cache):
                        empty_cache()
            except (AttributeError, RuntimeError, TypeError):
                pass
    with _reserve_zero_gather_params(gather_params) as reserved:
        if not reserved:
            yield
            return
        modifier_rank = None if gather_all_ranks else 0
        gather_ctx = _call_gather_fn(gather_fn, reserved, modifier_rank=modifier_rank)
        with gather_ctx:
            yield


__all__ = [
    "_disable_hf_deepspeed_zero3_init",
    "_is_deepspeed_engine",
    "_maybe_zero_gather_embedding",
    "_maybe_zero_gather_params",
    "_maybe_patch_zero_no_sync",
    "_reserve_zero_gather_params",
]
