"""Shared vLLM helper utilities reused across generation modules."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any, Callable, Optional, Sequence

from maxent_grpo.utils.fallbacks import optional_import


def zero3_gather_factory(
    accelerator: Any,
    import_fn: Optional[Callable[[str], Any]] = None,
) -> Callable[[Sequence[Any]], AbstractContextManager[Any]]:
    """Return a callable that gathers parameters when ZeRO-3 is active.

    :param accelerator: Accelerate object exposing ``state.deepspeed_plugin``.
    :type accelerator: Any
    :param import_fn: Optional import helper used to lazily import deepspeed.
    :type import_fn: Callable[[str], Any] | None
    :returns: Callable that wraps a parameter sequence in a gather context
        manager, or a no-op ``nullcontext`` when ZeRO-3 is not active.
    :rtype: Callable[[Sequence[Any]], contextlib.AbstractContextManager[Any]]
    """

    importer = import_fn or optional_import
    ds_plugin = getattr(getattr(accelerator, "state", None), "deepspeed_plugin", None)
    zero_stage = getattr(ds_plugin, "zero_stage", 0) or 0
    gather_cls = None
    if zero_stage == 3:
        deepspeed_mod = importer("deepspeed")
        zero_mod = getattr(deepspeed_mod, "zero", None) if deepspeed_mod else None
        gather_cls = getattr(zero_mod, "GatheredParameters", None)

    if gather_cls is None:
        return lambda _params: nullcontext()

    def _factory(params: Sequence[Any]) -> AbstractContextManager[Any]:
        return gather_cls(params)

    return _factory


def import_vllm_client_cls(
    import_fn: Optional[Callable[[str], Any]] = None,
) -> Optional[type]:
    """Return TRL's VLLMClient class if available.

    :param import_fn: Optional import helper to load TRL modules.
    :type import_fn: Callable[[str], Any] | None
    :returns: VLLMClient class when import succeeds, otherwise ``None``.
    :rtype: type | None
    """

    importer = import_fn or optional_import
    vllm_module = importer("trl.extras.vllm_client")
    if vllm_module is None:
        return None
    return getattr(vllm_module, "VLLMClient", None)


__all__ = ["import_vllm_client_cls", "zero3_gather_factory"]
