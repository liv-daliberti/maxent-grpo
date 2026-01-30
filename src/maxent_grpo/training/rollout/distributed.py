"""Distributed helpers shared across generation utilities."""

from __future__ import annotations

from typing import Any, List, Optional

from maxent_grpo.utils.fallbacks import dist_with_fallback

from maxent_grpo.training.runtime import require_accelerator, require_torch

torch = require_torch("generation_comm")
Accelerator = require_accelerator("generation_comm")
dist = dist_with_fallback(getattr(torch, "distributed", None))


def _gather_object_list(accelerator: Accelerator, value: List[Any]) -> List[List[Any]]:
    """Gather Python lists across ranks with graceful Accelerate fallbacks.

    :param accelerator: Accelerator providing collective utilities.
    :type accelerator: accelerate.Accelerator
    :param value: Local list payload to gather across ranks.
    :type value: list[Any]
    :returns: List of gathered payloads ordered by rank.
    :rtype: list[list[Any]]
    """
    gather_fn = getattr(accelerator, "gather_object", None)
    if callable(gather_fn):
        return gather_fn(value)
    if dist is not None and dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        gathered: List[List[str]] = [[] for _ in range(world_size)]
        dist.all_gather_object(gathered, value)
        return gathered
    # Single-process fallback
    return [value]


def _broadcast_object_list(
    accelerator: Accelerator, payload: List[Any], *, src: int = 0
) -> None:
    """Broadcast python objects even when Accelerate lacks the helper."""
    broadcast_fn = getattr(accelerator, "broadcast_object_list", None)
    if callable(broadcast_fn):
        broadcast_fn(payload, src=src)
        return
    if dist is not None and dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(payload, src=src)


def _scatter_object(
    accelerator: Accelerator,
    input_list: Optional[List[Any]],
    *,
    src: int = 0,
) -> Any:
    """Scatter python objects from src to all ranks.

    :param accelerator: Accelerator providing distributed process metadata.
    :type accelerator: accelerate.Accelerator
    :param input_list: Objects available on ``src`` used for scattering.
    :type input_list: list[Any] | None
    :param src: Rank to scatter from.
    :type src: int
    :returns: Object assigned to the current rank (or ``None`` if unavailable).
    :rtype: Any
    """
    if accelerator.num_processes <= 1:
        if input_list is None:
            return None
        return input_list[0]
    scatter_fn = getattr(accelerator, "scatter_object", None)
    if callable(scatter_fn):
        return scatter_fn(
            input_list if accelerator.process_index == src else None,
            src=src,
        )
    if dist is not None and dist.is_available() and dist.is_initialized():
        # Prefer a broadcast-based implementation when possible. Some torch/
        # backend combinations have flaky support for scatter_object_list,
        # whereas broadcast_object_list tends to be more reliable.
        if callable(getattr(dist, "broadcast_object_list", None)):
            try:
                world_size = int(dist.get_world_size())
            except (RuntimeError, TypeError, ValueError):
                world_size = int(getattr(accelerator, "num_processes", 1) or 1)
            list_ok = input_list is None or (
                isinstance(input_list, list) and len(input_list) == world_size
            )
            if world_size > 0 and list_ok:
                payload = (
                    input_list
                    if accelerator.process_index == src and input_list is not None
                    else [None for _ in range(world_size)]
                )
                dist.broadcast_object_list(payload, src=src)
                try:
                    return payload[int(accelerator.process_index)]
                except (IndexError, TypeError, ValueError):
                    return None
        output: List[Any] = [None]
        dist.scatter_object_list(
            output,
            input_list if accelerator.process_index == src else None,
            src=src,
        )
        return output[0]
    # Fallback to best-effort local selection if no distributed backend is initialized.
    if input_list is None:
        return None
    return input_list[accelerator.process_index]


__all__ = ["_broadcast_object_list", "_gather_object_list", "_scatter_object"]
