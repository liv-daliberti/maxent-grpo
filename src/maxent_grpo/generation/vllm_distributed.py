"""Distributed helpers used by the vLLM generation helper."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple
import sys

from maxent_grpo.training.runtime import require_accelerator, require_torch

torch = require_torch("generation_vllm_dist")
Accelerator = require_accelerator("generation_vllm_dist")


def _current_torch() -> Any:
    """Return torch, preferring the vLLM module shim when patched in tests.

    :returns: The torch module (possibly a shim injected by tests).
    :rtype: Any
    """

    vllm_mod = sys.modules.get("maxent_grpo.generation.vllm")
    if vllm_mod is not None and getattr(vllm_mod, "torch", None) is not None:
        return vllm_mod.torch  # type: ignore[attr-defined]
    return torch


def _gather_object_list(accelerator: Accelerator, value: List[Any]) -> List[List[Any]]:
    """Gather python lists across ranks with Accelerate/torch fallbacks.

    :param accelerator: Accelerate instance providing distributed utilities.
    :type accelerator: accelerate.Accelerator
    :param value: Python list to broadcast to every process.
    :type value: list[Any]
    :returns: List of lists containing gathered values per rank.
    :rtype: list[list[Any]]
    """
    gather_fn = getattr(accelerator, "gather_object", None)
    if callable(gather_fn):
        return gather_fn(value)
    dist = getattr(_current_torch(), "distributed", None)
    if dist is not None and dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        gathered: List[List[str]] = [[] for _ in range(world_size)]
        dist.all_gather_object(gathered, value)
        return gathered
    return [value]


def _scatter_object(
    accelerator: Accelerator,
    input_list: Optional[List[Any]],
    *,
    src: int = 0,
) -> Any:
    """Scatter python objects from ``src`` rank to every other process.

    :param accelerator: Accelerate instance providing distributed utilities.
    :type accelerator: accelerate.Accelerator
    :param input_list: Sequence of objects to scatter; only required on the
        source rank.
    :type input_list: list[Any] | None
    :param src: Source rank that owns ``input_list``.
    :type src: int
    :returns: Object slice corresponding to the current rank, or ``None`` when
        ``input_list`` is missing.
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
    dist = getattr(_current_torch(), "distributed", None)
    if dist is not None and dist.is_available() and dist.is_initialized():
        output: List[Any] = [None]
        dist.scatter_object_list(
            output,
            input_list if accelerator.process_index == src else None,
            src=src,
        )
        return output[0]
    if input_list is None:
        return None
    return input_list[accelerator.process_index]


class VLLMDistributedMixin:
    """Split out scatter/gather helpers from the vLLM helper."""

    ctx: Any

    def _flatten_prompts_for_broadcast(
        self,
        prompts: List[str],
        per_prompt_counts: Optional[List[int]] = None,
    ) -> Tuple[List[str], List[int], Optional[List[int]]]:
        """Gather prompts and counts from all ranks and flatten them.

        :param prompts: Local prompt list for the current rank.
        :type prompts: list[str]
        :param per_prompt_counts: Optional completion counts aligned to
            ``prompts``.
        :type per_prompt_counts: list[int] | None
        :returns: Tuple of flattened prompts, offsets indicating each rank's
            slice start, and flattened counts if provided.
        :rtype: tuple[list[str], list[int], list[int] | None]
        """
        accelerator = self.ctx.accelerator
        gathered = _gather_object_list(accelerator, prompts)
        flat_prompts: List[str] = []
        offsets: List[int] = []
        running = 0
        for group in gathered:
            offsets.append(running)
            running += len(group)
            flat_prompts.extend(group)
        flat_counts: Optional[List[int]] = None
        if per_prompt_counts is not None:
            gathered_counts = _gather_object_list(accelerator, per_prompt_counts)
            flat_counts = []
            for group in gathered_counts:
                flat_counts.extend(int(val) for val in group)
        return flat_prompts, offsets, flat_counts

    def _build_scatter_payload(
        self,
        offsets: List[int],
        world_size: int,
        flat_prompts: List[str],
        grouped_all: Optional[List[List[str]]],
        meta_all: Optional[List[List[Optional[Any]]]],
    ) -> List[Tuple[List[List[str]], Optional[List[List[Optional[Any]]]]]]:
        """Build payload slices for scatter, trimming to each rank's prompt slice.

        :param offsets: Offsets computed by ``_flatten_prompts_for_broadcast``.
        :type offsets: list[int]
        :param world_size: Total number of ranks.
        :type world_size: int
        :param flat_prompts: Flattened prompt list across all ranks.
        :type flat_prompts: list[str]
        :param grouped_all: Grouped completions aligned to ``flat_prompts``.
        :type grouped_all: list[list[str]] | None
        :param meta_all: Grouped metadata aligned to ``flat_prompts``.
        :type meta_all: list[list[object | None]] | None
        :returns: List of tuples containing grouped completions and metadata for
            each rank.
        :rtype: list[tuple[list[list[str]], list[list[object | None]] | None]]
        """
        payload: List[Tuple[List[List[str]], Optional[List[List[Optional[Any]]]]]] = []
        total = len(flat_prompts)
        for rank in range(world_size):
            start = offsets[rank]
            end = offsets[rank + 1] if rank + 1 < len(offsets) else total
            slice_grouped = [] if grouped_all is None else grouped_all[start:end]
            slice_meta = None if meta_all is None else meta_all[start:end]
            payload.append((slice_grouped, slice_meta))
        return payload

    def _scatter_vllm_payload(
        self,
        flat_prompts: List[str],
        offsets: List[int],
        grouped_all: Optional[List[List[str]]],
        meta_all: Optional[List[List[Optional[Any]]]],
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[Any]]]]]:
        """Scatter aggregated outputs from main process to all ranks.

        :param flat_prompts: Flattened prompts gathered across ranks.
        :type flat_prompts: list[str]
        :param offsets: Per-rank offsets into ``flat_prompts``.
        :type offsets: list[int]
        :param grouped_all: Grouped completions generated on main process.
        :type grouped_all: list[list[str]] | None
        :param meta_all: Grouped metadata generated on main process.
        :type meta_all: list[list[object | None]] | None
        :returns: Local grouped completions and metadata for the current rank.
        :rtype: tuple[list[list[str]], list[list[object | None]] | None]
        """
        accelerator = self.ctx.accelerator
        world_size = accelerator.num_processes
        if world_size <= 1:
            return self._pluck_rank_outputs(
                grouped_all or [],
                meta_all,
                offsets,
                flat_prompts,
            )

        scatter_payload: Optional[
            List[Tuple[List[List[str]], Optional[List[List[Optional[Any]]]]]]
        ] = None
        if accelerator.is_main_process:
            scatter_payload = self._build_scatter_payload(
                offsets,
                world_size,
                flat_prompts,
                grouped_all,
                meta_all,
            )
        scatter_fn = getattr(self, "_scatter_object", _scatter_object)
        scatter_result = scatter_fn(accelerator, scatter_payload, src=0)
        if scatter_result is None:
            return [], None
        grouped_local, meta_local = scatter_result
        if grouped_local is None:
            return [], None
        filled_local: List[List[str]] = []
        for group in grouped_local or []:
            filled_local.append(group if group is not None else [])
        return filled_local, meta_local

    def _pluck_rank_outputs(
        self,
        grouped_all: List[List[str]],
        meta_all: Optional[List[List[Optional[Any]]]],
        offsets: List[int],
        prompts: List[str],
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[Any]]]]]:
        """Return this rank's slice from globally grouped outputs.

        :param grouped_all: Grouped completions for every prompt across ranks.
        :type grouped_all: list[list[str]]
        :param meta_all: Grouped metadata for every prompt across ranks.
        :type meta_all: list[list[object | None]] | None
        :param offsets: Offsets produced by ``_flatten_prompts_for_broadcast``.
        :type offsets: list[int]
        :param prompts: Prompts owned by the current rank.
        :type prompts: list[str]
        :returns: Grouped completions and metadata for the current rank.
        :rtype: tuple[list[list[str]], list[list[object | None]] | None]
        """
        accelerator = self.ctx.accelerator
        rank = accelerator.process_index
        start = offsets[rank]
        end = start + len(prompts)
        grouped_local = grouped_all[start:end]
        meta_local = None if meta_all is None else meta_all[start:end]
        filled_local: List[List[str]] = []
        for group in grouped_local:
            filled_local.append(group if group is not None else [])
        return filled_local, meta_local


__all__ = ["VLLMDistributedMixin", "_gather_object_list", "_scatter_object"]
