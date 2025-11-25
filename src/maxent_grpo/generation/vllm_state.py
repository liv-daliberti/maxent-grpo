"""State containers used by the vLLM generation helper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from maxent_grpo.patches.vllm import VLLMLogprobResult


@dataclass
class _VLLMGenerationState:
    """Track state shared across multiple vLLM retries.

    :param prompts: Prompt strings requested from vLLM.
    :type prompts: list[str]
    :param target_counts: Desired completion counts per prompt.
    :type target_counts: list[int]
    :param requested_n: Requested completions per prompt passed to vLLM.
    :type requested_n: int
    :param round_limit: Maximum number of retry rounds to execute.
    :type round_limit: int
    :param track_logprobs: Whether to allocate metadata buffers for logprobs.
    :type track_logprobs: bool
    """

    prompts: List[str]
    target_counts: List[int]
    requested_n: int
    round_limit: int
    track_logprobs: bool
    aggregated: List[List[str]] = None
    aggregated_meta: Optional[List[List[Optional[VLLMLogprobResult]]]] = None

    def __post_init__(self) -> None:
        """Initialize aggregate storage and validate prompt alignment."""
        if len(self.target_counts) != len(self.prompts):
            raise ValueError("target_counts must align with prompts for vLLM state")
        self.aggregated = [[] for _ in self.prompts]
        if self.track_logprobs:
            self.aggregated_meta = [[] for _ in self.prompts]

    def pending_indices(self) -> List[int]:
        """Return prompt indices that still need completions.

        :returns: List of indices where collected completions are below target.
        :rtype: list[int]
        """
        pending: List[int] = []
        for idx, (completions, target) in enumerate(
            zip(self.aggregated, self.target_counts)
        ):
            if target > 0 and len(completions) < target:
                pending.append(idx)
        return pending

    def remaining_counts(self, indices: List[int]) -> List[int]:
        """Return outstanding completion counts for ``indices``.

        :param indices: Prompt indices to inspect.
        :type indices: list[int]
        :returns: Outstanding completion counts per index.
        :rtype: list[int]
        """
        counts: List[int] = []
        for idx in indices:
            target = self.target_counts[idx]
            counts.append(max(0, target - len(self.aggregated[idx])))
        return counts

    def trim(
        self,
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
        """Trim aggregated data down to the requested counts.

        :returns: Grouped completions and optional metadata truncated to target
            counts.
        :rtype: tuple[list[list[str]], list[list[VLLMLogprobResult | None]] | None]
        """
        trimmed = [
            group[:target] for group, target in zip(self.aggregated, self.target_counts)
        ]
        if self.aggregated_meta is None:
            return trimmed, None
        trimmed_meta = [
            meta_group[:target]
            for meta_group, target in zip(self.aggregated_meta, self.target_counts)
        ]
        return trimmed, trimmed_meta

    def drop_meta(self) -> None:
        """Discard stored metadata to reduce memory usage."""
        self.aggregated_meta = None


__all__ = ["_VLLMGenerationState"]
