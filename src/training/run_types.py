"""Legacy dataclasses for checkpoint/training setup compatibility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from .types import PromptCacheEntry


@dataclass
class HubPushConfig:
    """Minimal Hub push configuration used by checkpoint helpers."""

    enabled: bool
    model_id: Optional[str]
    token: Optional[str]


@dataclass
class CheckpointConfig:
    """Checkpoint scheduling/configuration wrapper."""

    output_dir: str
    save_strategy: str
    save_steps: int
    save_total_limit: int
    hub: HubPushConfig


@dataclass
class PromptIOConfig:
    """Configuration for mapping dataset rows to prompts/answers."""

    prompt_column: str
    solution_column: str
    prompt_length_cache_get: Callable[[str], PromptCacheEntry]


@dataclass
class TrainDataBundle:
    """Bundle of dataset/loader artifacts used during training."""

    train_dataset: Any
    train_loader: Any
    train_sampler: Any
    prompt_io: PromptIOConfig
    steps_per_epoch: int
    batch_size: int


__all__ = [
    "CheckpointConfig",
    "HubPushConfig",
    "PromptIOConfig",
    "TrainDataBundle",
]
