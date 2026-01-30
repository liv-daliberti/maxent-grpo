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

"""Logging protocols and dataclasses shared across the training stack."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Callable, TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from .rewards import LossOutputs, BatchDiagnostics, LengthStats
    from ..weighting.types import WeightingSettings, WeightLoggingView
    from .runtime import ClipSettings, OptimizationSchedule
else:  # pragma: no cover - typing fallbacks
    LossOutputs = Any
    BatchDiagnostics = Any
    LengthStats = Any
    WeightLoggingView = Any
    WeightingSettings = Any
    ClipSettings = Any
    OptimizationSchedule = Any


class MetricWriter(Protocol):
    """Protocol describing a metric writer used by the training loop."""

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        """Record ``metrics`` for a training ``step``."""

    def flush(self) -> None:
        """Flush buffered metrics to their storage backend."""


@dataclass
class LoggingHandles:
    """Callbacks for logging and checkpointing."""

    metric_writer: MetricWriter
    save_checkpoint: Callable[[str], None]
    save_strategy: str
    save_steps: int
    wandb_run: Optional[Any]
    checkpoint_state_ref: Optional[Dict[str, Any]] = None

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Send metrics to the configured writer.

        :param metrics: Scalar payload to log.
        :type metrics: dict[str, Any]
        :param step: Current global training step.
        :type step: int
        """
        self.metric_writer.log(metrics, step)

    def flush_metrics(self) -> None:
        """Flush the writer when it exposes a ``flush`` method."""
        flush = getattr(self.metric_writer, "flush", None)
        if callable(flush):
            flush()

    @contextmanager
    def step_logger(
        self, step: int, *, enabled: bool = True
    ) -> Iterator["_MetricStepLogger | _NoopMetricLogger"]:
        """Yield a helper that logs metrics for a specific training step.

        :param step: Current training step being logged.
        :type step: int
        :param enabled: Disable logging when ``False`` (e.g., eval only).
        :type enabled: bool
        :yields: A helper exposing ``log`` for the provided ``step``.
        """
        if not enabled:
            yield _NoopMetricLogger()
            return
        logger = _MetricStepLogger(self.metric_writer, step)
        try:
            yield logger
        finally:
            self.flush_metrics()


class _MetricStepLogger:
    """Small helper that binds a metric writer to a fixed step."""

    def __init__(self, writer: MetricWriter, step: int) -> None:
        """Store the writer and associated step.

        :param writer: Metric writer used to persist values.
        :type writer: MetricWriter
        :param step: Step index applied to each log call.
        :type step: int
        """
        self._writer = writer
        self._step = step

    def log(self, metrics: Dict[str, Any]) -> None:
        """Log ``metrics`` using the bound step."""
        self._writer.log(metrics, self._step)


class _NoopMetricLogger:
    """Logger used when metric logging is disabled."""

    def log(self, _metrics: Dict[str, Any]) -> None:  # pragma: no cover - trivial
        """Discard metrics without side effects."""
        return


class MetricState(Protocol):
    """Minimal interface required for metric accumulation."""

    global_step: int
    num_input_tokens_seen: float
    metric_sums: Dict[str, float]
    metric_counts: Dict[str, int]


@dataclass
class LogStepArtifacts:
    """Helper container for optimizer/loss diagnostics per step."""

    loss_outputs: "LossOutputs"
    diagnostics: "BatchDiagnostics"
    grad_norm_scalar: Optional[float]
    epoch_progress: float

    def as_dict(self) -> Dict[str, Any]:
        """Return a dict view useful for debugging/log statements."""
        return {
            "loss_outputs": self.loss_outputs,
            "diagnostics": self.diagnostics,
            "grad_norm_scalar": self.grad_norm_scalar,
            "epoch_progress": self.epoch_progress,
        }


@dataclass
class LoggingConfigView:
    """Pointers to configs referenced while logging."""

    weighting: "WeightingSettings"
    clipping: "ClipSettings"
    schedule: "OptimizationSchedule"


@dataclass
class TokenUsageStats:
    """Aggregate completion/input token statistics."""

    avg_completion_tokens: float
    num_completion_tokens: float
    num_input_tokens: float


@dataclass
class TrainingScalarStats:
    """Scalar values that vary every logging step."""

    ref_logp_mean: float
    tokens: TokenUsageStats
    current_lr: float
    grad_norm_scalar: Optional[float]
    epoch_progress: float
    vllm_latency_ms: Optional[float]

    @property
    def avg_completion_tokens(self) -> float:
        """Return the average completion token length.

        :returns: Running average of completion token counts.
        :rtype: float
        """
        return self.tokens.avg_completion_tokens

    @avg_completion_tokens.setter
    def avg_completion_tokens(self, value: float) -> None:
        """Update the average completion token length.

        :param value: New average completion token count.
        :type value: float
        """
        self.tokens.avg_completion_tokens = value

    @property
    def num_completion_tokens(self) -> float:
        """Return the total completion token count processed.

        :returns: Total completion token count accumulated.
        :rtype: float
        """
        return self.tokens.num_completion_tokens

    @num_completion_tokens.setter
    def num_completion_tokens(self, value: float) -> None:
        """Update the total completion token count processed.

        :param value: New total completion token count.
        :type value: float
        """
        self.tokens.num_completion_tokens = value

    @property
    def num_input_tokens(self) -> float:
        """Return the total input token count processed.

        :returns: Total input token count accumulated.
        :rtype: float
        """
        return self.tokens.num_input_tokens

    @num_input_tokens.setter
    def num_input_tokens(self, value: float) -> None:
        """Update the total input token count processed.

        :param value: New total input token count.
        :type value: float
        """
        self.tokens.num_input_tokens = value


@dataclass
class RewardComponentStats:
    """Mean/std summary for an individual reward component."""

    mean: float
    std: float


@dataclass
class RewardLoggingView:
    """Aggregated reward/advantage statistics for logging."""

    reward_mean: float
    reward_std: float
    frac_zero_std: float
    advantage_mean: float
    advantage_std: float
    advantage_count: int
    per_reward: Dict[str, RewardComponentStats]
    q_entropy_mean: float
    q_entropy_std: float
    q_entropy_min: float
    q_entropy_max: float


@dataclass
class TrainingMetricsPayload:
    """Container for scalar values used by the training logger."""

    reward_stats: RewardLoggingView
    weight_stats: "WeightLoggingView"
    loss_outputs: "LossOutputs"
    diagnostics: "BatchDiagnostics"
    length_stats: "LengthStats"
    config: LoggingConfigView
    scalars: TrainingScalarStats
    seed_metrics: Optional[Dict[str, float]] = None


__all__ = [
    "LoggingConfigView",
    "LoggingHandles",
    "LogStepArtifacts",
    "MetricState",
    "MetricWriter",
    "RewardComponentStats",
    "RewardLoggingView",
    "TokenUsageStats",
    "TrainingMetricsPayload",
    "TrainingScalarStats",
]
