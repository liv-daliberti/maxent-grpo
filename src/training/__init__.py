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

"""Core helpers powering the MaxEnt-GRPO trainer."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List

# Provide a lightweight wandb stub so legacy imports in test environments
# don't pull the full dependency graph.
import sys

if "wandb" not in sys.modules:
    sys.modules["wandb"] = SimpleNamespace(errors=SimpleNamespace(Error=RuntimeError))

from .loop import run_training_loop
from .types import (
    RuntimeHandles,
    RewardSpec,
    GenerationSettings,
    EvaluationSettings,
    OptimizationSettings,
    OptimizationSchedule,
    OptimizerHandles,
    ScoringSettings,
    ClipSettings,
    BatchingSettings,
    LoggingHandles,
    ControllerPaths,
    LoopSettings,
    TrainingLoopContext,
    GenerationBatch,
    PromptCompletionBatch,
    AdvantageStats,
    QDistribution,
    RewardMoments,
    RewardComputation,
    ScoreBatch,
    ReferenceLogprobs,
    LossOutputs,
    LossScalarBundle,
    LengthStats,
    BatchDiagnostics,
    ValidationContext,
    LoggingConfigView,
    TokenUsageStats,
    TrainingScalarStats,
    RewardComponentStats,
    RewardLoggingView,
    TrainingMetricsPayload,
    PromptCacheEntry,
    StepBatchInfo,
    StepResources,
    TrainingLoopState,
)
from .weighting.loss import SequenceScores
from .pipeline import PreparedBatch

__all__: List[str] = [
    "run_maxent_grpo",
    "run_training_loop",
    "TrainingLoopState",
    "StepBatchInfo",
    "StepResources",
    "RuntimeHandles",
    "RewardSpec",
    "GenerationSettings",
    "EvaluationSettings",
    "OptimizationSettings",
    "OptimizationSchedule",
    "OptimizerHandles",
    "ScoringSettings",
    "ClipSettings",
    "BatchingSettings",
    "LoggingHandles",
    "ControllerPaths",
    "LoopSettings",
    "TrainingLoopContext",
    "GenerationBatch",
    "PromptCompletionBatch",
    "AdvantageStats",
    "QDistribution",
    "RewardMoments",
    "RewardComputation",
    "ScoreBatch",
    "ReferenceLogprobs",
    "LossOutputs",
    "LossScalarBundle",
    "LengthStats",
    "BatchDiagnostics",
    "ValidationContext",
    "LoggingConfigView",
    "TokenUsageStats",
    "TrainingScalarStats",
    "RewardComponentStats",
    "RewardLoggingView",
    "TrainingMetricsPayload",
    "PromptCacheEntry",
    "SequenceScores",
    "PreparedBatch",
]


def run_maxent_grpo(*args: Any, **kwargs: Any) -> Any:
    """Compatibility wrapper for callers still importing ``training.run_maxent_grpo``.

    The legacy shim modules have been removed. To launch training, invoke the
    Hydra CLI entrypoint (``maxent-grpo-maxent``) or compose your own runner
    using the building blocks in ``training.loop``, ``training.pipeline``, and
    ``training.run_helpers``.
    """
    raise NotImplementedError(
        "training.run_maxent_grpo is no longer provided. Use the Hydra CLI "
        "entrypoint (maxent-grpo-maxent) or build a runner with training.loop/"
        "training.pipeline."
    )


def __dir__():
    """Return sorted public attributes for IDE completion."""
    return sorted(__all__)
