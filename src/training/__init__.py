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
    """Lazy shim that defers importing heavy training dependencies.

    :param args: Positional arguments forwarded to ``training.run.run_maxent_grpo``.
    :type args: Any
    :param kwargs: Keyword arguments forwarded to ``training.run.run_maxent_grpo``.
    :type kwargs: Any
    :returns: Result returned by the actual ``run_maxent_grpo`` implementation.
    :rtype: Any
    """
    from .run import run_maxent_grpo as _run  # pylint: disable=import-error

    return _run(*args, **kwargs)


def __dir__():
    """Return sorted public attributes for IDE completion."""
    return sorted(__all__)
