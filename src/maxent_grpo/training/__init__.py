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

import importlib as _importlib
import logging
from typing import Any, List, TYPE_CHECKING

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
    SequenceScores,
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

_LAZY_ATTRS = {
    "AnalyticControllerObjective": (
        "maxent_grpo.training.controller_objective",
        "AnalyticControllerObjective",
    ),
    "TruncatedBackpropControllerObjective": (
        "maxent_grpo.training.controller_objective",
        "TruncatedBackpropControllerObjective",
    ),
}

LOG = logging.getLogger(__name__)

# Lazy-only declarations to satisfy ``__all__`` without eager imports.
PreparedBatch: Any

__all__: List[str] = [
    "AnalyticControllerObjective",
    "TruncatedBackpropControllerObjective",
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
    "run_baseline_training",
    "generation",
    "patches",
    "pipeline",
    "runtime",
    "state",
    "rollout",
    "telemetry",
    "cli",
    "scoring",
]

# Try eager resolution when dependencies are available; fall back to lazy __getattr__.
try:
    from maxent_grpo.training.pipeline import PreparedBatch
except ImportError:
    LOG.debug("Deferring training imports until dependencies are available.")


def __dir__() -> List[str]:
    """Return sorted public attributes for IDE completion.

    :returns: Sorted list of public attribute names.
    :rtype: list[str]
    """
    return sorted(__all__)


def __getattr__(name: str) -> Any:
    """Lazily import heavy submodules to avoid circular imports during startup.

    :param name: Attribute name to resolve.
    :returns: Imported module or attribute for ``name`` when available.
    :raises AttributeError: If the requested name is not a known lazy attribute.
    """

    if name in {
        "pipeline",
        "state",
        "rollout",
        "cli",
        "scoring",
        "optim",
        "generation",
        "patches",
        "runtime",
        "telemetry",
    }:
        module = _importlib.import_module(f"maxent_grpo.training.{name}")
        globals()[name] = module
        return module
    lazy_target = _LAZY_ATTRS.get(name)
    if lazy_target is not None:
        module_name, attr_name = lazy_target
        module = _importlib.import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    if name == "PreparedBatch":
        module = _importlib.import_module("maxent_grpo.training.pipeline")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name == "run_baseline_training":
        module = _importlib.import_module("maxent_grpo.training.baseline")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!s} has no attribute {name!s}")


# Statically expose lazy-resolved attributes for linters/IDE while deferring imports.
if TYPE_CHECKING:
    from maxent_grpo.training.controller_objective import (
        AnalyticControllerObjective as AnalyticControllerObjective,
    )
    from maxent_grpo.training.controller_objective import (
        TruncatedBackpropControllerObjective as TruncatedBackpropControllerObjective,
    )
    from maxent_grpo.training.baseline import (
        run_baseline_training as run_baseline_training,
    )
    from maxent_grpo.training.pipeline import PreparedBatch as PreparedBatch
    from maxent_grpo.training import generation as generation
    from maxent_grpo.training import patches as patches
    from maxent_grpo.training import pipeline as pipeline
    from maxent_grpo.training import runtime as runtime
    from maxent_grpo.training import state as state
    from maxent_grpo.training import rollout as rollout
    from maxent_grpo.training import telemetry as telemetry
    from maxent_grpo.training import cli as cli
    from maxent_grpo.training import scoring as scoring
