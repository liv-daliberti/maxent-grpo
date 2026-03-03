"""
Copyright 2025 Liv d'Aliberti

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Re-export MaxEnt reward helpers under a dedicated namespace.
"""

from __future__ import annotations

from maxent_grpo.training.rewards import (
    AggregatedGenerationState,
    compute_reward_statistics,
    compute_reward_totals,
    group_advantages,
    prepare_generation_batch,
    reward_moments,
)

# Weighting helpers are part of the MaxEnt implementation but live in the
# ``training.weighting`` package.  Historically the ``rewards.maxent`` sub
# package only re-exported a handful of reward/statistics helpers; tests made
# sure those names were available but nothing else ever imported the module.
#
# That has been a continual source of confusion: callers who expect "MaxEnt"
# utilities inside the rewards namespace are surprised to find nothing there.
# In particular the pipeline and evaluation code operate on weighted
# distributions, so users often try ``from maxent_grpo.rewards.maxent import
# compute_weight_stats`` and fail.  To make the public API more coherent we
# now re-export the entire public interface of ``maxent_grpo.training.weighting``
# under ``maxent_grpo.rewards.maxent``.  The module remains lightweight by
# lazily importing as needed.

from importlib import import_module

# ``weighting`` defines an ``__all__`` with all of the names we want to forward.
_weighting = import_module("maxent_grpo.training.weighting")

# bring the public names into this module's namespace and expand __all__.
for _name in getattr(_weighting, "__all__", []):
    globals()[_name] = getattr(_weighting, _name)

__all__ = [
    "AggregatedGenerationState",
    "compute_reward_statistics",
    "compute_reward_totals",
    "group_advantages",
    "prepare_generation_batch",
    "reward_moments",
]
__all__.extend(getattr(_weighting, "__all__", []))
