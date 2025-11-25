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
"""

# Shared generation helpers used across the MaxEnt training stack.

from __future__ import annotations

from .common import (
    AggregatedGenerationState,
    append_completion_group,
    determine_retry_limit,
    drop_empty_prompt_groups,
    flatten_ref_metadata,
    pending_generation_indices,
    retry_incomplete_prompts,
    seed_generation_groups,
    truncate_to_expected_counts,
)
from .helpers import flatten_prompt_completions

__all__ = [
    "AggregatedGenerationState",
    "append_completion_group",
    "determine_retry_limit",
    "drop_empty_prompt_groups",
    "flatten_prompt_completions",
    "flatten_ref_metadata",
    "pending_generation_indices",
    "retry_incomplete_prompts",
    "seed_generation_groups",
    "truncate_to_expected_counts",
]
