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

CLI helper utilities shared across entrypoints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from .generate import build_generate_parser

if TYPE_CHECKING:
    from trl import ModelConfig

    from maxent_grpo.config import GRPOConfig, GRPOScriptArguments

__all__ = ["parse_grpo_args", "build_generate_parser"]


def parse_grpo_args() -> Tuple["GRPOScriptArguments", "GRPOConfig", "ModelConfig"]:
    """Parse GRPO CLI arguments (lazy import to keep deps light).

    :returns: Tuple of script args, training args, and model config parsed by TRL.
    :raises ImportError: If the TRL dependency is missing.
    """
    from maxent_grpo.training.cli import parse_grpo_args as _parse_grpo_args

    return _parse_grpo_args()
