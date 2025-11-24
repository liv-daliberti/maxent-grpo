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

from .generate import build_generate_parser

__all__ = ["parse_grpo_args", "build_generate_parser"]


def parse_grpo_args():
    """Lazily import the TRL parser to avoid heavyweight deps at import time."""
    from maxent_grpo.training.cli import parse_grpo_args as _parse_grpo_args

    return _parse_grpo_args()