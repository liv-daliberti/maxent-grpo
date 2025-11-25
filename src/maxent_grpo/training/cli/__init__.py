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

"""Training-specific CLI helpers (TRL argument parsing, etc.)."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from maxent_grpo.config import GRPOConfig, GRPOScriptArguments
    from trl import ModelConfig

__all__ = ["parse_grpo_args"]


def __getattr__(name: str):
    if name == "parse_grpo_args":
        from . import trl as cli_trl

        value = cli_trl.parse_grpo_args
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__} has no attribute {name}")
