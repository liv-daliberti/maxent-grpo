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

"""Convenience exports for the MaxEnt-GRPO helpers package."""

from typing import Any

__all__ = ["run_maxent_grpo"]


def run_maxent_grpo(*args: Any, **kwargs: Any) -> Any:
    """Lazy shim that defers importing heavy training dependencies."""
    from .run import run_maxent_grpo as _run  # Imported lazily to keep import time fast.

    return _run(*args, **kwargs)


def __dir__():
    return sorted(__all__)
