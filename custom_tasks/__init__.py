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

"""Custom LightEval task definitions used by the MaxEnt-GRPO repo."""

from __future__ import annotations

from .math_500_short import TASKS_TABLE as _MATH_500_SHORT

TASKS_TABLE = [*_MATH_500_SHORT]

__all__ = ["TASKS_TABLE"]
