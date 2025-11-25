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

"""Lightweight fallbacks used when optional dependencies are absent."""

from __future__ import annotations

from typing import Any

from maxent_grpo.utils.imports import optional_import


def is_peft_model_safe(target: Any) -> bool:
    """Return True if accelerate.utils reports that the model uses PEFT adapters.

    :param target: Model or module to inspect for PEFT adapters.
    :returns: ``True`` when ``accelerate.utils.is_peft_model`` reports the target as PEFT-enabled; ``False`` otherwise or when accelerate is unavailable.
    """
    accelerate_utils = optional_import("accelerate.utils")
    if accelerate_utils is None:
        return False
    is_peft_model = getattr(accelerate_utils, "is_peft_model", None)
    if not callable(is_peft_model):
        return False
    try:
        return bool(is_peft_model(target))
    except (TypeError, AttributeError, ValueError):
        return False


def dist_with_fallback(dist: Any) -> Any:
    """Return ``torch.distributed`` or a safe stub when unavailable.

    :param dist: Candidate ``torch.distributed`` module or ``None``.
    :returns: The provided ``dist`` module with missing APIs patched, or a fallback stub that mimics the minimal API surface.
    """

    class _DistFallback:
        """Minimal subset of torch.distributed API exposed for environments without torch."""

        @staticmethod
        def is_available() -> bool:
            """Always report distributed as unavailable."""
            return False

        @staticmethod
        def is_initialized() -> bool:
            """Always report distributed as uninitialized."""
            return False

        @staticmethod
        def get_world_size() -> int:
            """Return a world size of 1 for single-process fallbacks."""
            return 1

        @staticmethod
        def all_gather_object(output_list, input_obj):
            """Populate ``output_list`` with ``input_obj`` when available."""
            if output_list:
                output_list[0] = input_obj

        @staticmethod
        def broadcast_object_list(_payload, _src=0):
            """No-op broadcast placeholder for stubbed environments."""
            return None

    if dist is None:
        return _DistFallback()
    if not hasattr(dist, "is_available"):
        dist.is_available = lambda: False
    if not hasattr(dist, "is_initialized"):
        dist.is_initialized = lambda: False
    if not hasattr(dist, "get_world_size"):
        dist.get_world_size = lambda: 1
    if not hasattr(dist, "all_gather_object"):
        dist.all_gather_object = lambda output_list, input_obj: output_list.__setitem__(
            0, input_obj
        )
    if not hasattr(dist, "broadcast_object_list"):
        dist.broadcast_object_list = lambda payload, src=0: None
    return dist
