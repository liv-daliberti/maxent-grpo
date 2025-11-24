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

Shared helpers/types for pipeline entrypoints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

LOG = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Lightweight container for pipeline outputs/metrics."""

    name: str
    metrics: Optional[dict[str, Any]] = None
    artifacts: Any = None


def log_pipeline_banner(name: str, cfg: Any) -> None:
    """Emit a consistent banner before running a pipeline."""

    LOG.info("Starting pipeline: %s", name)
    if cfg is not None:
        LOG.debug("Pipeline config for %s: %s", name, cfg)


__all__ = ["PipelineResult", "log_pipeline_banner"]