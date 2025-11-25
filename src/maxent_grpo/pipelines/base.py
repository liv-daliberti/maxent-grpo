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
    """Lightweight container for pipeline outputs/metrics.

    :param name: Identifier for the pipeline entrypoint (e.g., ``training.maxent``).
    :type name: str
    :param metrics: Optional mapping of metric names to values surfaced by the pipeline.
    :type metrics: dict[str, Any] | None
    :param artifacts: Optional opaque object returned by the pipeline implementation
        (dataset handles, model paths, etc.).
    :type artifacts: Any
    """

    name: str
    metrics: Optional[dict[str, Any]] = None
    artifacts: Any = None


def log_pipeline_banner(name: str, cfg: Any) -> None:
    """Emit a consistent banner before running a pipeline.

    :param name: Human-readable pipeline identifier shown in logs.
    :type name: str
    :param cfg: Configuration object/namespace whose fields help contextualize the run.
        Logged at debug level to avoid noisy output in production.
    :type cfg: Any
    :returns: ``None`` after logging the banner.
    :rtype: None
    """

    LOG.info("Starting pipeline: %s", name)
    if cfg is not None:
        LOG.debug("Pipeline config for %s: %s", name, cfg)


__all__ = ["PipelineResult", "log_pipeline_banner"]
