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

CLI entrypoint for the distilabel generation pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly (e.g., python src/maxent_grpo/generate.py)
# by ensuring the package root is on sys.path.
if __package__ is None or __package__ == "":
    project_src = Path(__file__).resolve().parents[1]
    project_src_str = str(project_src)
    if project_src_str in sys.path:
        sys.path.remove(project_src_str)
    sys.path.insert(0, project_src_str)

from argparse import Namespace
from typing import Callable, Optional

from maxent_grpo.cli.generate import (
    app as generate_cli_app,
    build_generate_parser,
    run_cli,
)
from maxent_grpo.pipelines.generation.distilabel import (
    DistilabelGenerationConfig,
    DistilabelPipelineConfig,
    build_distilabel_pipeline,
    run_generation_job as _run_generation_job,
)
from maxent_grpo.pipelines.base import PipelineResult

__all__ = [
    "DistilabelGenerationConfig",
    "DistilabelPipelineConfig",
    "build_distilabel_pipeline",
    "build_generate_parser",
    "main",
    "run_cli",
]


def run_generation_job(
    cfg: DistilabelGenerationConfig | Namespace,
    builder: Optional[Callable[[DistilabelPipelineConfig], object]] = None,
) -> PipelineResult:
    """Run the distilabel generation pipeline for a given configuration.

    :param cfg: Distilabel generation config or argparse namespace produced by the CLI.
    :param builder: Optional pipeline factory overriding :func:`build_distilabel_pipeline`.
    :returns: Pipeline execution result containing the job name and status.
    """

    return _run_generation_job(cfg, builder)


def main() -> None:
    """Entry point for ``python -m maxent_grpo.generate``.

    Delegates to the Typer CLI declared in :mod:`maxent_grpo.cli.generate`.

    :returns: ``None``.
    """

    generate_cli_app()


if __name__ == "__main__":
    main()
