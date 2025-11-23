"""CLI entrypoint for the distilabel generation pipeline."""

from __future__ import annotations

from argparse import Namespace

from cli.generate import app as generate_cli_app
from cli.generate import build_generate_parser
from pipelines.generation.distilabel import (
    DistilabelGenerationConfig,
    DistilabelPipelineConfig,
    build_distilabel_pipeline,
    run_generation_job,
)

__all__ = [
    "DistilabelGenerationConfig",
    "DistilabelPipelineConfig",
    "build_distilabel_pipeline",
    "build_generate_parser",
    "main",
    "run_cli",
]


def run_cli(args: Namespace) -> None:
    """Backward-compatible helper used by tests and programmatic callers."""
    run_generation_job(
        DistilabelGenerationConfig.from_namespace(args),
        builder=build_distilabel_pipeline,
    )


def main() -> None:
    """Entry point for ``python -m src.generate`` (delegates to Typer CLI)."""
    generate_cli_app()


if __name__ == "__main__":
    main()
