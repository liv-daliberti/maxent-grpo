"""CLI entrypoint for InfoSeed-GRPO training using the custom loop."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly
if __package__ is None or __package__ == "":
    project_src = Path(__file__).resolve().parents[1]
    project_src_str = str(project_src)
    if project_src_str in sys.path:
        sys.path.remove(project_src_str)
    sys.path.insert(0, project_src_str)

from typing import Any, Optional

from maxent_grpo.cli._test_hooks import ensure_usercustomize_loaded

ensure_usercustomize_loaded()

from maxent_grpo.config import GRPOConfig, GRPOScriptArguments
from maxent_grpo.cli import hydra_cli, parse_grpo_args

__all__ = ["cli", "main"]


def main(
    script_args: Optional[GRPOScriptArguments] = None,
    training_args: Optional[GRPOConfig] = None,
    model_args: "Optional[object]" = None,
) -> Any:
    """Run InfoSeed-GRPO via the custom runner or delegate to Hydra when args are missing.

    :param script_args: Optional script arguments; parsed from the CLI if omitted.
    :param training_args: Optional training configuration; parsed from the CLI if omitted.
    :param model_args: Optional model configuration; parsed from the CLI if omitted.
    :returns: Result of :func:`run_infoseed_training` or Hydra entrypoint invocation.
    :raises RuntimeError: If no CLI parser or Hydra entrypoint is available.
    :raises Exception: Propagates parser or training pipeline exceptions.
    """

    if script_args is None or training_args is None or model_args is None:
        try:
            script_args, training_args, model_args = parse_grpo_args()
        except (ImportError, RuntimeError, SystemExit, ValueError):
            return hydra_cli.infoseed_entry()
    from maxent_grpo.pipelines.training.infoseed import run_infoseed_training

    return run_infoseed_training(script_args, training_args, model_args)


def cli() -> None:
    """Invoke the InfoSeed entrypoint (CLI style).

    :returns: ``None``. The function executes for side effects.
    """

    main()


if __name__ == "__main__":  # pragma: no cover
    cli()
