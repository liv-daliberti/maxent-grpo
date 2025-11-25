"""Baseline GRPO training entrypoint.

Provides a thin wrapper around the training pipeline that either parses TRL
arguments from the CLI or delegates to the Hydra-based CLI when explicit args
are not provided. Exposed for ``python -m maxent_grpo.grpo`` and for
programmatic invocation inside orchestration code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from maxent_grpo.config import GRPOConfig, GRPOScriptArguments
from maxent_grpo.cli import hydra_cli, parse_grpo_args

if TYPE_CHECKING:
    from trl import ModelConfig

__all__ = ["cli", "main"]


def main(
    script_args: Optional[GRPOScriptArguments] = None,
    training_args: Optional[GRPOConfig] = None,
    model_args: "Optional[ModelConfig]" = None,
):
    """Run the baseline GRPO trainer or delegate to Hydra.

    :param script_args: Dataset/reward script arguments parsed via TRL or provided directly.
    :param training_args: GRPO training configuration produced by TRL.
    :param model_args: Model configuration passed to TRL/transformers trainers.
    :returns: Training result from :func:`maxent_grpo.pipelines.training.baseline.run_baseline_training`,
        or the Hydra CLI invocation result when no args are supplied.
    """

    if script_args is None or training_args is None or model_args is None:
        try:
            script_args, training_args, model_args = parse_grpo_args()
        except (ImportError, RuntimeError, SystemExit, ValueError):
            return hydra_cli.baseline_entry()
    from maxent_grpo.pipelines.training.baseline import run_baseline_training

    return run_baseline_training(script_args, training_args, model_args)


def cli() -> None:
    """Invoke the baseline entrypoint (CLI style).

    :returns: ``None``. Side effects include running training or delegating to Hydra.
    """

    main()


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    cli()
