"""Baseline GRPO CLI entrypoint backed by :mod:`pipelines.training.baseline`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cli import parse_grpo_args
from maxent_grpo.config import GRPOConfig, GRPOScriptArguments
from pipelines.training.baseline import run_baseline_training

if TYPE_CHECKING:  # pragma: no cover - import only for typing
    from trl import ModelConfig


def main(
    script_args: GRPOScriptArguments,
    training_args: GRPOConfig,
    model_args: "ModelConfig",
) -> None:
    """Run the baseline GRPO trainer with the provided TRL configs."""

    run_baseline_training(script_args, training_args, model_args)


def cli() -> None:
    """Parse TRL configs via ``TrlParser`` and run the baseline trainer."""

    script_args, training_args, model_args = parse_grpo_args()
    main(script_args, training_args, model_args)


if __name__ == "__main__":
    cli()
