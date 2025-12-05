"""Baseline GRPO training entrypoint.

Provides a thin wrapper around the training pipeline that either parses TRL
arguments from the CLI or delegates to the Hydra-based CLI when explicit args
are not provided. Exposed for ``python -m maxent_grpo.grpo`` and for
programmatic invocation inside orchestration code.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly (e.g., accelerate launch src/maxent_grpo/grpo.py)
# by ensuring the package root is on sys.path.
if __package__ is None or __package__ == "":
    project_src = Path(__file__).resolve().parents[1]
    project_src_str = str(project_src)
    if project_src_str in sys.path:
        sys.path.remove(project_src_str)
    sys.path.insert(0, project_src_str)

from typing import TYPE_CHECKING, Optional

from maxent_grpo.config import GRPOConfig, GRPOScriptArguments

if TYPE_CHECKING:
    from trl import ModelConfig

try:  # Best-effort to expose CLI helpers when available.
    from maxent_grpo.cli import hydra_cli, parse_grpo_args  # type: ignore
except (ImportError, ModuleNotFoundError, AttributeError):  # pragma: no cover - optional deps may be absent
    hydra_cli = None  # type: ignore
    parse_grpo_args = None  # type: ignore

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
        # Prefer monkeypatched attributes (used in tests). Only fall back to Hydra when parsing is unavailable or fails.
        _parse_grpo_args = parse_grpo_args
        _hydra_cli = hydra_cli
        if not callable(_parse_grpo_args):
            try:
                from maxent_grpo.cli import parse_grpo_args as _parse_grpo_args  # type: ignore
            except (ImportError, ModuleNotFoundError, AttributeError):
                _parse_grpo_args = None
        if _hydra_cli is None:
            try:
                from maxent_grpo.cli import hydra_cli as _hydra_cli  # type: ignore
            except (ImportError, ModuleNotFoundError, AttributeError):
                try:
                    from maxent_grpo import cli as _cli_pkg  # type: ignore

                    _hydra_cli = getattr(_cli_pkg, "hydra_cli", None)
                except (ImportError, ModuleNotFoundError, AttributeError):
                    _hydra_cli = None
        if callable(_parse_grpo_args):
            try:
                script_args, training_args, model_args = _parse_grpo_args()
            except (
                RuntimeError,
                ImportError,
                ModuleNotFoundError,
                TypeError,
                ValueError,
                AttributeError,
            ):
                if _hydra_cli is not None:
                    return _hydra_cli.baseline_entry()
                raise
        elif _hydra_cli is not None:
            return _hydra_cli.baseline_entry()
        else:
            raise RuntimeError("No CLI parser available")
    baseline_mod = sys.modules.get("maxent_grpo.pipelines.training.baseline")
    if baseline_mod and hasattr(baseline_mod, "run_baseline_training"):
        run_baseline_training = baseline_mod.run_baseline_training  # type: ignore[attr-defined]
    else:
        from maxent_grpo.pipelines.training.baseline import run_baseline_training

    return run_baseline_training(script_args, training_args, model_args)


def cli() -> None:
    """Invoke the baseline entrypoint (CLI style).

    :returns: ``None``. Side effects include running training or delegating to Hydra.
    """

    main()


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    cli()
