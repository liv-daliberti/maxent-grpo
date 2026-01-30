"""Baseline GRPO training entrypoint.

Provides a thin wrapper around the training pipeline that either parses TRL
arguments from the CLI or delegates to the Hydra-based CLI when explicit args
are not provided. Exposed for ``python -m maxent_grpo.grpo`` and for
programmatic invocation inside orchestration code.
"""

from __future__ import annotations

import importlib
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

from typing import TYPE_CHECKING, Any, Callable, Optional, cast
from types import SimpleNamespace

from maxent_grpo.cli._test_hooks import ensure_usercustomize_loaded

ensure_usercustomize_loaded()

from maxent_grpo.config import GRPOConfig, GRPOScriptArguments

if TYPE_CHECKING:
    from trl import ModelConfig  # type: ignore[reportMissingTypeStubs]

def _missing_hydra_entry(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - fallback stub
    raise RuntimeError(
        "Hydra CLI entrypoints are unavailable; install optional CLI dependencies."
    )


try:  # Best-effort to expose CLI helpers when available.
    from maxent_grpo.cli import hydra_cli, parse_grpo_args  # type: ignore
except (ImportError, ModuleNotFoundError, AttributeError):  # pragma: no cover - optional deps may be absent
    hydra_cli = SimpleNamespace(  # type: ignore
        baseline_entry=_missing_hydra_entry,
        maxent_entry=_missing_hydra_entry,
        infoseed_entry=_missing_hydra_entry,
        hydra_entry=_missing_hydra_entry,
    )
    parse_grpo_args = None  # type: ignore

__all__ = ["cli", "main"]


def _resolve_cli_attr(attr_name: str) -> Any:
    """Best-effort import helper for optional CLI attributes."""

    try:
        cli_mod = importlib.import_module("maxent_grpo.cli")
    except (ImportError, ModuleNotFoundError, AttributeError):
        cli_mod = None
    if cli_mod is not None:
        attr = getattr(cli_mod, attr_name, None)
        if attr is not None:
            return attr
    try:
        pkg = importlib.import_module("maxent_grpo")
    except (ImportError, ModuleNotFoundError, AttributeError):
        return None
    cli_pkg = getattr(pkg, "cli", None)
    if cli_pkg is None:
        return None
    return getattr(cli_pkg, attr_name, None)


def main(
    script_args: Optional[GRPOScriptArguments] = None,
    training_args: Optional[GRPOConfig] = None,
    model_args: "Optional[ModelConfig]" = None,
) -> Any:
    """Run the baseline GRPO trainer or delegate to Hydra.

    :param script_args: Dataset/reward script arguments parsed via TRL or provided directly.
    :param training_args: GRPO training configuration produced by TRL.
    :param model_args: Model configuration passed to TRL/transformers trainers.
    :returns: Training result from :func:`maxent_grpo.pipelines.training.baseline.run_baseline_training`,
        or the Hydra CLI invocation result when no args are supplied.
    :raises RuntimeError: If no CLI parser or Hydra entrypoint is available.
    :raises Exception: Propagates parser or training pipeline exceptions.
    """

    if script_args is None or training_args is None or model_args is None:
        # Prefer monkeypatched attributes (used in tests). Only fall back to Hydra when parsing is unavailable or fails.
        _parse_grpo_args = parse_grpo_args
        _hydra_cli = hydra_cli
        if not callable(_parse_grpo_args):
            parsed = _resolve_cli_attr("parse_grpo_args")
            _parse_grpo_args = parsed if callable(parsed) else None
        if _hydra_cli is None:
            resolved_hydra = _resolve_cli_attr("hydra_cli")
            _hydra_cli = resolved_hydra if resolved_hydra is not None else None
        if callable(_parse_grpo_args):
            try:
                parser = cast(
                    Callable[
                        [],
                        tuple[GRPOScriptArguments, GRPOConfig, "ModelConfig"],
                    ],
                    _parse_grpo_args,
                )
                script_args, training_args, model_args = parser()
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
    meta_enabled = bool(getattr(training_args, "controller_meta_enabled", False))
    if meta_enabled:
        training_args.train_grpo_objective = True
        from maxent_grpo.pipelines.training.maxent import run_maxent_training

        return run_maxent_training(script_args, training_args, model_args)

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
