"""MaxEnt-GRPO training entrypoint."""

from __future__ import annotations

import importlib
import os
import sys
import logging
from pathlib import Path
import signal
import faulthandler
from typing import Any, Callable, cast

# Allow running this file directly (e.g., accelerate launch src/maxent_grpo/maxent_grpo.py)
# by ensuring the package root is on sys.path.
if __package__ is None or __package__ == "":
    project_src = Path(__file__).resolve().parents[1]
    project_src_str = str(project_src)
    if project_src_str in sys.path:
        sys.path.remove(project_src_str)
    sys.path.insert(0, project_src_str)

from maxent_grpo.cli._test_hooks import ensure_usercustomize_loaded

__all__ = ["main"]

LOG = logging.getLogger(__name__)

if os.environ.get("MAXENT_FAULTHANDLER", "").strip():
    try:
        faulthandler.enable(all_threads=True)
    except (OSError, RuntimeError, ValueError) as exc:
        LOG.warning("Failed to enable faulthandler: %s", exc)
    if hasattr(signal, "SIGUSR1"):
        try:
            faulthandler.register(signal.SIGUSR1, all_threads=True)
        except (OSError, RuntimeError, ValueError) as exc:
            LOG.warning("Failed to register faulthandler SIGUSR1 handler: %s", exc)


def _resolve_cli_attr(attr_name: str) -> Any:
    """Best-effort import helper for optional CLI attributes."""

    try:
        pkg = importlib.import_module("maxent_grpo")
    except (ImportError, ModuleNotFoundError, AttributeError):
        pkg = None
    if pkg is not None:
        attr = getattr(pkg, attr_name, None)
        if attr is not None:
            return attr
    try:
        if attr_name == "hydra_cli":
            return importlib.import_module("maxent_grpo.cli.hydra_cli")
        cli_mod = importlib.import_module("maxent_grpo.cli")
    except (ImportError, ModuleNotFoundError, AttributeError):
        return None
    return getattr(cli_mod, attr_name, None)


def main(
    script_args: Any = None,
    training_args: Any = None,
    model_args: Any = None,
) -> Any:
    """Run the MaxEnt trainer when configs are provided, else delegate to Hydra.

    :param script_args: Optional GRPO script arguments; when ``None`` they are parsed via CLI.
    :param training_args: Optional GRPO training configuration; parsed when ``None``.
    :param model_args: Optional TRL model configuration; parsed when ``None``.
    :returns: Result of :func:`run_maxent_training` or Hydra entrypoint invocation.
    :raises RuntimeError: If no CLI parser or Hydra entrypoint is available.
    :raises Exception: Propagates parser or training pipeline exceptions.
    """

    ensure_usercustomize_loaded()

    if script_args is None or training_args is None or model_args is None:
        _parse_grpo_args = _resolve_cli_attr("parse_grpo_args")
        _hydra_cli = _resolve_cli_attr("hydra_cli")
        if callable(_parse_grpo_args):
            parse_grpo_args = cast(
                Callable[[], tuple[Any, Any, Any]],
                _parse_grpo_args,
            )
            try:
                script_args, training_args, model_args = parse_grpo_args()
            except (
                ImportError,
                ModuleNotFoundError,
                RuntimeError,
                SystemExit,
                TypeError,
                ValueError,
                AttributeError,
            ):
                if _hydra_cli is not None:
                    maxent_entry = getattr(_hydra_cli, "maxent_entry", None)
                    if callable(maxent_entry):
                        return maxent_entry()
                raise
        elif _hydra_cli is not None:
            maxent_entry = getattr(_hydra_cli, "maxent_entry", None)
            if callable(maxent_entry):
                return maxent_entry()
            raise RuntimeError("Hydra CLI entrypoint is unavailable")
        else:
            raise RuntimeError("No CLI parser available")
    from maxent_grpo.pipelines.training.maxent import run_maxent_training

    return run_maxent_training(script_args, training_args, model_args)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
