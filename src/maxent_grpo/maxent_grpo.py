"""GRPO training entrypoint (supports entropy bonus and MaxEnt weighting)."""

from __future__ import annotations

import importlib
import os
import sys
import logging
import atexit
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

_fh_requested = bool(os.environ.get("MAXENT_FAULTHANDLER", "").strip())
_stack_dump_raw = os.environ.get("MAXENT_STACK_DUMP_S", "").strip()
if _stack_dump_raw:
    _fh_requested = True

if _fh_requested:
    try:
        faulthandler.enable(all_threads=True)
    except (OSError, RuntimeError, ValueError) as exc:
        LOG.warning("Failed to enable faulthandler: %s", exc)
    if hasattr(signal, "SIGUSR1"):
        try:
            faulthandler.register(signal.SIGUSR1, all_threads=True)
        except (OSError, RuntimeError, ValueError) as exc:
            LOG.warning("Failed to register faulthandler SIGUSR1 handler: %s", exc)
    if _stack_dump_raw:
        try:
            interval = float(_stack_dump_raw)
        except (TypeError, ValueError):
            interval = 0.0
        if interval > 0:
            try:
                faulthandler.dump_traceback_later(
                    interval, repeat=True, all_threads=True
                )
            except TypeError:
                # Python < 3.12 does not accept all_threads for dump_traceback_later.
                faulthandler.dump_traceback_later(interval, repeat=True)
            except (OSError, RuntimeError, ValueError) as exc:
                LOG.warning("Failed to schedule periodic stack dumps: %s", exc)
            else:
                atexit.register(faulthandler.cancel_dump_traceback_later)


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
    """Run the GRPO/MaxEnt trainer, parsing TRL-style configs when needed.

    :param script_args: Optional GRPO script arguments; when ``None`` they are parsed via CLI.
    :param training_args: Optional GRPO training configuration; parsed when ``None``.
    :param model_args: Optional TRL model configuration; parsed when ``None``.
    :returns: Result of :func:`run_maxent_training`.
    :raises RuntimeError: If no CLI parser is available.
    :raises Exception: Propagates parser or training pipeline exceptions.
    """

    ensure_usercustomize_loaded()

    if script_args is None or training_args is None or model_args is None:
        _parse_grpo_args = _resolve_cli_attr("parse_grpo_args")
        if callable(_parse_grpo_args):
            parse_grpo_args = cast(
                Callable[[], tuple[Any, Any, Any]],
                _parse_grpo_args,
            )
            script_args, training_args, model_args = parse_grpo_args()
        else:
            raise RuntimeError(
                "No CLI parser available. Ensure TRL is installed and "
                "maxent_grpo.cli.parse_grpo_args is importable."
            )
    from maxent_grpo.pipelines.training.maxent import run_maxent_training

    return run_maxent_training(script_args, training_args, model_args)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
