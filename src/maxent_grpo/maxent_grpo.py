"""MaxEnt-GRPO training entrypoint."""

from __future__ import annotations

import os
import sys
from pathlib import Path
import signal
import faulthandler

# Allow running this file directly (e.g., accelerate launch src/maxent_grpo/maxent_grpo.py)
# by ensuring the package root is on sys.path.
if __package__ is None or __package__ == "":
    project_src = Path(__file__).resolve().parents[1]
    project_src_str = str(project_src)
    if project_src_str in sys.path:
        sys.path.remove(project_src_str)
    sys.path.insert(0, project_src_str)

from maxent_grpo.cli._test_hooks import ensure_usercustomize_loaded

ensure_usercustomize_loaded()

from maxent_grpo.cli import hydra_cli, parse_grpo_args

__all__ = ["main"]

if os.environ.get("MAXENT_FAULTHANDLER", "").strip():
    try:
        faulthandler.enable(all_threads=True)
    except Exception:
        pass
    try:
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    except Exception:
        pass


def main(script_args=None, training_args=None, model_args=None):
    """Run the MaxEnt trainer when configs are provided, else delegate to Hydra.

    :param script_args: Optional GRPO script arguments; when ``None`` they are parsed via CLI.
    :param training_args: Optional GRPO training configuration; parsed when ``None``.
    :param model_args: Optional TRL model configuration; parsed when ``None``.
    :returns: Result of :func:`run_maxent_training` or Hydra entrypoint invocation.
    """

    if script_args is None or training_args is None or model_args is None:
        try:
            script_args, training_args, model_args = parse_grpo_args()
        except (ImportError, RuntimeError, SystemExit, ValueError):
            return hydra_cli.maxent_entry()
    from maxent_grpo.pipelines.training.maxent import run_maxent_training

    return run_maxent_training(script_args, training_args, model_args)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
