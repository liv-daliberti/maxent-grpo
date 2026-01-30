"""InfoSeed-GRPO training entrypoint using the custom training loop."""

from __future__ import annotations

import logging
from typing import Any

from maxent_grpo.config import GRPOConfig, GRPOScriptArguments
from maxent_grpo.training import run_training_loop
from maxent_grpo.utils.deps_guard import ensure_real_dependencies
from .loop_common import build_training_loop_context

LOG = logging.getLogger(__name__)


def run_infoseed_training(
    script_args: GRPOScriptArguments,
    training_args: GRPOConfig,
    model_args: "Any",
) -> None:
    """Run InfoSeed-GRPO training via the custom loop.

    :param script_args: Script configuration including dataset and rewards.
    :type script_args: GRPOScriptArguments
    :param training_args: Training configuration (InfoSeed knobs enabled).
    :type training_args: GRPOConfig
    :param model_args: Model configuration forwarded to the loop context.
    :type model_args: Any
    :returns: ``None``. Side effects include training and checkpointing.
    :rtype: None
    """

    ensure_real_dependencies(context="InfoSeed-GRPO training")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )
    LOG.setLevel(training_args.get_process_log_level())
    LOG.info(
        "Starting InfoSeed-GRPO training | seeds=%s | lambda=%s",
        getattr(training_args, "info_seed_num_seeds", 0),
        getattr(training_args, "info_seed_lambda", 0.0),
    )
    ctx = build_training_loop_context(
        script_args,
        training_args,
        model_args,
        deps_namespace="infoseed",
        apply_info_seed_cfg=True,
        force_grpo_objective=True,
    )
    run_training_loop(ctx)


__all__ = ["run_infoseed_training"]
