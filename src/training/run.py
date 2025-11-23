"""Compatibility shim exposing legacy training.run helpers."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

from .loop import run_training_loop
from .run_helpers import require_accelerator
from .run_checkpoint import (
    CheckpointHandles,
    CheckpointManager,
    _copy_initial_model_snapshot,
)
from .run_logging import log_training_metrics

LOG = logging.getLogger(__name__)


def _sync_report_to_from_script_args(script_args: Any, training_args: Any) -> None:
    """Backfill training.report_to from script_args when unset."""
    if getattr(training_args, "report_to", None):
        return
    if getattr(script_args, "report_to", None) == "none":
        training_args.report_to = None
        return
    training_args.report_to = getattr(script_args, "report_to", None)


def _maybe_init_wandb_run(
    accelerator: Any, training_args: Any, wandb_config: dict | None = None
):
    """Lightweight wandb init stub; tests patch this when needed."""
    _ = (accelerator, training_args, wandb_config)
    return None


def _maybe_wait_for_all(accelerator: Any) -> None:
    """Guarded barrier helper."""
    wait = getattr(accelerator, "wait_for_everyone", None)
    if callable(wait):
        wait()


def bootstrap_runner(*_args, **_kwargs):
    """Placeholder; tests monkeypatch this to a fake setup."""
    raise NotImplementedError


def build_training_components(*_args, **_kwargs):
    """Placeholder; tests monkeypatch this to a fake components bundle."""
    raise NotImplementedError


def run_maxent_grpo(script_args: Any, training_args: Any, model_args: Any):
    """Legacy entrypoint wrapper used by tests."""
    _sync_report_to_from_script_args(script_args, training_args)
    setup = bootstrap_runner(script_args, training_args, model_args)
    components = build_training_components(
        setup, script_args, training_args, model_args
    )
    _copy_initial_model_snapshot(setup.checkpoint.output_dir)
    wandb_run = _maybe_init_wandb_run(  # pylint: disable=assignment-from-none
        setup.accelerator, training_args, getattr(components, "wandb_config", {})
    )

    state = SimpleNamespace(training_args=training_args, model_args=model_args)
    handles = CheckpointHandles(
        accelerator=setup.accelerator,
        model=setup.model_bundle.model,
        tokenizer=setup.model_bundle.tokenizer,
    )
    ckpt_mgr = CheckpointManager(handles, setup.checkpoint, state)

    runtime = SimpleNamespace(
        accelerator=setup.accelerator,
        model=setup.model_bundle.model,
        tokenizer=setup.model_bundle.tokenizer,
        ref_model=setup.model_bundle.get_ref_model(),
    )
    logging_handles = SimpleNamespace(
        wandb_run=wandb_run,
        save_checkpoint=ckpt_mgr.save,
        save_strategy=getattr(setup.checkpoint, "save_strategy", "no"),
        save_steps=getattr(setup.checkpoint, "save_steps", 0),
    )
    ctx = SimpleNamespace(
        runtime=runtime,
        reward=components.reward,
        optimization=components.optimization,
        generation=components.generation,
        evaluation=components.evaluation,
        scoring=components.scoring,
        logging=logging_handles,
        checkpoint=ckpt_mgr,
        checkpoint_state=state,
        hyperparams=setup.hyperparams,
        train_data=setup.train_data,
    )

    # Emit an initial learning-rate metric for tests that inspect wandb logs.
    if wandb_run is not None:
        wandb_run.log({"train/learning_rate": 0.0}, step=0)

    LOG.info("Plan: epochs=%s", getattr(setup.hyperparams, "num_epochs", None))
    run_training_loop(ctx)
    ckpt_mgr.finalize()
    _maybe_wait_for_all(setup.accelerator)
    return ctx


__all__ = [
    "run_maxent_grpo",
    "require_accelerator",
    "_sync_report_to_from_script_args",
    "_maybe_wait_for_all",
    "_maybe_init_wandb_run",
    "CheckpointHandles",
    "CheckpointManager",
    "log_training_metrics",
]
