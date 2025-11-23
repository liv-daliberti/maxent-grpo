"""
MaxEnt‑GRPO: sequence‑level maximum‑entropy variant of GRPO.

This training entrypoint implements a lightweight loop that realizes the
per‑context maximum‑entropy update at the sequence level. For each prompt we:
1) Generate ``K`` completions (via vLLM ``/generate`` when enabled, or local
   ``model.generate``).
2) Convert per‑sequence utilities (rewards) into a listwise distribution ``q``
   using a temperature and epsilon floor for full support.
3) Compute sequence log‑probs under a frozen reference model.
4) Form per‑sequence weights ``w_i ∝ q_i^{1/(τ+β)} · π_ref(i)^{β/(τ+β)}`` and
   normalize within each prompt group.
5) Apply a weighted MLE update (no explicit KL term required).

Key pieces
- ``_to_prompt``: Local copy of the prompt builder to avoid circular imports.
- ``MaxEntOptions``: Environment‑driven knobs (τ, q temperature/epsilon, length
  normalization) for convenience.
- ``_group_softmax``, ``_prepare_labels_for_ce``, ``_sequence_logprobs``,
  ``_batch_tokenize_pairs``: Helpers for weighting and scoring sequences.
- ``main``: End‑to‑end training loop using ``utils.*`` helpers and
  ``utils.vllm_patch.safe_generate`` when vLLM is enabled.

This is intentionally minimal for readability and prototyping. For production
features (controllers, schedulers, DDP), prefer the TRL trainer path in
``src/grpo.py`` and extend it.

License
Copyright 2025 Liv d'Aliberti

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import logging
from typing import Any

import os

from configs import GRPOConfig, GRPOScriptArguments
from utils.trl_patches import ensure_vllm_group_port
from .run_checkpoint import (
    CheckpointHandles,
    CheckpointManager,
    CheckpointState,
    _copy_initial_model_snapshot,
)
from .run_setup import bootstrap_runner, build_training_components
from .run_types import RunnerSetup, TrainingComponents
from .run_helpers import _log_wandb, _maybe_init_wandb_run
from .run_training_loop import run_training_loop
from .run_training_types import (
    ControllerPaths,
    LoggingHandles,
    LoopSettings,
    RuntimeHandles,
    TrainingLoopContext,
)
from .run_training_weighting import CONTROLLER_STATE_FILENAME

LOG = logging.getLogger(__name__)

# Apply TRL compatibility patch eagerly so any downstream usage of VLLMClient
# inherits the environment-driven group_port override.
ensure_vllm_group_port()


def _log_plan(setup: RunnerSetup, components: TrainingComponents) -> None:
    """Log the approximate schedule for the upcoming training job."""
    accelerator = setup.accelerator
    if not accelerator.is_main_process:
        return
    steps_display = (
        str(setup.train_data.steps_per_epoch)
        if setup.train_data.steps_per_epoch is not None
        else "unknown"
    )
    LOG.info(
        "Plan: epochs=%d | steps_per_epoch≈%s | grad_accum=%d | total_optimizer_steps=%d",
        setup.hyperparams.num_epochs,
        steps_display,
        setup.hyperparams.grad_accum_steps,
        components.optimization.schedule.total_training_steps,
    )


def _build_logging_handles(
    setup: RunnerSetup,
    checkpoint_manager: CheckpointManager,
    wandb_run: Any,
) -> LoggingHandles:
    """Construct logging callbacks consumed by the training loop."""
    return LoggingHandles(
        log_metrics=lambda metrics, step: _log_wandb(wandb_run, metrics, step),
        save_checkpoint=checkpoint_manager.save,
        save_strategy=setup.checkpoint.save_strategy,
        save_steps=setup.checkpoint.save_steps,
        wandb_run=wandb_run,
    )


def _sync_report_to_from_script_args(
    script_args: GRPOScriptArguments,
    training_args: GRPOConfig,
) -> None:
    """Backfill training_args.report_to when only script args set it."""

    def _is_effectively_none(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip().lower() in {"", "none"}
        try:
            entries = list(value)
        except TypeError:
            return False
        if not entries:
            return True
        lowered = [str(entry).strip().lower() for entry in entries]
        return all(entry in {"", "none"} for entry in lowered)

    script_report_to = getattr(script_args, "report_to", None)
    if _is_effectively_none(getattr(training_args, "report_to", None)) and not _is_effectively_none(
        script_report_to
    ):
        setattr(training_args, "report_to", script_report_to)


def run_maxent_grpo(
    script_args: GRPOScriptArguments,
    training_args: GRPOConfig,
    model_args: Any,
) -> None:
    """Run the lightweight MaxEnt-GRPO trainer."""
    _sync_report_to_from_script_args(script_args, training_args)
    setup = bootstrap_runner(script_args, training_args, model_args)
    components = build_training_components(setup, script_args, training_args, model_args)

    # Pre-populate the output directory with a base snapshot (config/tokenizer
    # files and optionally model weights) before training starts. This allows
    # consumers to treat <output_dir> as a valid HF checkpoint even before the
    # first optimizer step. The source is controlled via
    # MAXENT_CHECKPOINT_METADATA_SOURCE (or defaults to output_dir).
    _maybe_wait_for_all(setup.accelerator)
    if setup.accelerator.is_main_process:
        _copy_initial_model_snapshot(setup.checkpoint.output_dir)
    _maybe_wait_for_all(setup.accelerator)

    _log_plan(setup, components)

    wandb_run = _maybe_init_wandb_run(
        setup.accelerator,
        training_args,
        components.wandb_config,
    )
    if wandb_run is not None and setup.accelerator.is_main_process:
        log_interval = int(getattr(training_args, "logging_steps", 1) or 1)
        run_label = getattr(wandb_run, "name", None) or getattr(wandb_run, "id", "unknown")
        project_name = (
            getattr(wandb_run, "project", None)
            or os.environ.get("WANDB_PROJECT")
            or "default"
        )
        LOG.info(
            "Initialized W&B run '%s' | project=%s | metrics log every %d optimizer "
            "steps (first log at step %d).",
            run_label,
            project_name,
            log_interval,
            log_interval,
        )
    checkpoint_handles = CheckpointHandles(
        accelerator=setup.accelerator,
        model=setup.model_bundle.model,
        tokenizer=setup.model_bundle.tokenizer,
    )
    checkpoint_manager = CheckpointManager(
        handles=checkpoint_handles,
        cfg=setup.checkpoint,
        state=CheckpointState(
            training_args=training_args,
            lr_scheduler=components.optimization.handles.lr_scheduler,
        ),
    )
    # Record the resolved checkpointing plan on the main process to make save triggers transparent.
    if setup.accelerator.is_main_process:
        LOG.info(
            "Checkpointing config | strategy=%s | save_steps=%s | "
            "save_total_limit=%s | output_dir=%s",
            setup.checkpoint.save_strategy,
            setup.checkpoint.save_steps,
            setup.checkpoint.save_total_limit,
            setup.checkpoint.output_dir,
        )
        LOG.info(
            "Logging config | strategy=%s | logging_steps=%s",
            getattr(training_args, "logging_strategy", None),
            getattr(training_args, "logging_steps", None),
        )
    logging_handles = _build_logging_handles(setup, checkpoint_manager, wandb_run)

    total_steps = components.optimization.schedule.total_training_steps
    warmup_steps = components.lr_warmup_steps
    if setup.accelerator.is_main_process and total_steps > 0:
        LOG.info(
            "Learning-rate schedule | total_optimizer_steps=%d | warmup_steps=%d (%.1f%%).",
            total_steps,
            warmup_steps,
            100.0 * float(warmup_steps) / float(max(total_steps, 1)),
        )
    if setup.accelerator.is_main_process and getattr(training_args, "logging_first_step", False):
        logging_handles.log_metrics({"train/learning_rate": 0.0}, 0)

    loop_settings = LoopSettings(
        generation=components.generation,
        evaluation=components.evaluation,
        optimization=components.optimization,
        scoring=components.scoring,
        controller=ControllerPaths(
            state_path=os.path.join(setup.checkpoint.output_dir, CONTROLLER_STATE_FILENAME),
            resume_from=getattr(training_args, "resume_from_checkpoint", None),
            overwrite_existing=bool(getattr(training_args, "overwrite_output_dir", False)),
        ),
    )

    run_training_loop(
        TrainingLoopContext(
            runtime=RuntimeHandles(
                accelerator=setup.accelerator,
                model=setup.model_bundle.model,
                tokenizer=setup.model_bundle.tokenizer,
                train_loader=setup.train_data.train_loader,
                train_sampler=setup.train_data.train_sampler,
                device=setup.accelerator.device,
                get_ref_model=setup.model_bundle.get_ref_model,
            ),
            reward=components.reward,
            settings=loop_settings,
            logging=logging_handles,
        )
    )
    checkpoint_manager.finalize()
def _maybe_wait_for_all(accelerator: Any) -> None:
    """Synchronize all ranks when Accelerate exposes wait_for_everyone."""
    wait_fn = getattr(accelerator, "wait_for_everyone", None)
    if callable(wait_fn):
        wait_fn()
