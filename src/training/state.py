# Copyright 2025 Liv d'Aliberti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training loop state helpers for controller and checkpoint management."""

from __future__ import annotations

import logging
import os
from typing import Optional

from .types import (
    Accelerator,
    ControllerPaths,
    LoggingHandles,
    OptimizationSchedule,
    TrainingLoopState,
)
from .weighting import WeightingSettings
from .weighting.logic import CONTROLLER_STATE_FILENAME, load_controller_state


LOG = logging.getLogger(__name__)
_checkpoint_log_once = {"config": False, "strategy": False, "steps": False}


def maybe_clear_stale_controller_state(
    accelerator: Accelerator, controller_cfg: ControllerPaths
) -> None:
    """Delete a stale controller state file when overwriting the output dir.

    :param accelerator: Accelerate handle used to determine the main process
        and trigger ``wait_for_everyone`` guards.
    :type accelerator: accelerate.Accelerator
    :param controller_cfg: Paths describing the active controller
        checkpoint/restore locations.
    :type controller_cfg: training.types.ControllerPaths
    """
    resume_path = getattr(controller_cfg, "resume_from", None)
    if resume_path:
        return
    if not getattr(controller_cfg, "overwrite_existing", False):
        return
    state_path = getattr(controller_cfg, "state_path", None)
    if not state_path or not os.path.isfile(state_path):
        return
    if accelerator.is_main_process:
        try:
            os.remove(state_path)
            LOG.info(
                "Removed stale controller state at %s due to overwrite_output_dir.",
                state_path,
            )
        except OSError as exc:  # pragma: no cover - filesystem race
            LOG.warning(
                "Failed to remove stale controller state %s: %s", state_path, exc
            )
    wait_for_all = getattr(accelerator, "wait_for_everyone", None)
    if callable(wait_for_all):
        wait_for_all()


def _load_controller_file(
    path: Optional[str], accelerator: Accelerator, weighting_cfg: WeightingSettings
) -> bool:
    """Load controller parameters from ``path`` when available.

    :param path: Filesystem path to a serialized controller state.
    :type path: str | None
    :param accelerator: Accelerate handle for logging/synchronization.
    :type accelerator: accelerate.Accelerator
    :param weighting_cfg: Mutable weighting configuration that will receive
        the loaded parameters.
    :type weighting_cfg: training.types.WeightingSettings
    :returns: ``True`` when the controller state was loaded successfully.
    :rtype: bool
    """
    if not path:
        return False
    success = load_controller_state(path, weighting_cfg)
    if success and accelerator.is_main_process:
        LOG.info(
            "Loaded controller state from %s | beta=%.6f tau=%.6f",
            path,
            weighting_cfg.beta,
            weighting_cfg.tau,
        )
    return success


def load_controller_state_chain(
    controller_cfg: ControllerPaths,
    accelerator: Accelerator,
    weighting_cfg: WeightingSettings,
) -> bool:
    """Attempt to load controller state from resume directory or the current state.

    :param controller_cfg: Filesystem paths for controller checkpoints.
    :type controller_cfg: training.types.ControllerPaths
    :param accelerator: Accelerate handle performing logging/synchronization.
    :type accelerator: accelerate.Accelerator
    :param weighting_cfg: Mutable weighting settings that receive the loaded parameters.
    :type weighting_cfg: training.types.WeightingSettings
    :returns: ``True`` when a checkpoint was successfully loaded.
    :rtype: bool
    """
    maybe_clear_stale_controller_state(accelerator, controller_cfg)
    resume_path = getattr(controller_cfg, "resume_from", None)
    controller_loaded = False
    if isinstance(resume_path, str) and resume_path:
        resume_state_file = os.path.join(resume_path, CONTROLLER_STATE_FILENAME)
        controller_loaded = _load_controller_file(
            resume_state_file, accelerator, weighting_cfg
        )
    if not controller_loaded:
        controller_loaded = _load_controller_file(
            controller_cfg.state_path, accelerator, weighting_cfg
        )
    return controller_loaded


def maybe_load_accelerator_state(
    resume_state_path: Optional[str], accelerator: Accelerator
) -> None:
    """Load an accelerator state directory when resuming if available.

    :param resume_state_path: Filesystem path to an accelerator state directory
        (e.g., saved by ``accelerator.save_state``).
    :type resume_state_path: str | None
    :param accelerator: Accelerate handle whose ``load_state`` method will be invoked.
    :type accelerator: accelerate.Accelerator
    """
    load_state_fn = getattr(accelerator, "load_state", None)
    if (
        isinstance(resume_state_path, str)
        and resume_state_path
        and os.path.isdir(resume_state_path)
        and callable(load_state_fn)
    ):
        try:
            load_state_fn(resume_state_path)
            accelerator.wait_for_everyone()
            LOG.info("Loaded accelerator state from %s", resume_state_path)
        except OSError as exc:  # pragma: no cover - environment dependent
            LOG.warning(
                "Failed to load accelerator state from %s: %s", resume_state_path, exc
            )


def maybe_checkpoint(
    logging_cfg: LoggingHandles, accelerator: Accelerator, global_step: int
) -> None:
    """Checkpoint periodically while on the main process.

    :param logging_cfg: Logging handles containing checkpoint callbacks and
        scheduling knobs.
    :type logging_cfg: training.types.LoggingHandles
    :param accelerator: Accelerate handle used for synchronization and
        main-process checks.
    :type accelerator: accelerate.Accelerator
    :param global_step: Current optimizer step; used to decide whether
        ``save_steps`` divides the step index evenly.
    :type global_step: int
    """
    if accelerator.is_main_process and not _checkpoint_log_once["config"]:
        LOG.info(
            "Checkpoint guard | strategy=%s | save_steps=%s",
            getattr(logging_cfg, "save_strategy", None),
            getattr(logging_cfg, "save_steps", None),
        )
        _checkpoint_log_once["config"] = True
    strategy = str(getattr(logging_cfg, "save_strategy", "")).lower()
    for prefix in ("savestrategy.", "intervalstrategy."):
        if strategy.startswith(prefix):
            strategy = strategy.split(".", 1)[1]
    should_save = (
        strategy == "steps"
        and logging_cfg.save_steps > 0
        and global_step % logging_cfg.save_steps == 0
    )
    if accelerator.is_main_process:
        if strategy != "steps":
            if not _checkpoint_log_once["strategy"]:
                LOG.info(
                    "Skipping checkpoint: save_strategy=%s (global_step=%s)",
                    strategy,
                    global_step,
                )
                _checkpoint_log_once["strategy"] = True
        elif logging_cfg.save_steps <= 0:
            if not _checkpoint_log_once["steps"]:
                LOG.info(
                    "Skipping checkpoint: save_steps<=0 (save_steps=%s | global_step=%s)",
                    logging_cfg.save_steps,
                    global_step,
                )
                _checkpoint_log_once["steps"] = True
    wait_for_all = getattr(accelerator, "wait_for_everyone", None)
    if callable(wait_for_all):
        wait_for_all()
    if should_save:
        if accelerator.is_main_process:
            LOG.info(
                "Triggering checkpoint save at step %s (save_steps=%s)",
                global_step,
                logging_cfg.save_steps,
            )
        logging_cfg.save_checkpoint(f"checkpoint-{global_step}")
    if callable(wait_for_all):
        wait_for_all()


def check_stop_condition(
    schedule: OptimizationSchedule, state: "TrainingLoopState"
) -> None:
    """Set stop flag when the configured number of steps is reached.

    :param schedule: Optimization schedule describing ``total_training_steps``.
    :type schedule: training.types.OptimizationSchedule
    :param state: Mutable training loop state whose ``stop_training`` flag
        should be updated when the threshold is crossed.
    :type state: training.loop.TrainingLoopState
    """
    if (
        schedule.total_training_steps > 0
        and state.global_step >= schedule.total_training_steps
    ):
        state.stop_training = True


__all__ = [
    "maybe_clear_stale_controller_state",
    "load_controller_state_chain",
    "maybe_load_accelerator_state",
    "maybe_checkpoint",
    "check_stop_condition",
]
