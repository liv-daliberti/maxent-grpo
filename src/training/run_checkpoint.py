"""Compatibility checkpoint helpers retained for legacy imports in tests."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Any

from .run_types import CheckpointConfig
from .zero_utils import _maybe_zero_gather_params


@dataclass
class CheckpointHandles:
    """Container for accelerator/model/tokenizer used during saves."""

    accelerator: Any
    model: Any
    tokenizer: Any


def _should_skip_accelerator_state(accelerator: any) -> bool:
    """Respect environment flags controlling DeepSpeed state saves."""
    force_skip = (
        os.environ.get("MAXENT_FORCE_SKIP_DEEPSPEED_STATE_SAVE", "").lower() == "true"
    )
    if force_skip:
        return True
    allow = os.environ.get("MAXENT_ALLOW_DEEPSPEED_STATE_SAVE", "").lower() == "true"
    skip = os.environ.get("MAXENT_SKIP_DEEPSPEED_STATE_SAVE", "").lower() == "true"
    dist_type = getattr(getattr(accelerator, "state", None), "distributed_type", None)
    if allow:
        return False
    return bool(skip and dist_type == "DEEPSPEED")


def _copy_initial_model_snapshot(target_dir: str) -> None:
    """Copy static model metadata into ``target_dir`` for shallow checkpoints."""
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    source = os.environ.get("MAXENT_CHECKPOINT_METADATA_SOURCE")
    source_path: Optional[Path] = Path(source) if source else None

    if source_path is None or not source_path.is_dir():
        # Heuristic: prefer a sibling GRPO directory when available.
        parent = target_path.parent
        for candidate in parent.iterdir():
            if (
                candidate.is_dir()
                and "GRPO" in candidate.name
                and "MaxEnt" not in candidate.name
            ):
                source_path = candidate
                break

    if source_path is None or not source_path.is_dir():
        return

    for fname in ("config.json", "model.safetensors"):
        src = source_path / fname
        dst = target_path / fname
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)


def _copy_static_checkpoint_files(source_dir: str, target_dir: str) -> None:
    """Copy common static files (config/model) into a target checkpoint dir."""
    _copy_initial_model_snapshot(target_dir)
    # Preserve the trainer config when present.
    cfg_path = Path(source_dir) / "config.json"
    if cfg_path.is_file():
        shutil.copyfile(cfg_path, Path(target_dir) / "config.json")


def _save_training_args(
    output_dir: str, args: Any
) -> None:  # pylint: disable=unused-argument
    """Placeholder that mirrors HF trainer behaviour for tests."""
    # We intentionally avoid serialising; tests patch this hook.
    return None


def _save_state_or_deepspeed_checkpoint(
    accelerator: Any, model: Any, path: str
) -> None:  # pylint: disable=unused-argument
    """Save accelerator/deepspeed state respecting skip flags."""
    if _should_skip_accelerator_state(accelerator):
        return
    prefer_accelerate = (
        os.environ.get("MAXENT_PREFER_ACCELERATE_STATE_SAVE", "").lower() == "true"
    )
    engine = getattr(getattr(accelerator, "state", None), "deepspeed_engine", None)
    if engine is not None and not prefer_accelerate:
        engine.save_checkpoint(path)
        return
    save_state = getattr(accelerator, "save_state", None)
    if callable(save_state):
        save_state(path)


def _save_trainer_state_like_hf(**_kwargs: Any) -> None:
    """Compatibility stub; tests patch this to observe calls."""
    return None


def _prune_checkpoints(output_dir: str, keep_limit: int) -> None:
    """Delete older checkpoints when exceeding the keep limit."""
    if keep_limit <= 0:
        return
    base = Path(output_dir)
    checkpoints = sorted(
        [p for p in base.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime
    )
    excess = len(checkpoints) - keep_limit
    for path in checkpoints[: max(0, excess)]:
        shutil.rmtree(path, ignore_errors=True)


class CheckpointManager:
    """Lightweight wrapper that mirrors the legacy checkpoint interface."""

    def __init__(
        self,
        handles: CheckpointHandles,
        config: CheckpointConfig,
        state: Optional[Any] = None,
    ):
        self.handles = handles
        self.config = config
        self._state = state or SimpleNamespace(training_args=None)

    def _save_model_and_tokenizer(self, path: Path) -> None:
        mode = os.environ.get("MAXENT_CHECKPOINT_METADATA_MODE", "").lower()
        if mode == "shallow":
            return
        unwrap = getattr(self.handles.accelerator, "unwrap_model", None)
        model = unwrap(self.handles.model) if callable(unwrap) else self.handles.model
        if not self.handles.accelerator.is_main_process:
            return
        with _maybe_zero_gather_params(self.handles.model, enabled=True):
            model.save_pretrained(str(path))
            self.handles.tokenizer.save_pretrained(str(path))

    def _save_accelerator_state(self, path: Path) -> None:
        accel = self.handles.accelerator
        _save_state_or_deepspeed_checkpoint(accel, self.handles.model, str(path))

    def save(self, label: str) -> None:
        accel = self.handles.accelerator
        out_dir = Path(self.config.output_dir) / label
        out_dir.mkdir(parents=True, exist_ok=True)
        accel.wait_for_everyone()
        self._save_accelerator_state(out_dir)
        accel.wait_for_everyone()
        mode = os.environ.get("MAXENT_CHECKPOINT_METADATA_MODE", "").lower()
        if mode == "shallow":
            _copy_static_checkpoint_files(self.config.output_dir, str(out_dir))
            if self._state.training_args is not None:
                _save_training_args(str(out_dir), self._state.training_args)
        else:
            self._save_model_and_tokenizer(out_dir)
        if accel.is_main_process and self.config.save_total_limit > 0:
            _prune_checkpoints(self.config.output_dir, self.config.save_total_limit)

    def finalize(self) -> None:
        accel = self.handles.accelerator
        out_dir = Path(self.config.output_dir)
        accel.wait_for_everyone()
        self._save_accelerator_state(out_dir)
        accel.wait_for_everyone()
        mode = os.environ.get("MAXENT_CHECKPOINT_METADATA_MODE", "").lower()
        if mode == "shallow":
            _copy_static_checkpoint_files(self.config.output_dir, str(out_dir))
            if self._state.training_args is not None:
                _save_training_args(str(out_dir), self._state.training_args)
        else:
            self._save_model_and_tokenizer(out_dir)


def finalize_training(
    accelerator: any, model: any, tokenizer: any, config: CheckpointConfig
) -> None:
    """Finalize training by saving accelerator/model/tokenizer."""
    mgr = CheckpointManager(CheckpointHandles(accelerator, model, tokenizer), config)
    mgr.finalize()


__all__ = [
    "CheckpointHandles",
    "CheckpointManager",
    "_copy_initial_model_snapshot",
    "_copy_static_checkpoint_files",
    "_should_skip_accelerator_state",
    "_maybe_zero_gather_params",
    "_save_training_args",
    "_save_state_or_deepspeed_checkpoint",
    "_save_trainer_state_like_hf",
    "_prune_checkpoints",
    "finalize_training",
]
