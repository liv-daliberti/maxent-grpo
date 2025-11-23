"""Checkpoint management helpers for the MaxEnt-GRPO runner."""

from __future__ import annotations

import json
import logging
import os
import random
import re
import shutil
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Optional, Tuple

from .run_types import CheckpointConfig
from .zero_utils import _maybe_zero_gather_params

LOG = logging.getLogger(__name__)

try:  # Optional dependency in some test environments
    import torch  # type: ignore
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
    torch = None  # type: ignore

try:
    import numpy as np  # type: ignore
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
    np = None  # type: ignore

try:
    from accelerate.utils import DistributedType  # type: ignore
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
    DistributedType = None  # type: ignore

try:
    import requests  # type: ignore
    from requests import RequestException as _RequestException  # type: ignore
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
    requests = None  # type: ignore

    class _RequestException(Exception):  # type: ignore
        """Fallback exception class used when requests is unavailable."""

RequestException = _RequestException


@dataclass
class CheckpointState:
    """Container for auxiliary training artifacts needed during checkpointing."""

    training_args: Optional[Any] = None
    lr_scheduler: Optional[Any] = None


@dataclass(frozen=True)
class CheckpointHandles:
    """Bundle of runtime handles needed for checkpointing."""

    accelerator: Any
    model: Any
    tokenizer: Any


def _hub_error_types() -> Tuple[type, ...]:
    """Return a tuple of Hub-related exception types for defensive catches."""
    base_errors: Tuple[type, ...] = (OSError, ValueError, RuntimeError)
    try:
        hub_module = __import__("huggingface_hub.utils", fromlist=["dummy"])
    except ModuleNotFoundError:
        return base_errors
    hub_error = getattr(hub_module, "HfHubHTTPError", None)
    if isinstance(hub_error, type) and issubclass(hub_error, BaseException):
        return base_errors + (hub_error,)
    return base_errors


def _checkpoint_metadata_mode() -> str:
    """Return how aggressively to write HF-style checkpoint metadata."""
    mode = os.environ.get("MAXENT_CHECKPOINT_METADATA_MODE", "").strip().lower()
    if mode in {"", "full"}:
        return "full"
    if mode in {"shallow", "light", "ds_only", "deepspeed_only", "final_only"}:
        # Treat any non-full mode as "avoid heavy save_pretrained on periodic checkpoints".
        return "shallow"
    return "full"


def _resolve_metadata_source(default_root: str) -> str:
    """Resolve the directory to use as the source for static checkpoint metadata.

    Priority:
    1) MAXENT_CHECKPOINT_METADATA_SOURCE when it points to an existing directory.
    2) For MaxEnt runs, a sibling GRPO directory inferred from the output_dir
       (e.g., Qwen2.5-1.5B-Open-R1-MaxEnt-GRPO-... → Qwen2.5-1.5B-Open-R1-GRPO-...).
    3) The default_root itself.
    """
    explicit = os.environ.get("MAXENT_CHECKPOINT_METADATA_SOURCE")
    if explicit and os.path.isdir(explicit):
        return explicit
    if not default_root:
        return default_root
    root_dir, leaf = os.path.split(default_root)
    candidates = []
    if "MaxEnt-GRPO" in leaf:
        candidates.append(os.path.join(root_dir, leaf.replace("MaxEnt-GRPO-", "GRPO-")))
        candidates.append(os.path.join(root_dir, leaf.replace("-MaxEnt-GRPO", "-GRPO")))
    for cand in candidates:
        if os.path.isdir(cand):
            return cand
    return default_root


def _copy_static_checkpoint_files(source_root: str, target_root: str) -> None:
    """Best-effort copy of small, static HF metadata files into a checkpoint dir.

    This is used when we rely on DeepSpeed + zero_to_fp32 for the actual weights
    but still want config/tokenizer metadata alongside the ZeRO shards.
    """
    meta_source = _resolve_metadata_source(source_root)
    if not meta_source or not os.path.isdir(meta_source):
        return
    static_files = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
    ]
    for filename in static_files:
        src = os.path.join(meta_source, filename)
        dst = os.path.join(target_root, filename)
        if not os.path.isfile(src):
            continue
        if os.path.isfile(dst):
            continue
        try:
            shutil.copy2(src, dst)
        except OSError as exc:  # pragma: no cover - filesystem dependent
            LOG.warning("Failed to copy static checkpoint file %s -> %s: %s", src, dst, exc)


def _copy_initial_model_snapshot(target_root: str) -> None:
    """Best-effort copy of a base HF-style snapshot into the output root.

    This runs once before training starts so that <output_dir> already
    contains config/tokenizer files (and optionally the base model weights)
    even before the first optimizer step.
    """
    base_source = _resolve_metadata_source(target_root)
    if not base_source or not os.path.isdir(base_source):
        return
    initial_files = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
        "model.safetensors",
        "zero_to_fp32.py",
    ]
    for filename in initial_files:
        src = os.path.join(base_source, filename)
        dst = os.path.join(target_root, filename)
        if not os.path.isfile(src):
            continue
        if os.path.isfile(dst):
            continue
        try:
            shutil.copy2(src, dst)
        except OSError as exc:  # pragma: no cover - filesystem dependent
            LOG.warning("Failed to copy initial model file %s -> %s: %s", src, dst, exc)


class CheckpointManager:
    """Manage checkpoint rotation and optional Hub pushes."""

    def __init__(
        self,
        handles: CheckpointHandles,
        cfg: CheckpointConfig,
        state: Optional[CheckpointState] = None,
    ) -> None:
        self.accelerator = handles.accelerator
        self.model = handles.model
        self.tokenizer = handles.tokenizer
        self.cfg = cfg
        self._state = state or CheckpointState()
        self._saved: Deque[str] = deque()
        self._hub_errors = _hub_error_types()
        # Keep training args reachable from the config for finalize_training.
        if self._state.training_args is not None:
            try:
                setattr(self.cfg, "training_args", self._state.training_args)
            except (AttributeError, TypeError):
                LOG.debug(
                    "Unable to attach training_args to checkpoint config; "
                    "continuing without it."
                )

    def save(self, step_label: str) -> None:
        """Persist a checkpoint and optionally prune older ones."""
        accelerator = self.accelerator
        wait_for_all = getattr(accelerator, "wait_for_everyone", None)
        if callable(wait_for_all):
            wait_for_all()
        ckpt_dir = os.path.join(self.cfg.output_dir, step_label)
        os.makedirs(ckpt_dir, exist_ok=True)
        if accelerator.is_main_process:
            LOG.info("Saving checkpoint to %s", ckpt_dir)
        _save_state_or_deepspeed_checkpoint(accelerator, self.model, ckpt_dir)
        if callable(wait_for_all):
            wait_for_all()
        unwrapped: Optional[Any] = None
        mode = _checkpoint_metadata_mode()
        if mode == "full":
            unwrap_model = getattr(accelerator, "unwrap_model", None)
            unwrapped = unwrap_model(self.model) if callable(unwrap_model) else self.model
            with _maybe_zero_gather_params(unwrapped, True):
                if accelerator.is_main_process:
                    unwrapped.save_pretrained(ckpt_dir)
            if accelerator.is_main_process:
                _save_generation_config(unwrapped, ckpt_dir)
                if hasattr(self.tokenizer, "save_pretrained"):
                    self.tokenizer.save_pretrained(ckpt_dir)
                _save_training_args(ckpt_dir, self._state.training_args)
        else:
            # Light-weight periodic checkpoint: rely on DeepSpeed shards plus
            # static metadata copied from a base snapshot.
            if accelerator.is_main_process:
                _copy_static_checkpoint_files(self.cfg.output_dir, ckpt_dir)
                _save_training_args(ckpt_dir, self._state.training_args)
        _save_trainer_state_like_hf(
            output_dir=ckpt_dir,
            step_label=step_label,
            accelerator=accelerator,
            lr_scheduler=self._state.lr_scheduler,
        )
        if not accelerator.is_main_process:
            return
        self._saved.append(ckpt_dir)
        self._prune_old_checkpoints()
        if unwrapped is not None:
            self._maybe_push(unwrapped, f"checkpoint {step_label}")
        _maybe_trigger_vllm_snapshot(ckpt_dir, step_label, source="checkpoint")

    def finalize(self) -> None:
        """Store the last checkpoint and push if requested."""
        finalize_training(
            self.accelerator,
            self.model,
            self.tokenizer,
            self.cfg,
            hub_errors=self._hub_errors,
        )

    def _prune_old_checkpoints(self) -> None:
        while self.cfg.save_total_limit > 0 and len(self._saved) > self.cfg.save_total_limit:
            old_ckpt = self._saved.popleft()
            try:
                shutil.rmtree(old_ckpt)
            except OSError as exc:
                LOG.warning("Failed to delete old checkpoint %s: %s", old_ckpt, exc)

    def _maybe_push(self, model: Any, commit_message: str) -> None:
        if not (self.cfg.hub.enabled and self.cfg.hub.model_id):
            return
        try:
            model.push_to_hub(
                repo_id=self.cfg.hub.model_id,
                commit_message=commit_message,
                token=self.cfg.hub.token,
            )
        except self._hub_errors as exc:  # pragma: no cover - best effort push
            LOG.warning("Failed to push checkpoint %s: %s", commit_message, exc)


def finalize_training(
    accelerator: Any,
    model: Any,
    tokenizer: Any,
    checkpoint_cfg: CheckpointConfig,
    hub_errors: Optional[Tuple[type, ...]] = None,
) -> None:
    """Save the final checkpoint + tokenizer (and optionally push to hub)."""
    wait_for_all = getattr(accelerator, "wait_for_everyone", None)
    if callable(wait_for_all):
        wait_for_all()
    output_dir = checkpoint_cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    if accelerator.is_main_process:
        LOG.info("Saving final checkpoint to %s", output_dir)
    _save_state_or_deepspeed_checkpoint(accelerator, model, output_dir)
    if callable(wait_for_all):
        wait_for_all()
    unwrap_model = getattr(accelerator, "unwrap_model", None)
    unwrapped = unwrap_model(model) if callable(unwrap_model) else model
    with _maybe_zero_gather_params(unwrapped, True):
        if accelerator.is_main_process:
            unwrapped.save_pretrained(output_dir)
    if accelerator.is_main_process:
        _save_generation_config(unwrapped, output_dir)
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(output_dir)
        _save_training_args(output_dir, getattr(checkpoint_cfg, "training_args", None))
        _maybe_trigger_vllm_snapshot(output_dir, "final", source="finalize")
        if checkpoint_cfg.hub.enabled and checkpoint_cfg.hub.model_id:
            if hub_errors is None:
                hub_errors = _hub_error_types()
            try:
                unwrapped.push_to_hub(
                    repo_id=checkpoint_cfg.hub.model_id,
                    commit_message="final checkpoint",
                    token=checkpoint_cfg.hub.token,
                )
            except hub_errors as exc:  # pragma: no cover - best effort push
                LOG.warning("Failed to push final checkpoint to Hub: %s", exc)


def _process_index(accelerator: Any) -> int:
    """Best-effort process index lookup matching HF Trainer conventions."""
    for attr in ("process_index", "local_process_index"):
        if hasattr(accelerator, attr):
            try:
                return int(getattr(accelerator, attr))
            except (TypeError, ValueError):
                continue
    state = getattr(accelerator, "state", None)
    for attr in ("process_index", "local_process_index"):
        if hasattr(state, attr):
            try:
                return int(getattr(state, attr))
            except (TypeError, ValueError):
                continue
    return 0


def _save_rng_state(output_dir: str, accelerator: Any) -> None:
    """Persist RNG state in a HF Trainer compatible format."""
    if torch is None:
        LOG.debug("Torch not available; skipping RNG state save.")
        return
    torch_random = getattr(torch, "random", None)
    get_rng_state = (
        getattr(torch_random, "get_rng_state", None)
        if torch_random is not None
        else None
    )
    if not callable(get_rng_state):
        LOG.debug("Torch RNG helpers unavailable; skipping RNG state save.")
        return
    state = {
        "python": random.getstate(),
        "torch": get_rng_state(),
    }
    if np is not None:
        state["numpy"] = np.random.get_state()
    torch_cuda = getattr(torch, "cuda", None)
    is_available = getattr(torch_cuda, "is_available", None)
    if callable(is_available) and is_available():
        get_cuda_state = getattr(torch_cuda, "get_rng_state_all", None)
        if callable(get_cuda_state):
            state["cuda"] = get_cuda_state()
    save_path = os.path.join(output_dir, f"rng_state_{_process_index(accelerator)}.pth")
    try:
        torch.save(state, save_path)
    except OSError as exc:  # pragma: no cover - filesystem dependent
        LOG.warning("Failed to save RNG state to %s: %s", save_path, exc)


def _save_generation_config(model: Any, output_dir: str) -> None:
    """Write generation_config.json when available to mirror HF checkpoints."""
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is None or not hasattr(gen_cfg, "save_pretrained"):
        return
    try:
        gen_cfg.save_pretrained(output_dir)
    except OSError as exc:  # pragma: no cover - filesystem dependent
        LOG.warning("Failed to save generation config to %s: %s", output_dir, exc)


def _write_global_step_marker(output_dir: str, step: Optional[int]) -> None:
    """Create the trainer-style global_step marker file."""
    if step is None:
        return
    marker_path = os.path.join(output_dir, f"global_step{step}")
    try:
        with open(marker_path, "w", encoding="utf-8") as marker:
            marker.write(str(step))
    except OSError as exc:  # pragma: no cover - filesystem dependent
        LOG.warning("Failed to write global step marker %s: %s", marker_path, exc)


def _write_trainer_state(output_dir: str, step: Optional[int]) -> None:
    """Create a minimal trainer_state.json mirroring HF GRPO checkpoints."""
    trainer_state = {"global_step": step}
    state_path = os.path.join(output_dir, "trainer_state.json")
    try:
        with open(state_path, "w", encoding="utf-8") as state_file:
            json.dump(trainer_state, state_file, indent=2)
    except OSError as exc:  # pragma: no cover - filesystem dependent
        LOG.warning("Failed to write trainer_state.json to %s: %s", state_path, exc)


def _save_scheduler_state(output_dir: str, lr_scheduler: Optional[Any]) -> None:
    """Persist the LR scheduler state to match HF checkpoint contents."""
    if lr_scheduler is None or not hasattr(lr_scheduler, "state_dict"):
        return
    if torch is None:
        LOG.debug("Torch not available; skipping scheduler state save.")
        return
    state_dict = lr_scheduler.state_dict()
    sched_path = os.path.join(output_dir, "scheduler.pt")
    try:
        torch.save(state_dict, sched_path)
    except OSError as exc:  # pragma: no cover - filesystem dependent
        LOG.warning("Failed to save scheduler state to %s: %s", sched_path, exc)


def _hf_global_step_from_label(step_label: str) -> Optional[int]:
    """Extract an integer global step from a checkpoint-<N> label."""
    match = re.search(r"(\d+)$", step_label)
    if not match:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def _save_trainer_state_like_hf(
    output_dir: str,
    step_label: str,
    accelerator: Any,
    lr_scheduler: Optional[Any],
) -> None:
    """Add HF Trainer-style metadata (rng, trainer_state, scheduler)."""
    global_step = _hf_global_step_from_label(step_label)
    _write_global_step_marker(output_dir, global_step)
    _write_trainer_state(output_dir, global_step)
    _save_rng_state(output_dir, accelerator)
    _save_scheduler_state(output_dir, lr_scheduler)


def _save_training_args(output_dir: str, training_args: Optional[Any]) -> None:
    """Persist training_args as HF training_args.bin."""
    if training_args is None:
        return
    if torch is None:
        LOG.debug("Torch not available; skipping training_args.bin save.")
        return
    path = os.path.join(output_dir, "training_args.bin")
    try:
        torch.save(training_args, path)
    except OSError as exc:  # pragma: no cover - filesystem dependent
        LOG.warning("Failed to save training_args.bin to %s: %s", path, exc)


def _should_skip_accelerator_state(accelerator: Any) -> bool:
    """Optionally skip accelerator.save_state when DeepSpeed is active."""
    env_skip = os.environ.get("MAXENT_SKIP_DEEPSPEED_STATE_SAVE", "false").lower() == "true"
    allow_save = os.environ.get("MAXENT_ALLOW_DEEPSPEED_STATE_SAVE", "false").lower() == "true"
    force_skip = os.environ.get("MAXENT_FORCE_SKIP_DEEPSPEED_STATE_SAVE", "false").lower() == "true"
    state = getattr(accelerator, "state", None)
    ds_plugin = getattr(state, "deepspeed_plugin", None)
    distributed_type = getattr(state, "distributed_type", None)

    def _is_deepspeed(distributed: Any) -> bool:
        """Best-effort check for DeepSpeed even when the plugin is None."""
        if DistributedType is not None and distributed == DistributedType.DEEPSPEED:
            return True
        try:
            return str(distributed).lower() == "deepspeed"
        except (AttributeError, ValueError):
            return False

    ds_detected = ds_plugin is not None or _is_deepspeed(distributed_type)

    if force_skip:
        if accelerator.is_main_process:
            LOG.info(
                "Forcing skip of accelerator.save_state "
                "(MAXENT_FORCE_SKIP_DEEPSPEED_STATE_SAVE=true)."
            )
        return True

    if ds_detected and env_skip and not allow_save:
        if accelerator.is_main_process:
            LOG.warning(
                "DeepSpeed detected; honoring MAXENT_SKIP_DEEPSPEED_STATE_SAVE=true "
                "and skipping accelerator.save_state. "
                "Set MAXENT_ALLOW_DEEPSPEED_STATE_SAVE=true to save despite "
                "DeepSpeed (may hang).",
            )
        return True

    if ds_detected and allow_save and env_skip and accelerator.is_main_process:
        LOG.info(
            "DeepSpeed detected; MAXENT_ALLOW_DEEPSPEED_STATE_SAVE=true so "
            "accelerator.save_state will run despite "
            "MAXENT_SKIP_DEEPSPEED_STATE_SAVE=true.",
        )

    # If no deepspeed plugin is active, honor the env flag directly.
    if not ds_detected:
        return env_skip

    # DeepSpeed present and no skip conditions matched.
    return False


def _deepspeed_engine(accelerator: Any, model: Any) -> Optional[Any]:
    """Return a DeepSpeed engine when one is attached to the accelerator/model."""
    for candidate in (
        getattr(accelerator, "deepspeed_engine", None),
        getattr(getattr(accelerator, "state", None), "deepspeed_engine", None),
        model,
    ):
        if candidate is None:
            continue
        if hasattr(candidate, "save_checkpoint"):
            return candidate
    return None


def _save_state_or_deepspeed_checkpoint(accelerator: Any, model: Any, path: str) -> None:
    """Prefer DeepSpeed's save_checkpoint when active; otherwise use save_state."""
    save_state = getattr(accelerator, "save_state", None)
    ds_engine = _deepspeed_engine(accelerator, model)
    state = getattr(accelerator, "state", None)
    distributed_type = getattr(state, "distributed_type", None)
    ds_plugin = getattr(state, "deepspeed_plugin", None)
    env_skip = os.environ.get("MAXENT_SKIP_DEEPSPEED_STATE_SAVE", "false").lower() == "true"
    allow_save = os.environ.get("MAXENT_ALLOW_DEEPSPEED_STATE_SAVE", "false").lower() == "true"
    force_skip = os.environ.get("MAXENT_FORCE_SKIP_DEEPSPEED_STATE_SAVE", "false").lower() == "true"
    prefer_accelerator = (
        os.environ.get("MAXENT_PREFER_ACCELERATE_STATE_SAVE", "false").lower() == "true"
    )

    if (
        ds_engine is not None
        or ds_plugin is not None
        or str(distributed_type).lower() == "deepspeed"
    ):
        if env_skip or force_skip:
            if getattr(accelerator, "is_main_process", True):
                LOG.info(
                    "DeepSpeed detected; MAXENT_SKIP_DEEPSPEED_STATE_SAVE is true "
                    "so skipping DeepSpeed checkpoint/accelerator.save_state."
                )
            return
        if prefer_accelerator:
            if getattr(accelerator, "is_main_process", True):
                LOG.info(
                    "DeepSpeed detected; MAXENT_PREFER_ACCELERATE_STATE_SAVE=true so "
                    "using accelerator.save_state instead of DeepSpeed."
                )
            if callable(save_state) and not _should_skip_accelerator_state(accelerator):
                save_state(path)
            return
        # Default: mirror HF Trainer by using the DeepSpeed engine's checkpoint writer.
        try:
            ds_engine = ds_engine or accelerator
            save_ckpt = getattr(ds_engine, "save_checkpoint", None)
            if callable(save_ckpt):
                if getattr(accelerator, "is_main_process", True):
                    LOG.info("DeepSpeed save_checkpoint starting -> %s", path)
                save_ckpt(path)
                if getattr(accelerator, "is_main_process", True):
                    LOG.info("DeepSpeed save_checkpoint finished -> %s", path)
                return
        except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - runtime dependent
            LOG.warning(
                "DeepSpeed save_checkpoint failed, will fall back to "
                "accelerator.save_state if allowed: %s",
                exc,
            )
        # Optional fallback to accelerator.save_state when explicitly allowed.
        if not allow_save:
            LOG.info(
                "DeepSpeed detected but MAXENT_ALLOW_DEEPSPEED_STATE_SAVE is not true; "
                "skipping accelerator.save_state.",
            )
            return
    if callable(save_state) and not _should_skip_accelerator_state(accelerator):
        save_state(path)


def _maybe_trigger_vllm_snapshot(output_dir: str, step_label: str, source: str) -> None:
    """Best-effort hook to ask an external vLLM server to persist weights."""
    url = os.environ.get("VLLM_SNAPSHOT_URL") or os.environ.get("VLLM_SAVE_URL")
    if not url:
        return
    if requests is None:
        LOG.warning(
            "VLLM snapshot URL set but requests is unavailable; "
            "skipping vLLM save trigger."
        )
        return
    timeout = float(os.environ.get("VLLM_SNAPSHOT_TIMEOUT", "5"))
    token = os.environ.get("VLLM_SNAPSHOT_TOKEN") or os.environ.get("VLLM_SAVE_TOKEN")
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = {
        "checkpoint_path": output_dir,
        "step": step_label,
        "timestamp": int(time.time()),
        "source": source,
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout, headers=headers)
        if resp.status_code >= 300:
            LOG.warning("vLLM snapshot request failed (%s): %s", resp.status_code, resp.text[:200])
    except RequestException as exc:  # pragma: no cover - network dependent
        LOG.warning("vLLM snapshot request failed: %s", exc)


__all__ = [
    "CheckpointHandles",
    "CheckpointManager",
    "CheckpointState",
    "finalize_training",
    "_copy_initial_model_snapshot",
]
