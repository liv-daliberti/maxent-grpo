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

import json
import logging
import os
import shutil
import sys
import inspect
from typing import Any, Dict, Optional, Tuple, Protocol
from types import SimpleNamespace

from .types import (
    LoggingHandles,
    OptimizationSchedule,
    TrainingLoopState,
)
from .weighting.types import WeightingConfigLike
from .weighting.logic import (
    CONTROLLER_STATE_FILENAME,
    broadcast_controller_state,
    load_controller_state,
)


LOG = logging.getLogger(__name__)
_checkpoint_log_once = {"config": False, "strategy": False, "steps": False}


class ControllerPathsLike(Protocol):
    """Minimal controller path settings used by checkpoint helpers."""

    state_path: Optional[str]


class AcceleratorLike(Protocol):
    """Subset of Accelerator API used by training state utilities."""

    is_main_process: bool

    def wait_for_everyone(self) -> None:
        """Synchronize all processes."""

    def load_state(self, path: str) -> Any:
        """Load accelerator state from ``path``."""


def _is_safetensors_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("safetensors") is not None


def _callable_accepts_kwargs(fn: Any) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return True
    return any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
    )


def _callable_accepts_param(fn: Any, name: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return True
    if name in sig.parameters:
        return True
    return _callable_accepts_kwargs(fn)


def _checkpoint_has_hf_weights(checkpoint_dir: str) -> bool:
    candidates = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    return any(os.path.isfile(os.path.join(checkpoint_dir, name)) for name in candidates)


def _safetensors_header_has_valid_tensors(path: str) -> bool:
    """Return True when a safetensors file declares non-empty tensors.

    This is a lightweight validation that reads only the JSON header and checks that:
    - at least one tensor entry exists
    - no tensor has a zero-sized dimension
    """

    try:
        with open(path, "rb") as handle:
            header_len_raw = handle.read(8)
            if len(header_len_raw) != 8:
                return False
            header_len = int.from_bytes(header_len_raw, "little", signed=False)
            if header_len <= 0 or header_len > 128 * 1024 * 1024:
                return False
            header = handle.read(header_len)
        meta = json.loads(header.decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return False

    if not isinstance(meta, dict):
        return False

    saw_tensor = False
    for key, value in meta.items():
        if key == "__metadata__":
            continue
        if not isinstance(value, dict):
            continue
        shape = value.get("shape")
        if not isinstance(shape, list):
            continue
        if not shape:
            return False
        dims: list[int] = []
        for dim in shape:
            if not isinstance(dim, int):
                return False
            dims.append(dim)
        if any(dim <= 0 for dim in dims):
            return False
        saw_tensor = True
    return saw_tensor


def _checkpoint_has_valid_hf_weights(checkpoint_dir: str) -> bool:
    """Return True when a checkpoint directory contains loadable, non-empty HF weights."""

    safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.isfile(safetensors_path):
        return _safetensors_header_has_valid_tensors(safetensors_path)

    torch_bin = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.isfile(torch_bin):
        try:
            return os.path.getsize(torch_bin) > 1_000_000
        except OSError:
            return False

    # Sharded checkpoints: validate that the index exists and at least one shard file is present.
    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as handle:
                idx = json.load(handle) or {}
            weight_map = idx.get("weight_map") or {}
            if not isinstance(weight_map, dict) or not weight_map:
                return False
            shard_files = sorted({str(v) for v in weight_map.values() if v})
        except (OSError, ValueError, UnicodeDecodeError):
            return False
        for shard in shard_files:
            shard_path = os.path.join(checkpoint_dir, shard)
            if os.path.isfile(shard_path):
                try:
                    if os.path.getsize(shard_path) > 1_000_000:
                        return True
                except OSError:
                    continue
        return False

    index_path = os.path.join(checkpoint_dir, "pytorch_model.bin.index.json")
    if os.path.isfile(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as handle:
                idx = json.load(handle) or {}
            weight_map = idx.get("weight_map") or {}
            if not isinstance(weight_map, dict) or not weight_map:
                return False
            shard_files = sorted({str(v) for v in weight_map.values() if v})
        except (OSError, ValueError, UnicodeDecodeError):
            return False
        for shard in shard_files:
            shard_path = os.path.join(checkpoint_dir, shard)
            if os.path.isfile(shard_path):
                try:
                    if os.path.getsize(shard_path) > 1_000_000:
                        return True
                except OSError:
                    continue
        return False

    return False


def _read_checkpoint_latest_tag(checkpoint_dir: str) -> Optional[str]:
    """Return the DeepSpeed/Accelerate checkpoint tag stored in ``latest`` if present."""

    latest_path = os.path.join(checkpoint_dir, "latest")
    if not os.path.isfile(latest_path):
        return None
    try:
        with open(latest_path, "r", encoding="utf-8") as handle:
            tag = (handle.read() or "").strip()
    except OSError:
        return None
    if not tag:
        return None
    # Defensive: tags should be a single path component (avoid traversal).
    if os.sep in tag or (os.altsep and os.altsep in tag):
        return None
    return tag


def _checkpoint_has_accelerate_state(checkpoint_dir: str) -> bool:
    """Return True when a checkpoint directory looks loadable via ``accelerator.load_state``.

    Note: ``accelerator.load_state`` only supports checkpoints produced by
    ``accelerator.save_state`` (it expects specific filenames/structure, e.g.
    ``<checkpoint_dir>/pytorch_model/...`` for DeepSpeed). Hugging Face
    ``Trainer`` checkpoints (e.g. ``checkpoint-150/global_step150``) are *not*
    compatible with ``accelerator.load_state`` even though they may contain a
    DeepSpeed tag directory.
    """

    # Accelerate (DeepSpeed) stores shards under a directory named after MODEL_NAME,
    # e.g. <checkpoint_dir>/pytorch_model/{latest,global_step...}
    # Accelerate (FSDP) uses a different model directory name.
    for name in ("pytorch_model", "pytorch_model_fsdp"):
        if os.path.isdir(os.path.join(checkpoint_dir, name)):
            return True
    try:
        for entry in os.listdir(checkpoint_dir):
            if entry.startswith("pytorch_model_") and os.path.isdir(
                os.path.join(checkpoint_dir, entry)
            ):
                return True
    except OSError as exc:
        LOG.debug("Failed to scan checkpoint dir for model shards: %s", exc)

    # Accelerate's non-DeepSpeed checkpoints use `.bin` for optimizer/scheduler and
    # `random_states_<rank>.pkl` (torch-saved payloads) for RNG tracking.
    for name in (
        "optimizer.bin",
        "scheduler.bin",
        "sampler.bin",
        "dl_state_dict.bin",
        "scaler.pt",
        "random_states_0.pkl",
    ):
        if os.path.isfile(os.path.join(checkpoint_dir, name)):
            return True
    try:
        for entry in os.listdir(checkpoint_dir):
            if entry.startswith("optimizer_") and entry.endswith(".bin"):
                return True
            if entry.startswith("scheduler_") and entry.endswith(".bin"):
                return True
            if entry.startswith("random_states_") and entry.endswith(".pkl"):
                return True
    except OSError as exc:
        LOG.debug("Failed to scan checkpoint dir for optimizer/random state: %s", exc)
    return False


def _checkpoint_has_deepspeed_engine_state(checkpoint_dir: str) -> bool:
    """Return True when a checkpoint looks like a DeepSpeed engine checkpoint.

    This matches Hugging Face Trainer/TRL checkpoints that contain a ``latest`` file
    pointing at a ``global_step*`` directory with ZeRO shards (for example
    ``zero_pp_rank_*_model_states.pt``).
    """

    tag = _read_checkpoint_latest_tag(checkpoint_dir)
    if not tag or not tag.startswith("global_step"):
        return False
    tag_dir = os.path.join(checkpoint_dir, tag)
    if not os.path.isdir(tag_dir):
        return False
    try:
        for name in os.listdir(tag_dir):
            if name.endswith("_model_states.pt") or name.endswith("_optim_states.pt"):
                return True
    except OSError:
        return False
    return False


def _normalize_checkpoint_dir(path: str) -> str:
    """Promote DeepSpeed tag subfolders (e.g., ``global_step100``/``pytorch_model``) to their parent."""

    if not isinstance(path, str) or not path:
        return path
    normalized = path.rstrip(os.sep)
    if not os.path.isdir(normalized):
        return path
    base = os.path.basename(normalized)
    if base == "pytorch_model" or base.startswith("global_step"):
        parent = os.path.dirname(normalized)
        parent_tag = _read_checkpoint_latest_tag(parent)
        if parent_tag == base:
            return parent
        if os.path.basename(parent).startswith("checkpoint-"):
            return parent
    return path


def _state_dict_has_zero_sized_tensors(state_dict: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(state_dict, dict) or not state_dict:
        return True
    try:
        import torch
    except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional runtime
        return False
    for value in state_dict.values():
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return True
    # If there are no tensors (e.g., stubbed accelerators in tests), treat the
    # payload as non-indicative rather than invalid.
    return False


def _remove_hf_weight_files(checkpoint_dir: str) -> None:
    candidates = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    for name in candidates:
        path = os.path.join(checkpoint_dir, name)
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError:
            continue


def _save_consolidated_hf_weights(
    *,
    model_to_save: Any,
    checkpoint_dir: str,
    state_dict: Optional[Dict[str, Any]],
    max_shard_size: str = "100GB",
) -> None:
    save_pretrained = getattr(model_to_save, "save_pretrained", None)
    if not callable(save_pretrained):
        raise TypeError("Model does not define save_pretrained()")

    save_kwargs: Dict[str, Any] = {}
    if state_dict is not None and _callable_accepts_param(save_pretrained, "state_dict"):
        save_kwargs["state_dict"] = state_dict

    safetensors_ok = _is_safetensors_available()
    if _callable_accepts_param(save_pretrained, "safe_serialization"):
        save_kwargs["safe_serialization"] = bool(safetensors_ok)

    if _callable_accepts_param(save_pretrained, "max_shard_size"):
        save_kwargs["max_shard_size"] = max_shard_size

    try:
        save_pretrained(checkpoint_dir, **save_kwargs)
    except (OSError, RuntimeError, TypeError, ValueError):
        if save_kwargs.get("safe_serialization") is True:
            retry_kwargs = dict(save_kwargs)
            retry_kwargs["safe_serialization"] = False
            save_pretrained(checkpoint_dir, **retry_kwargs)
            return
        raise


def _parse_checkpoint_step(path: str) -> Optional[int]:
    """Return the numeric suffix from a ``checkpoint-<n>`` directory."""

    tail = os.path.basename(path.rstrip(os.sep))
    if tail.startswith("checkpoint-"):
        try:
            return int(tail.split("-")[-1])
        except (TypeError, ValueError):
            return None
    return None


def _parse_save_total_limit(value: Any) -> int:
    """Normalize ``save_total_limit`` configuration values."""

    if value is None:
        return 0
    try:
        limit = int(value)
    except (TypeError, ValueError):
        return 0
    return max(limit, 0)


def _prune_old_checkpoints(output_dir: Optional[str], limit: int) -> None:
    """Delete checkpoints to respect ``save_total_limit``."""

    if not output_dir or limit <= 0:
        return
    try:
        entries: list[tuple[int, str]] = []
        for name in os.listdir(output_dir):
            if not name.startswith("checkpoint-"):
                continue
            path = os.path.join(output_dir, name)
            if not os.path.isdir(path):
                continue
            step = _parse_checkpoint_step(name)
            key = step if step is not None else -1
            entries.append((key, name))
    except OSError:
        return
    if len(entries) <= limit:
        return
    entries.sort(key=lambda pair: (pair[0], pair[1]))
    to_remove = entries[: len(entries) - limit]
    for _, name in to_remove:
        path = os.path.join(output_dir, name)
        try:
            shutil.rmtree(path)
        except OSError as exc:
            LOG.warning("Failed to prune checkpoint %s: %s", path, exc)


def _get_last_checkpoint(output_dir: Optional[str]) -> Optional[str]:
    """Best-effort discovery of the latest checkpoint under ``output_dir``."""

    if not output_dir or not os.path.isdir(output_dir):
        return None
    try:
        from transformers.trainer_utils import get_last_checkpoint
    except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency
        get_last_checkpoint = None
    if callable(get_last_checkpoint):
        try:
            last = get_last_checkpoint(output_dir)
        except (OSError, RuntimeError, ValueError):  # pragma: no cover - defensive
            last = None
        if last:
            return last
    try:
        entries = [
            d
            for d in os.listdir(output_dir)
            if d.startswith("checkpoint-")
            and os.path.isdir(os.path.join(output_dir, d))
        ]
    except OSError:
        return None
    if not entries:
        return None
    entries.sort(key=lambda name: _parse_checkpoint_step(name) or -1)
    return os.path.join(output_dir, entries[-1])


def resolve_resume_checkpoint(
    training_args: Any,
) -> Tuple[Optional[str], bool]:
    """Resolve the checkpoint path to resume from, if any.

    :param training_args: Trainer configuration with resume flags and output_dir.
    :type training_args: Any
    :returns: Tuple of (checkpoint path or None, whether resume was requested).
    :rtype: tuple[str | None, bool]
    """

    resume_cfg = getattr(training_args, "resume_from_checkpoint", None)
    init_path = getattr(training_args, "init_from_checkpoint", None)
    output_dir = getattr(training_args, "output_dir", None)
    requested = bool(resume_cfg) or bool(init_path)

    def _validate(path: Optional[str]) -> Optional[str]:
        if isinstance(path, str) and path:
            candidate = _normalize_checkpoint_dir(path)
            if os.path.isdir(candidate):
                if _checkpoint_has_valid_hf_weights(candidate):
                    return candidate
                if _checkpoint_has_accelerate_state(candidate):
                    if _checkpoint_has_hf_weights(candidate):
                        LOG.warning(
                            "Checkpoint %s contains HF weight files but they look invalid; "
                            "will attempt resume via accelerator state instead.",
                            candidate,
                        )
                    else:
                        LOG.info(
                            "Checkpoint %s does not contain consolidated HF weights; "
                            "will attempt resume via accelerator state.",
                            candidate,
                        )
                    return candidate
                if _checkpoint_has_hf_weights(candidate):
                    LOG.warning(
                        "Checkpoint %s contains HF weight files but they look invalid "
                        "(e.g., zero-sized tensors); ignoring this checkpoint.",
                        candidate,
                    )
                else:
                    LOG.warning(
                        "Checkpoint %s does not contain a loadable HF weight file "
                        "(expected model.safetensors or pytorch_model.bin) or accelerator state; "
                        "ignoring this checkpoint.",
                        candidate,
                    )
                return None
        if path:
            LOG.warning(
                "resume_from_checkpoint=%s was requested but the path does not exist; "
                "starting from scratch.",
                path,
            )
        return None

    if isinstance(init_path, str) and init_path:
        resolved = _validate(init_path)
        if resolved:
            return resolved, True
    resolved = None
    if isinstance(resume_cfg, str) and resume_cfg:
        resolved = _validate(resume_cfg)
    elif resume_cfg:
        resolved = _validate(_get_last_checkpoint(output_dir))
        if resolved is None:
            LOG.warning(
                "resume_from_checkpoint was requested but no checkpoint was found under %s; "
                "starting from scratch.",
                output_dir or "<unspecified>",
            )
    else:
        resolved = None
    return resolved, requested


def load_trainer_state_metadata(checkpoint_path: Optional[str]) -> Dict[str, Any]:
    """Load trainer_state.json if available for resume bookkeeping.

    :param checkpoint_path: Path to a checkpoint directory.
    :type checkpoint_path: str | None
    :returns: Parsed metadata fields (global_step, best metrics, etc.).
    :rtype: dict[str, Any]
    """

    metadata: Dict[str, Any] = {}
    if not checkpoint_path:
        return metadata
    state_file = os.path.join(checkpoint_path, "trainer_state.json")
    if os.path.isfile(state_file):
        try:
            with open(state_file, "r", encoding="utf-8") as fh:
                raw = json.load(fh) or {}
            for key in ("global_step", "num_input_tokens_seen"):
                if key in raw:
                    metadata[key] = raw[key]
            for key in (
                "best_model_checkpoint",
                "best_metric",
                "best_global_step",
                "log_history",
            ):
                if key in raw:
                    metadata[key] = raw[key]
        except (OSError, ValueError, TypeError) as exc:  # pragma: no cover - best-effort
            LOG.warning(
                "Failed to read trainer state from %s: %s", state_file, exc
            )
    if "global_step" not in metadata:
        step = _parse_checkpoint_step(checkpoint_path)
        if step is not None:
            metadata["global_step"] = step
    return metadata


def _write_trainer_state_json(
    checkpoint_dir: str,
    training_args: Any,
    *,
    global_step: Optional[int],
    num_input_tokens_seen: Optional[float] = None,
    base_state: Optional[Dict[str, Any]] = None,
    accelerator: Optional[Any] = None,
) -> None:
    """Persist a minimal trainer_state.json so future resumes find the step."""

    payload: Dict[str, Any] = {
        "global_step": int(global_step or 0),
        "max_steps": getattr(training_args, "max_steps", None),
        "num_train_epochs": getattr(training_args, "num_train_epochs", None),
        "save_steps": getattr(training_args, "save_steps", None),
        "logging_steps": getattr(training_args, "logging_steps", None),
        "is_local_process_zero": True,
        "is_world_process_zero": True,
        "log_history": [],
    }
    if num_input_tokens_seen is not None:
        payload["num_input_tokens_seen"] = float(num_input_tokens_seen)
    if base_state:
        for key in ("best_model_checkpoint", "best_metric", "best_global_step", "log_history"):
            if key in base_state:
                payload[key] = base_state[key]
    if accelerator is not None:
        payload["is_local_process_zero"] = bool(
            getattr(
                accelerator,
                "is_local_process_zero",
                getattr(accelerator, "is_local_main_process", True),
            )
        )
        payload["is_world_process_zero"] = bool(
            getattr(
                accelerator,
                "is_world_process_zero",
                getattr(accelerator, "is_main_process", True),
            )
        )
    state_path = os.path.join(checkpoint_dir, "trainer_state.json")
    try:
        with open(state_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    except (OSError, TypeError, ValueError) as exc:  # pragma: no cover - filesystem errors
        LOG.warning("Failed to write trainer_state.json to %s: %s", checkpoint_dir, exc)


def build_checkpoint_saver(
    training_args: Any,
    runtime_handles: Any,
    optim_handles: Any,
    tokenizer: Any,
    *,
    state_ref: Optional[Dict[str, Any]] = None,
    base_trainer_state: Optional[Dict[str, Any]] = None,
    controller_cfg: Optional[ControllerPathsLike] = None,
) -> Any:
    """Return a save_checkpoint callable compatible with LoggingHandles.

    The returned callable snapshots accelerator state, model/optimizer weights,
    trainer state metadata, and optional controller state into a checkpoint
    directory under ``output_dir``.

    :param training_args: Training configuration containing output/checkpoint options.
    :param runtime_handles: Runtime bundle providing model/accelerator references.
    :param optim_handles: Optimizer bundle used for saving optimizer state.
    :param tokenizer: Tokenizer to serialize alongside checkpoints.
    :param state_ref: Mutable state dict used for cross-callback coordination.
    :param base_trainer_state: Optional base trainer state JSON to merge into saves.
    :param controller_cfg: Optional controller state paths for MaxEnt/InfoSeed.
    :returns: Callable ``save_checkpoint(name: str) -> None``.
    :rtype: Callable[[str], None]
    """

    output_dir = getattr(training_args, "output_dir", None)
    accelerator = getattr(runtime_handles, "accelerator", None)
    model = getattr(runtime_handles, "model", None)
    optimizer = getattr(optim_handles, "optimizer", None)
    save_total_limit = _parse_save_total_limit(
        getattr(training_args, "save_total_limit", None)
    )
    state_ref = state_ref if isinstance(state_ref, dict) else {}
    push_enabled = bool(
        getattr(training_args, "push_to_hub", False)
        or getattr(training_args, "push_to_hub_revision", False)
    )
    hub_strategy = str(getattr(training_args, "hub_strategy", "end") or "end").lower()
    push_every_save = push_enabled and hub_strategy in {"every_save", "checkpoint"}

    def _step_from_name(name: str) -> Optional[int]:
        if not isinstance(name, str):
            return None
        return _parse_checkpoint_step(name)

    def _save_checkpoint(checkpoint_name: str) -> None:
        if not output_dir:
            return
        checkpoint_dir = os.path.join(output_dir, checkpoint_name)
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
        except OSError as exc:  # pragma: no cover - filesystem guard
            LOG.warning("Failed to create checkpoint directory %s: %s", checkpoint_dir, exc)
            return
        wait_for_all = getattr(accelerator, "wait_for_everyone", None)
        if callable(wait_for_all):
            wait_for_all()
        save_state_fn = getattr(accelerator, "save_state", None)
        if callable(save_state_fn):
            try:
                save_state_fn(checkpoint_dir)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:  # pragma: no cover - accelerator dependent
                LOG.warning("Failed to save accelerator state to %s: %s", checkpoint_dir, exc)

        state_dict = None
        get_state_dict_fn = getattr(accelerator, "get_state_dict", None)
        if callable(get_state_dict_fn) and model is not None:
            # Needed for ZeRO-3/FSDP: gathers a full (saveable) state_dict.
            # This may involve collective ops, so run it on all ranks.
            candidates = [model]
            # Some Accelerate versions behave differently depending on whether the model is wrapped.
            unwrap = getattr(accelerator, "unwrap_model", None)
            if callable(unwrap):
                try:
                    unwrapped = unwrap(model)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    unwrapped = model
                candidates.append(unwrapped)
            for candidate in candidates:
                try:
                    gathered = get_state_dict_fn(candidate)
                except (OSError, RuntimeError, TypeError, ValueError) as exc:  # pragma: no cover - backend specific
                    LOG.warning(
                        "Failed to gather consolidated model state_dict for %s: %s",
                        checkpoint_dir,
                        exc,
                    )
                    continue
                if _state_dict_has_zero_sized_tensors(gathered):
                    LOG.warning(
                        "Accelerator returned an invalid consolidated state_dict for %s "
                        "(zero-sized tensors detected); trying fallback.",
                        checkpoint_dir,
                    )
                    continue
                state_dict = gathered
                break
        if getattr(accelerator, "is_main_process", True):
            model_to_save = model
            unwrap = getattr(accelerator, "unwrap_model", None)
            if callable(unwrap):
                try:
                    model_to_save = unwrap(model)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    model_to_save = model
            try:
                _save_consolidated_hf_weights(
                    model_to_save=model_to_save,
                    checkpoint_dir=checkpoint_dir,
                    state_dict=state_dict,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:  # pragma: no cover - model save guard
                LOG.warning("Failed to save model weights to %s: %s", checkpoint_dir, exc)
            if not _checkpoint_has_hf_weights(checkpoint_dir):
                LOG.warning(
                    "Checkpoint %s does not contain a loadable HF weight file "
                    "(expected model.safetensors or pytorch_model.bin).",
                    checkpoint_dir,
                )
            elif not _checkpoint_has_valid_hf_weights(checkpoint_dir):
                LOG.error(
                    "Checkpoint %s contains invalid HF weight files (e.g., zero-sized tensors). "
                    "Removing the HF weight artifacts to avoid poisoning future resumes.",
                    checkpoint_dir,
                )
                _remove_hf_weight_files(checkpoint_dir)
            try:
                tokenizer.save_pretrained(checkpoint_dir)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:  # pragma: no cover - tokenizer optional
                LOG.warning("Failed to save tokenizer to %s: %s", checkpoint_dir, exc)
            if optimizer is not None:
                try:
                    import torch

                    state_dict_fn = getattr(optimizer, "state_dict", None)
                    if callable(state_dict_fn):
                        torch.save(
                            state_dict_fn(),
                            os.path.join(checkpoint_dir, "optimizer.pt"),
                        )
                    else:
                        LOG.warning(
                            "Optimizer state_dict unavailable; skipping optimizer.pt for %s",
                            checkpoint_dir,
                        )
                except (OSError, RuntimeError, TypeError, ValueError) as exc:  # pragma: no cover - optimizer optional
                    LOG.warning(
                        "Failed to save optimizer state to %s: %s",
                        checkpoint_dir,
                        exc,
                    )
            if push_every_save:
                try:
                    from maxent_grpo.core.hub import push_to_hub_revision

                    push_args = SimpleNamespace(**getattr(training_args, "__dict__", {}))
                    push_args.output_dir = checkpoint_dir
                    push_args.push_to_hub_revision = True
                    push_to_hub_revision(
                        push_args,
                        extra_ignore_patterns=[],
                        include_checkpoints=True,
                    )
                except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:  # pragma: no cover - optional hub deps
                    LOG.warning(
                        "Failed to push checkpoint %s to Hub: %s",
                        checkpoint_dir,
                        exc,
                    )
            state_obj = state_ref.get("state")
            global_step = (
                int(getattr(state_obj, "global_step", 0))
                if state_obj is not None
                else (_step_from_name(checkpoint_name) or 0)
            )
            num_tokens = getattr(state_obj, "num_input_tokens_seen", None)
            _write_trainer_state_json(
                checkpoint_dir,
                training_args,
                global_step=global_step,
                num_input_tokens_seen=num_tokens,
                base_state=base_trainer_state,
                accelerator=accelerator,
            )
            controller_state_src = getattr(controller_cfg, "state_path", None) if controller_cfg else None
            if controller_state_src and os.path.isfile(controller_state_src):
                dst_path = os.path.join(checkpoint_dir, CONTROLLER_STATE_FILENAME)
                try:
                    shutil.copy2(controller_state_src, dst_path)
                except OSError as exc:
                    LOG.warning(
                        "Failed to copy controller state to %s: %s",
                        dst_path,
                        exc,
                    )
            if save_total_limit > 0:
                _prune_old_checkpoints(output_dir, save_total_limit)
        if callable(wait_for_all):
            wait_for_all()

    return _save_checkpoint


def maybe_clear_stale_controller_state(
    accelerator: AcceleratorLike, controller_cfg: ControllerPathsLike
) -> None:
    """Delete a stale controller state file when overwriting the output dir.

    :param accelerator: Accelerate handle used to determine the main process
        and trigger ``wait_for_everyone`` guards.
    :type accelerator: AcceleratorLike
    :param controller_cfg: Paths describing the active controller
        checkpoint/restore locations.
    :type controller_cfg: ControllerPathsLike
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
    path: Optional[str],
    _accelerator: Optional[AcceleratorLike],
    weighting_cfg: WeightingConfigLike,
) -> bool:
    """Load controller parameters from ``path`` when available.

    :param path: Filesystem path to a serialized controller state.
    :type path: str | None
    :param accelerator: Optional accelerator handle (unused, for signature parity/tests).
    :type accelerator: AcceleratorLike | None
    :param weighting_cfg: Mutable weighting configuration that will receive
        the loaded parameters.
    :type weighting_cfg: WeightingConfigLike
    :returns: ``True`` when the controller state was loaded successfully.
    :rtype: bool
    """
    if not path:
        return False
    load_fn = globals().get("load_controller_state", load_controller_state)
    success = False
    if callable(load_fn):
        try:
            success = bool(load_fn(path, weighting_cfg))
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            LOG.warning("Failed to load controller state %s: %s", path, exc)
            success = False
    else:
        success = bool(load_fn)
    if success:
        # Emit a simple success log for test visibility, then the detailed metrics
        LOG.info("Loaded controller state from %s", path)
        beta_val = getattr(weighting_cfg, "beta", None)
        tau_val = getattr(weighting_cfg, "tau", None)
        LOG.info(
            "Loaded controller state from %s | beta=%s tau=%s",
            path,
            beta_val,
            tau_val,
        )
    return success


def load_controller_state_chain(
    controller_cfg: ControllerPathsLike,
    accelerator: AcceleratorLike,
    weighting_cfg: WeightingConfigLike,
) -> bool:
    """Attempt to load controller state from resume directory or the current state.

    :param controller_cfg: Filesystem paths for controller checkpoints.
    :type controller_cfg: ControllerPathsLike
    :param accelerator: Accelerate handle performing logging/synchronization.
    :type accelerator: AcceleratorLike
    :param weighting_cfg: Mutable weighting settings that receive the loaded parameters.
    :type weighting_cfg: WeightingConfigLike
    :returns: ``True`` when controller resume was requested or a controller
        checkpoint was successfully loaded.
    :rtype: bool
    """
    maybe_clear_stale_controller_state(accelerator, controller_cfg)
    resume_path = getattr(controller_cfg, "resume_from", None)
    controller_loaded = False
    resume_attempted = False
    tried_paths: list[str] = []
    if isinstance(resume_path, str) and resume_path:
        resume_attempted = True
        resume_state_file = os.path.join(resume_path, CONTROLLER_STATE_FILENAME)
        tried_paths.append(resume_state_file)
        controller_loaded = _load_controller_file(
            resume_state_file, accelerator, weighting_cfg
        )
    if not controller_loaded and controller_cfg.state_path:
        tried_paths.append(controller_cfg.state_path)
        controller_loaded = _load_controller_file(
            controller_cfg.state_path, accelerator, weighting_cfg
        )
    if not controller_loaded and tried_paths:
        if resume_attempted:
            LOG.warning(
                "Controller resume was requested but no state was loaded | tried=%s",
                ", ".join(tried_paths),
            )
        else:
            LOG.info(
                "No controller state found; starting fresh | tried=%s",
                ", ".join(tried_paths),
            )
    broadcast_controller_state(accelerator, weighting_cfg)
    return bool(controller_loaded or resume_attempted)


def maybe_load_accelerator_state(
    resume_state_path: Optional[str], accelerator: AcceleratorLike
) -> None:
    """Load an accelerator state directory when resuming if available.

    :param resume_state_path: Filesystem path to an accelerator state directory
        (e.g., saved by ``accelerator.save_state``).
    :type resume_state_path: str | None
    :param accelerator: Accelerate handle whose ``load_state`` method will be invoked.
    :type accelerator: AcceleratorLike
    :returns: ``None``.
    """
    load_state_fn = getattr(accelerator, "load_state", None)
    if isinstance(resume_state_path, str) and resume_state_path:
        resume_state_path = _normalize_checkpoint_dir(resume_state_path)
    if (
        isinstance(resume_state_path, str)
        and resume_state_path
        and os.path.isdir(resume_state_path)
        and callable(load_state_fn)
    ):
        if not _checkpoint_has_accelerate_state(resume_state_path):
            if _checkpoint_has_deepspeed_engine_state(resume_state_path):
                loaded = False
                for model in getattr(accelerator, "_models", []) or []:
                    load_checkpoint = getattr(model, "load_checkpoint", None)
                    if not callable(load_checkpoint):
                        continue
                    # Custom loops may use a different optimizer parameter-group
                    # layout than the original Trainer run; default to loading
                    # weights only (no optimizer/scheduler state).
                    try:
                        load_checkpoint(
                            resume_state_path,
                            tag=None,
                            load_optimizer_states=False,
                            load_lr_scheduler_states=False,
                        )
                    except TypeError:
                        load_checkpoint(resume_state_path)
                    loaded = True
                if loaded:
                    accelerator.wait_for_everyone()
                    LOG.info(
                        "Loaded DeepSpeed checkpoint state from %s", resume_state_path
                    )
                return
        try:
            load_state_fn(resume_state_path)
            accelerator.wait_for_everyone()
            LOG.info("Loaded accelerator state from %s", resume_state_path)
        except (AssertionError, OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - environment dependent
            LOG.warning(
                "Failed to load accelerator state from %s: %s", resume_state_path, exc
            )


def maybe_checkpoint(
    logging_cfg: LoggingHandles, accelerator: AcceleratorLike, global_step: int
) -> None:
    """Checkpoint periodically while on the main process.

    :param logging_cfg: Logging handles containing checkpoint callbacks and
        scheduling knobs.
    :type logging_cfg: training.types.LoggingHandles
    :param accelerator: Accelerate handle used for synchronization and
        main-process checks.
    :type accelerator: AcceleratorLike
    :param global_step: Current optimizer step; used to decide whether
        ``save_steps`` divides the step index evenly.
    :type global_step: int
    :returns: ``None``.
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
    schedule: OptimizationSchedule, loop_state: "TrainingLoopState"
) -> None:
    """Set stop flag when the configured number of steps is reached.

    :param schedule: Optimization schedule describing ``total_training_steps``.
    :type schedule: training.types.OptimizationSchedule
    :param loop_state: Mutable training loop state whose ``stop_training`` flag
        should be updated when the threshold is crossed.
    :type loop_state: training.loop.TrainingLoopState
    :returns: ``None``.
    """
    if (
        schedule.total_training_steps > 0
        and loop_state.global_step >= schedule.total_training_steps
    ):
        loop_state.stop_training = True


def build_training_state(training_args) -> LoggingHandles:
    """Construct minimal logging handles for the custom runner.

    :param training_args: Training configuration providing save strategy/steps.
    :returns: ``LoggingHandles`` instance with a no-op checkpoint saver.
    :rtype: LoggingHandles
    """

    class _NoopWriter:
        def __init__(self):
            self.logged = []

        def log(self, metrics, step):
            self.logged.append((step, metrics))

        def flush(self):
            return

    writer = _NoopWriter()
    return LoggingHandles(
        metric_writer=writer,
        save_checkpoint=lambda *_a, **_k: None,
        save_strategy=getattr(training_args, "save_strategy", "no"),
        save_steps=int(getattr(training_args, "save_steps", 0) or 0),
        wandb_run=None,
    )


__all__ = [
    "maybe_clear_stale_controller_state",
    "load_controller_state_chain",
    "resolve_resume_checkpoint",
    "load_trainer_state_metadata",
    "maybe_load_accelerator_state",
    "maybe_checkpoint",
    "check_stop_condition",
    "build_checkpoint_saver",
    "build_training_state",
]

# Preserve a self-reference so monkeypatch paths like ``training.state.state`` resolve
# even after test shuffling or aliasing.
state = sys.modules[__name__]
