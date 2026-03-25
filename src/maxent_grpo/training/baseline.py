"""
Minimal GRPO training entrypoint built on TRL.

This script wires up a standard ``trl.GRPOTrainer`` with:

* Dataset loading via ``core.data.get_dataset``.
* Simple chat‑templated prompts built from a dataset column.
* A small registry of reward functions from ``maxent_grpo.rewards.basic``.

It aims to be a clean baseline without experimental features (e.g., replay
buffers, schedulers, or custom trainers). Use together with
``maxent_grpo.config.ScriptArguments``/``maxent_grpo.config.GRPOConfig`` and TRL's ``TrlParser``.

Key functions

* ``_to_prompt``: Convert a dataset row to a chat prompt + gold answer.
* ``main``: Load data/model, construct ``GRPOTrainer``, train/eval, and handle
  Hub push and model card creation.

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

# The module is import‑light: heavy libs are imported lazily inside functions.

from __future__ import annotations
# pylint: disable=broad-exception-caught

import atexit
from contextlib import contextmanager, nullcontext
from collections.abc import MutableMapping as MutableMappingABC
from importlib import import_module
import json
import logging
import os
import sys
import threading
import time
from urllib.parse import urlparse
from typing import (
    Dict,
    Optional,
    Any,
    List,
    MutableMapping,
    Union,
    Callable,
    Iterator,
    Protocol,
    cast,
    runtime_checkable,
    TYPE_CHECKING,
)
from maxent_grpo.config import GRPOConfig, GRPOScriptArguments
from maxent_grpo.training.rewards import load_reward_functions
from maxent_grpo.training.data import resolve_dataloader_kwargs
from maxent_grpo.rewards.basic import get_reward_funcs as _compat_get_reward_funcs
from maxent_grpo.core.data import get_dataset, load_dataset_split
from maxent_grpo.core.hub import ensure_hf_repo_ready
from maxent_grpo.core.model import get_model, get_tokenizer
from maxent_grpo.training.runtime import log_run_header, require_torch
from maxent_grpo.training.seed_paper_eval_callback import SeedPaperEvalCallback
from maxent_grpo.training.runtime.prompts import (
    PROMPT_CHAR_LIMIT,
    _prompt_char_limit_from_tokens,
    _to_prompt,
)
from maxent_grpo.training.scoring_common import (
    _coerce_optional_int,
    _get_embedding_vocab_size,
)
from maxent_grpo.training.trl_trainer import (
    build_custom_grpo_trainer,
    wrap_trl_trainer,
)
from maxent_grpo.utils.deps_guard import ensure_real_dependencies

if TYPE_CHECKING:
    from trl import ModelConfig


class _LazyModuleProxy:
    """Proxy that lazily imports a module on first attribute access."""

    def __init__(self, module_name: str) -> None:
        self._module_name = module_name
        self._module: Any | None = None

    def _load(self) -> Any:
        if self._module is None:
            self._module = import_module(self._module_name)
        return self._module

    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__:
            return self.__dict__[name]
        module = self._load()
        value = getattr(module, name)
        setattr(self, name, value)
        return value


transformers = _LazyModuleProxy("transformers")


def _maybe_align_model_tokenizer_vocab(model: Any, tokenizer: Any) -> None:
    """Resize model embeddings when tokenizer exposes additional addressable ids."""

    try:
        tokenizer_size = _coerce_optional_int(len(tokenizer))
    except Exception:
        tokenizer_size = None
    if not isinstance(tokenizer_size, int) or tokenizer_size <= 0:
        return

    config = getattr(model, "config", None)
    model_vocab_size = _get_embedding_vocab_size(model, config)
    if isinstance(model_vocab_size, int) and model_vocab_size >= tokenizer_size:
        return

    resize_fn = getattr(model, "resize_token_embeddings", None)
    if not callable(resize_fn):
        return
    LOG.info(
        "Resizing model token embeddings from %s to tokenizer size %s to align special tokens.",
        model_vocab_size,
        tokenizer_size,
    )
    resize_fn(int(tokenizer_size))


def _guided_decoding_kwargs(guided_decoding: Any) -> Dict[str, Any]:
    """Extract vLLM guided-decoding fields across version variants."""

    kwargs = dict(getattr(guided_decoding, "kwargs", {}) or {})
    for name in (
        "json",
        "regex",
        "choice",
        "grammar",
        "json_object",
        "disable_fallback",
        "disable_any_whitespace",
        "disable_additional_properties",
        "whitespace_pattern",
        "structural_tag",
    ):
        if name not in kwargs:
            value = getattr(guided_decoding, name, None)
            if value is not None:
                kwargs[name] = value
    return kwargs


def _patch_vllm_guided_decoding_compat() -> None:
    """Bridge TRL 0.18 guided decoding onto vLLM 0.16 structured outputs."""

    try:
        sampling_params_mod = import_module("vllm.sampling_params")
    except Exception:
        return

    structured_outputs_cls = getattr(
        sampling_params_mod, "StructuredOutputsParams", None
    )
    if structured_outputs_cls is None:
        return

    guided_cls = getattr(sampling_params_mod, "GuidedDecodingParams", None)
    if guided_cls is None:
        class GuidedDecodingParams:
            def __init__(self, backend: Optional[str] = None, **kwargs: Any) -> None:
                self.backend = backend
                self.kwargs = dict(kwargs)
                for key, value in kwargs.items():
                    setattr(self, key, value)

        guided_cls = GuidedDecodingParams
        setattr(sampling_params_mod, "GuidedDecodingParams", guided_cls)

    vllm_mod = import_module("vllm")
    original_sampling_params = getattr(vllm_mod, "SamplingParams", None)
    if original_sampling_params is None:
        return

    def _guided_to_structured_outputs(guided_decoding: Any) -> Any:
        if guided_decoding is None:
            return None
        if isinstance(guided_decoding, structured_outputs_cls):
            return guided_decoding
        structured = structured_outputs_cls(
            **_guided_decoding_kwargs(guided_decoding)
        )
        backend = getattr(guided_decoding, "backend", None)
        if backend is not None:
            try:
                setattr(structured, "_backend", backend)
            except Exception:
                pass
        return structured

    def _compat_sampling_params(*args: Any, **kwargs: Any) -> Any:
        if "guided_decoding" in kwargs and "structured_outputs" not in kwargs:
            kwargs["structured_outputs"] = _guided_to_structured_outputs(
                kwargs.pop("guided_decoding")
            )
        else:
            kwargs.pop("guided_decoding", None)
        return original_sampling_params(*args, **kwargs)

    for module_name in ("trl.trainer.grpo_trainer", "trl.scripts.vllm_serve"):
        module = sys.modules.get(module_name)
        if module is None or getattr(module, "_maxent_guided_decoding_patch", False):
            continue
        setattr(module, "GuidedDecodingParams", guided_cls)
        setattr(module, "SamplingParams", _compat_sampling_params)
        setattr(module, "_maxent_guided_decoding_patch", True)


def _main_process_first(training_args: Any, desc: str) -> Any:
    """Return a process-ordering context when TrainingArguments provides one."""

    main_process_first = getattr(training_args, "main_process_first", None)
    if not callable(main_process_first):
        return nullcontext()
    try:
        return main_process_first(local=True, desc=desc)
    except TypeError:
        try:
            return main_process_first(desc=desc)
        except TypeError:
            return main_process_first()


@contextmanager
def _force_vllm_dtype(training_args: GRPOConfig) -> Iterator[None]:
    """Ensure colocated vLLM uses the requested dtype instead of model defaults."""

    dtype_override = None
    if getattr(training_args, "fp16", False):
        dtype_override = "float16"
    elif getattr(training_args, "bf16", False):
        dtype_override = "bfloat16"

    if not (dtype_override and getattr(training_args, "use_vllm", False)):
        yield
        return

    try:
        import trl.trainer.grpo_trainer as grpo_mod
        from vllm import LLM as _LLM
    except (ImportError, AttributeError, RuntimeError):
        # If vLLM/TRL isn't available, fall through without patching.
        yield
        return

    orig_llm = getattr(grpo_mod, "LLM", None)

    def _patched_llm(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("dtype", dtype_override)
        return _LLM(*args, **kwargs)

    if orig_llm is not None:
        grpo_mod.LLM = _patched_llm
    try:
        yield
    finally:
        if orig_llm is not None:
            grpo_mod.LLM = orig_llm


LOG = logging.getLogger(__name__)
_VLLM_BATCH_UPDATE_PREFIX = "__maxent_vllm_batch__:"

GRPOTrainerOverride: Optional[type] = None
get_peft_config_override: Optional[Any] = (
    None  # Callable but kept lax to avoid importing typing.Callable
)

__all__ = [
    "GRPOTrainerOverride",
    "get_peft_config_override",
    "get_reward_funcs",
    "run_baseline_training",
    "_to_prompt",
    "PROMPT_CHAR_LIMIT",
]

# Backward compatibility hook for tests/legacy callers that monkeypatch reward resolution.
get_reward_funcs = _compat_get_reward_funcs

_EVAL_DATASET_PRESETS: Dict[str, Dict[str, Optional[str]]] = {
    "math_500": {
        "dataset_name": "HuggingFaceH4/MATH-500",
        "dataset_config": "default",
        "split": "test",
        "prompt_column": "problem",
        "solution_column": "answer",
    },
    "aime24": {
        "dataset_name": "HuggingFaceH4/aime_2024",
        "dataset_config": "default",
        "split": "train",
        "prompt_column": "problem",
        "solution_column": "answer",
    },
    "aime25": {
        "dataset_name": "yentinglin/aime_2025",
        "dataset_config": "default",
        "split": "train",
        "prompt_column": "problem",
        "solution_column": "answer",
    },
    "amc": {
        "dataset_name": "AI-MO/aimo-validation-amc",
        "dataset_config": "default",
        "split": "train",
        "prompt_column": "problem",
        "solution_column": "answer",
    },
    "minerva": {
        "dataset_name": "math-ai/minervamath",
        "dataset_config": "default",
        "split": "test",
        "prompt_column": "question",
        "solution_column": "answer",
    },
    "olympiad_bench": {
        "dataset_name": "knoveleng/OlympiadBench",
        "dataset_config": "default",
        "split": "train",
        "prompt_column": "question",
        "solution_column": "answer",
    },
}
_EVAL_DATASET_ALIASES = {
    "math": "math_500",
    "aime_24": "aime24",
    "aime_2024": "aime24",
    "aime_25": "aime25",
    "aime_2025": "aime25",
    "olympiadbench": "olympiad_bench",
    "olympiad": "olympiad_bench",
    "oly": "olympiad_bench",
}


def _resolve_eval_dataset_preset(spec: str) -> Optional[Dict[str, Optional[str]]]:
    """Resolve built-in benchmark aliases used by training eval configs."""

    normalized = spec.strip().lower().replace("-", "_")
    normalized = _EVAL_DATASET_ALIASES.get(normalized, normalized)
    preset = _EVAL_DATASET_PRESETS.get(normalized)
    if preset is None:
        return None
    return dict(preset)


@runtime_checkable
class ChatTemplate(Protocol):
    """Protocol for objects with chat templating capabilities."""

    def apply_chat_template(
        self,
        conversation: List[Dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = True,
    ) -> Union[str, List[int]]:
        """Render a chat conversation according to an internal template.

        :param conversation: Ordered list of chat messages.
        :type conversation: list[dict[str, str]]
        :param tokenize: Whether to return token IDs instead of text.
        :type tokenize: bool
        :param add_generation_prompt: Append assistant prefix at the end.
        :type add_generation_prompt: bool
        :returns: The templated conversation as text or token IDs.
        :rtype: str | list[int]
        """
        raise NotImplementedError


def _collect_dataset_columns(dataset: Any) -> Dict[str, List[str]]:
    """Return per-split column names when discoverable."""

    col_map: Dict[str, List[str]] = {}
    cols = getattr(dataset, "column_names", None)
    if isinstance(cols, dict):
        for split, names in cols.items():
            if isinstance(names, (list, tuple)) and names:
                col_map[str(split)] = list(names)
        return col_map
    if isinstance(cols, (list, tuple)) and cols:
        return {"all": list(cols)}
    if isinstance(dataset, dict):
        for split, split_ds in dataset.items():
            split_cols = getattr(split_ds, "column_names", None)
            if isinstance(split_cols, (list, tuple)) and split_cols:
                col_map[str(split)] = list(split_cols)
                continue
            if isinstance(split_ds, list) and split_ds:
                first = split_ds[0]
                if isinstance(first, dict):
                    col_map[str(split)] = list(first.keys())
    return col_map


def _get_column_names(dataset: Any) -> List[str]:
    """Return a best-effort list of column names for a dataset split."""

    cols = getattr(dataset, "column_names", None)
    if isinstance(cols, (list, tuple)):
        return list(cols)
    return []


def _validate_dataset_columns(
    dataset: Any,
    *,
    prompt_column: str,
    solution_column: str,
    label: str,
) -> None:
    """Fail fast if required dataset columns are missing."""

    col_map = _collect_dataset_columns(dataset)
    if not col_map:
        LOG.debug("Unable to infer columns for %s; skipping early validation.", label)
        return
    message_only = {"messages", "message"}
    if all(cols and set(cols).issubset(message_only) for cols in col_map.values()):
        LOG.debug(
            "Detected message-only dataset columns for %s; skipping early validation.",
            label,
        )
        return
    missing_by_split: Dict[str, List[str]] = {}
    for split, cols in col_map.items():
        if (
            "messages" in cols
            and prompt_column not in cols
            and solution_column not in cols
        ):
            continue
        missing = [
            name for name in (prompt_column, solution_column) if name not in cols
        ]
        if missing:
            missing_by_split[split] = missing
    if missing_by_split:
        if all(
            set(missing) == {solution_column} for missing in missing_by_split.values()
        ):
            LOG.info(
                "%s is missing '%s'; continuing with empty answers.",
                label,
                solution_column,
            )
            return
        missing_desc = "; ".join(
            f"{split} missing {', '.join(cols)}"
            for split, cols in missing_by_split.items()
        )
        available_desc = "; ".join(
            f"{split}={sorted(cols)}" for split, cols in col_map.items()
        )
        raise ValueError(
            f"{label} is missing required columns: {missing_desc}. "
            f"Available columns: {available_desc}"
        )


def _resolve_prompt_column(dataset: Any, prompt_column: str) -> str:
    """Return an inferred prompt column when the default is missing."""
    if prompt_column != "problem":
        return prompt_column
    col_map = _collect_dataset_columns(dataset)
    if not col_map:
        return prompt_column
    if all("problem" in cols for cols in col_map.values()):
        return prompt_column
    if all("prompt" in cols for cols in col_map.values()):
        LOG.info("Prompt column '%s' missing; falling back to 'prompt'.", prompt_column)
        return "prompt"
    return prompt_column


def _split_eval_dataset_specs(raw_name: Any) -> List[str]:
    """Return normalized evaluation dataset spec entries from config."""

    if raw_name is None:
        return []
    if isinstance(raw_name, (list, tuple)):
        specs: List[str] = []
        for item in raw_name:
            item_text = str(item).strip()
            if item_text:
                specs.extend(
                    part.strip() for part in item_text.split(",") if part.strip()
                )
        return specs
    text = str(raw_name).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _canonical_eval_benchmark_label(spec: str) -> str:
    """Return stable benchmark labels used in eval metric suffixes."""

    normalized = spec.strip().lower().replace("-", "_")
    aliases = {
        "math": "MATH",
        "math500": "MATH",
        "math_500": "MATH",
        "aime24": "AIME24",
        "aime_24": "AIME24",
        "aime_2024": "AIME24",
        "amc": "AMC",
    }
    if normalized in aliases:
        return aliases[normalized]
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in spec.strip())
    cleaned = cleaned.strip("_").upper()
    return cleaned or "EVAL"


def _resolve_eval_dataset_spec(
    spec: str,
    *,
    default_dataset_config: Optional[str],
    default_split: str,
    default_prompt_column: str,
    default_solution_column: str,
) -> tuple[str, Optional[str], str, str, str, str]:
    """Resolve one evaluation dataset spec (preset alias or HF dataset id)."""

    dataset_name = spec
    dataset_config = default_dataset_config
    dataset_split = default_split
    prompt_column = default_prompt_column
    solution_column = default_solution_column

    preset = _resolve_eval_dataset_preset(spec)

    if preset is not None:
        dataset_name = str(preset.get("dataset_name") or dataset_name)
        raw_dataset_config = preset.get("dataset_config")
        dataset_config = (
            str(raw_dataset_config) if raw_dataset_config is not None else None
        )
        dataset_split = str(preset.get("split") or default_split)
        prompt_column = str(preset.get("prompt_column") or default_prompt_column)
        solution_column = str(preset.get("solution_column") or default_solution_column)

    benchmark_label = _canonical_eval_benchmark_label(spec)
    return (
        dataset_name,
        dataset_config,
        dataset_split,
        prompt_column,
        solution_column,
        benchmark_label,
    )


def _ensure_split_mapping(dataset: Any) -> MutableMapping[str, Any]:
    """Coerce dataset-like objects into a split->dataset mapping."""

    if isinstance(dataset, MutableMappingABC):
        return cast(MutableMapping[str, Any], dataset)
    if hasattr(dataset, "keys") and hasattr(dataset, "__getitem__"):
        return cast(MutableMapping[str, Any], dataset)
    return {"train": dataset}


def _resolve_vllm_group_port() -> Optional[int]:
    """Resolve the vLLM communicator port from launcher environment."""

    for key in ("VLLM_GROUP_PORT", "PORT_FOR_COMMUNICATION"):
        raw = str(os.getenv(key, "")).strip()
        if not raw:
            continue
        try:
            port = int(raw)
        except ValueError:
            LOG.warning("Ignoring invalid %s=%r (expected integer port).", key, raw)
            continue
        if 1 <= port <= 65535:
            return port
        LOG.warning("Ignoring out-of-range %s=%r (expected 1..65535).", key, raw)
    return None


@contextmanager
def _temporary_env(overrides: Dict[str, str]) -> Iterator[None]:
    """Temporarily set environment variables while preserving prior values."""

    if not overrides:
        yield
        return
    previous: Dict[str, Optional[str]] = {}
    for key, value in overrides.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, prior in previous.items():
            if prior is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior


def _loopback_host(base_url: str) -> bool:
    try:
        parsed = urlparse(base_url)
        host = parsed.hostname or ""
    except Exception:
        host = ""
    if not host:
        host = base_url
    host = host.strip().lower()
    return host in {"localhost", "127.0.0.1", "::1"}


def _vllm_client_nccl_overrides(base_url: str) -> Dict[str, str]:
    """Return conservative NCCL settings for loopback vLLM sync."""

    overrides: Dict[str, str] = {}
    enable_overrides = str(
        os.getenv("MAXENT_VLLM_CLIENT_NCCL_OVERRIDES", "0")
    ).strip().lower() in {"1", "true", "yes", "on"}
    if not enable_overrides:
        return overrides

    if not _loopback_host(base_url):
        explicit = os.getenv("MAXENT_VLLM_CLIENT_NCCL_SOCKET_IFNAME")
        if explicit and "NCCL_SOCKET_IFNAME" not in os.environ:
            overrides["NCCL_SOCKET_IFNAME"] = explicit
        explicit = os.getenv("MAXENT_VLLM_CLIENT_NCCL_P2P_DISABLE")
        if explicit and "NCCL_P2P_DISABLE" not in os.environ:
            overrides["NCCL_P2P_DISABLE"] = explicit
        explicit = os.getenv("MAXENT_VLLM_CLIENT_NCCL_IB_DISABLE")
        if explicit and "NCCL_IB_DISABLE" not in os.environ:
            overrides["NCCL_IB_DISABLE"] = explicit
        return overrides

    if "NCCL_SOCKET_IFNAME" not in os.environ:
        overrides["NCCL_SOCKET_IFNAME"] = os.getenv(
            "MAXENT_VLLM_CLIENT_NCCL_SOCKET_IFNAME", "lo"
        )
    if "NCCL_P2P_DISABLE" not in os.environ:
        overrides["NCCL_P2P_DISABLE"] = os.getenv(
            "MAXENT_VLLM_CLIENT_NCCL_P2P_DISABLE", "1"
        )
    if "NCCL_IB_DISABLE" not in os.environ:
        overrides["NCCL_IB_DISABLE"] = os.getenv(
            "MAXENT_VLLM_CLIENT_NCCL_IB_DISABLE", "1"
        )
    return overrides


def _vllm_sync_chunk_bytes() -> int:
    """Return the max weight-sync batch size for server-mode vLLM updates."""

    raw = os.getenv("MAXENT_VLLM_SYNC_CHUNK_MB", "64")
    try:
        mb = int(raw)
    except (TypeError, ValueError):
        mb = 64
    if mb <= 0:
        mb = 64
    return mb * 1024 * 1024


def _encode_vllm_batched_update(
    names: list[str],
    dtypes: list[str],
    shapes: list[list[int]],
) -> dict[str, Any]:
    """Encode batched vLLM weight metadata through TRL's legacy request model."""

    return {
        "name": _VLLM_BATCH_UPDATE_PREFIX
        + json.dumps(
            {"names": names, "dtypes": dtypes, "shapes": shapes},
            separators=(",", ":"),
        ),
        "dtype": dtypes[0] if dtypes else "float16",
        "shape": shapes[0] if shapes else [0],
    }


def _tensor_nbytes(tensor: Any) -> int:
    """Best-effort tensor size in bytes for batching decisions."""

    try:
        return int(tensor.numel()) * int(tensor.element_size())
    except Exception:
        return 0


def _import_builtin_vllm_weight_transfer() -> Optional[type]:
    """Return vLLM's built-in NCCL transfer engine when available."""

    try:
        nccl_engine_mod = import_module(
            "vllm.distributed.weight_transfer.nccl_engine"
        )
        gpu_worker_mod = import_module("vllm.v1.worker.gpu_worker")
    except Exception:
        return None

    engine_cls = getattr(nccl_engine_mod, "NCCLWeightTransferEngine", None)
    if engine_cls is None:
        return None
    if not (
        callable(getattr(engine_cls, "trainer_init", None))
        or callable(getattr(engine_cls, "init_process_group", None))
    ):
        return None
    if not callable(getattr(engine_cls, "trainer_send_weights", None)):
        return None
    worker_cls = getattr(gpu_worker_mod, "GPUWorker", None)
    if worker_cls is not None:
        if not callable(getattr(worker_cls, "init_weight_transfer_engine", None)):
            return None
        if not callable(getattr(worker_cls, "update_weights", None)):
            return None
    return engine_cls


def _builtin_weight_transfer_trainer_init(
    engine_cls: type,
    init_info: dict[str, Any],
) -> Any:
    """Initialize trainer-side vLLM weight transfer across version variants."""

    trainer_init = getattr(engine_cls, "trainer_init", None)
    if callable(trainer_init):
        return trainer_init(init_info)
    legacy_init = getattr(engine_cls, "init_process_group", None)
    if callable(legacy_init):
        return legacy_init(init_info)
    raise RuntimeError("Built-in vLLM weight transfer lacks trainer init entrypoint")


def _clear_vllm_client_buffer(client: Any) -> None:
    """Reset any buffered trainer-side weight updates."""

    setattr(client, "_maxent_weight_buffer", [])
    setattr(client, "_maxent_weight_buffer_bytes", 0)


def _resolve_vllm_client_generate_boundary(client: Any) -> Dict[str, Any]:
    """Resolve tokenizer/model boundary metadata for live server-mode rollouts."""

    cached = getattr(client, "_maxent_generate_boundary", None)
    if isinstance(cached, dict):
        return cached

    model_id = str(os.getenv("MAXENT_VLLM_SERVER_MODEL_NAME", "") or "").strip()
    if not model_id:
        boundary = {
            "model_id": None,
            "tokenizer_limit": None,
            "model_limit": None,
            "blocked_token_ids": [],
        }
        setattr(client, "_maxent_generate_boundary", boundary)
        return boundary

    tokenizer_limit_env = _coerce_optional_int(
        os.getenv("MAXENT_VLLM_SERVER_TOKENIZER_VOCAB_LIMIT")
    )
    model_limit_env = _coerce_optional_int(
        os.getenv("MAXENT_VLLM_SERVER_MODEL_VOCAB_LIMIT")
    )
    tokenizer_limit = (
        int(tokenizer_limit_env)
        if isinstance(tokenizer_limit_env, int) and tokenizer_limit_env > 0
        else None
    )
    model_limit = (
        int(model_limit_env)
        if isinstance(model_limit_env, int) and model_limit_env > 0
        else None
    )

    if tokenizer_limit is None or model_limit is None:
        try:
            transformers_mod = import_module("transformers")
            auto_tokenizer = getattr(transformers_mod, "AutoTokenizer")
            auto_config = getattr(transformers_mod, "AutoConfig")
            tokenizer = auto_tokenizer.from_pretrained(model_id, trust_remote_code=True)
            config = auto_config.from_pretrained(model_id, trust_remote_code=True)
        except Exception as exc:
            raise RuntimeError(
                "Failed to resolve vLLM token boundary for live server-mode rollouts "
                f"(model={model_id}): {exc}"
            ) from exc

        if tokenizer_limit is None:
            tokenizer_limit = max(
                int(getattr(tokenizer, "vocab_size", 0) or 0),
                int(len(tokenizer)),
            )
        if model_limit is None:
            model_limit = int(getattr(config, "vocab_size", 0) or 0)

    if tokenizer_limit <= 0 or model_limit <= 0:
        raise RuntimeError(
            "Resolved invalid vLLM token boundary values "
            f"(model={model_id}, tokenizer_limit={tokenizer_limit}, model_limit={model_limit})"
        )

    blocked_token_ids: List[int] = []
    if model_limit > tokenizer_limit:
        blocked_token_ids = list(range(int(tokenizer_limit), int(model_limit)))

    boundary = {
        "model_id": model_id,
        "tokenizer_limit": int(tokenizer_limit),
        "model_limit": int(model_limit),
        "blocked_token_ids": blocked_token_ids,
    }
    setattr(client, "_maxent_generate_boundary", boundary)
    if not bool(getattr(client, "_maxent_generate_boundary_logged", False)):
        LOG.warning(
            "Patched TRL VLLMClient.generate boundary | model=%s tokenizer_limit=%d model_limit=%d blocked_tail=%d",
            model_id,
            int(tokenizer_limit),
            int(model_limit),
            len(blocked_token_ids),
        )
        setattr(client, "_maxent_generate_boundary_logged", True)
    return boundary


def _normalize_vllm_generate_url(base_url: str) -> str:
    """Return the canonical /generate endpoint for a vLLM server base URL."""

    base = str(base_url or "").strip()
    if not base:
        raise RuntimeError("vLLM client base_url is unavailable")
    if base.endswith("/generate/"):
        return base
    if base.endswith("/generate"):
        return f"{base}/"
    return f"{base.rstrip('/')}/generate/"


def _validate_vllm_completion_ids(
    completion_ids: List[List[int]],
    *,
    tokenizer_limit: Optional[int],
    model_id: Optional[str],
) -> None:
    """Fail fast when live rollouts contain tokenizer-inaccessible token IDs."""

    if not isinstance(tokenizer_limit, int) or tokenizer_limit <= 0:
        return
    invalid_tokens = [
        int(token_id)
        for sequence in completion_ids
        for token_id in sequence
        if int(token_id) < 0 or int(token_id) >= int(tokenizer_limit)
    ]
    if not invalid_tokens:
        return
    sample = invalid_tokens[:16]
    raise RuntimeError(
        "Detected completion token ids outside the tokenizer-addressable range "
        f"(model={model_id or 'unknown'}, tokenizer_limit={int(tokenizer_limit)}, sample={sample})"
    )


def _patch_trl_vllm_client_init() -> None:
    """Patch TRL VLLMClient init handshake to avoid POST-first deadlocks."""

    try:
        import trl.extras.vllm_client as trl_vllm_client_mod
    except Exception as exc:  # pragma: no cover - optional dependency path
        LOG.debug("Skipping vLLM client patch; trl.extras import failed: %s", exc)
        return

    client_cls = getattr(trl_vllm_client_mod, "VLLMClient", None)
    if client_cls is None:
        return
    if getattr(client_cls, "_maxent_async_init_patch", False):
        return

    try:
        from maxent_grpo.training.generation.vllm_utils import (
            init_vllm_client_communicator as _init_vllm_client_communicator,
        )
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning("Failed to import async vLLM init helper: %s", exc)
        return

    original_ctor = getattr(client_cls, "__init__", None)
    original_init_communicator = getattr(client_cls, "init_communicator", None)
    original_update_named_param = getattr(client_cls, "update_named_param", None)
    original_update_model_params = getattr(client_cls, "update_model_params", None)
    original_reset_prefix_cache = getattr(client_cls, "reset_prefix_cache", None)
    original_close_communicator = getattr(client_cls, "close_communicator", None)
    original_generate = getattr(client_cls, "generate", None)
    if (
        not callable(original_ctor)
        or not callable(original_init_communicator)
        or not callable(original_update_named_param)
        or not callable(original_generate)
    ):
        return

    builtin_weight_transfer = _import_builtin_vllm_weight_transfer()

    def _patched_ctor(self: Any, *args: Any, **kwargs: Any) -> None:
        if "group_port" not in kwargs or kwargs.get("group_port") in (None, 0):
            resolved_group_port = _resolve_vllm_group_port()
            if resolved_group_port is not None:
                kwargs["group_port"] = resolved_group_port
        original_ctor(self, *args, **kwargs)
        _clear_vllm_client_buffer(self)
        setattr(
            self,
            "_maxent_weight_chunk_bytes",
            0 if builtin_weight_transfer is not None else _vllm_sync_chunk_bytes(),
        )
        setattr(self, "_maxent_builtin_weight_transfer", builtin_weight_transfer is not None)
        setattr(self, "_maxent_generate_boundary", None)

    def _patched_generate(
        self: Any,
        prompts: List[str],
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 16,
        guided_decoding_regex: Optional[str] = None,
    ) -> List[List[int]]:
        boundary = _resolve_vllm_client_generate_boundary(self)
        blocked_token_ids = list(boundary.get("blocked_token_ids") or [])
        url = _normalize_vllm_generate_url(getattr(self, "base_url", ""))
        payload: Dict[str, Any] = {
            "prompts": prompts,
            "n": int(n),
            "repetition_penalty": float(repetition_penalty),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "min_p": float(min_p),
            "max_tokens": int(max_tokens),
            "guided_decoding_regex": guided_decoding_regex,
        }
        if blocked_token_ids:
            payload["blocked_token_ids"] = blocked_token_ids
        response = self.session.post(url, json=payload)
        if response.status_code != 200:
            raise RuntimeError(
                f"Request failed: {response.status_code}, {response.text}"
            )
        response_payload = response.json()
        raw_completion_ids = response_payload.get("completion_ids")
        if not isinstance(raw_completion_ids, list):
            raise RuntimeError("vLLM generate response missing completion_ids")
        completion_ids: List[List[int]] = []
        for idx, item in enumerate(raw_completion_ids):
            if not isinstance(item, list):
                raise RuntimeError(f"completion_ids[{idx}] is not a list")
            completion_ids.append([int(token_id) for token_id in item])
        _validate_vllm_completion_ids(
            completion_ids,
            tokenizer_limit=cast(Optional[int], boundary.get("tokenizer_limit")),
            model_id=cast(Optional[str], boundary.get("model_id")),
        )
        return completion_ids

    def _patched_init_communicator(self: Any) -> None:
        base_url = str(getattr(self, "base_url", ""))
        overrides = _vllm_client_nccl_overrides(base_url)
        if overrides:
            LOG.info(
                "vLLM client NCCL overrides applied | %s",
                ", ".join(f"{k}={v}" for k, v in overrides.items()),
            )
        with _temporary_env(overrides):
            if builtin_weight_transfer is None:
                bound_original = cast(
                    Callable[[], None],
                    original_init_communicator.__get__(self, type(self)),
                )
                _init_vllm_client_communicator(
                    self,
                    log=LOG.info,
                    init_fn=bound_original,
                )
                return

            timeout = float(os.getenv("MAXENT_VLLM_INIT_TIMEOUT_S", "60"))
            retries_raw = os.getenv("MAXENT_VLLM_INIT_RETRIES", "2")
            backoff_raw = os.getenv("MAXENT_VLLM_INIT_RETRY_BACKOFF_S", "2.0")
            try:
                retries = max(1, int(retries_raw))
            except (TypeError, ValueError):
                retries = 2
            try:
                backoff_s = max(0.0, float(backoff_raw))
            except (TypeError, ValueError):
                backoff_s = 2.0

            host = str(getattr(self, "host", "") or "").strip()
            if not host:
                raise RuntimeError("vLLM client host is unavailable for weight sync")
            group_port = getattr(self, "group_port", None)
            if group_port in (None, 0):
                raise RuntimeError("vLLM group_port is unavailable for weight sync")

            def _close_local_group() -> None:
                if getattr(self, "pynccl_comm", None) is not None:
                    try:
                        delattr(self, "pynccl_comm")
                    except Exception:
                        setattr(self, "pynccl_comm", None)

            last_error: Optional[BaseException] = None
            for attempt in range(1, retries + 1):
                _close_local_group()
                try:
                    response = self.session.get(
                        f"{self.base_url}/get_world_size/",
                        timeout=timeout,
                    )
                    response.raise_for_status()
                    vllm_world_size = int(response.json()["world_size"])
                    world_size = vllm_world_size + 1
                    init_url = f"{self.base_url}/init_communicator/"
                    payload = {
                        "host": host,
                        "port": int(group_port),
                        "world_size": world_size,
                    }
                    post_resp = self.session.post(
                        init_url,
                        json=payload,
                        timeout=timeout,
                    )
                    if post_resp.status_code != 200:
                        raise RuntimeError(
                            "vLLM init_communicator POST failed: "
                            f"{post_resp.status_code} {getattr(post_resp, 'text', '')}"
                        )
                    # Match TRL's original init ordering: let the server accept
                    # the init request first, then join the NCCL group locally.
                    time.sleep(0.1)
                    comm = _builtin_weight_transfer_trainer_init(
                        builtin_weight_transfer,
                        {
                            "master_address": host,
                            "master_port": int(group_port),
                            "rank_offset": 1,
                            "world_size": world_size,
                        }
                    )
                    self.rank = 0
                    self.pynccl_comm = comm
                    if self.pynccl_comm is None:
                        raise RuntimeError(
                            "vLLM trainer weight-transfer init produced no communicator"
                        )
                    atexit.register(self.close_communicator)
                    return
                except Exception as exc:
                    last_error = exc
                    LOG.info(
                        "vLLM init_communicator failed (attempt %d): %s",
                        attempt,
                        exc,
                    )
                    _close_local_group()
                    if attempt >= retries:
                        break
                    time.sleep(backoff_s)
            if last_error is not None:
                raise RuntimeError(str(last_error)) from last_error

    def _flush_weight_buffer(self: Any) -> None:
        buffer = list(getattr(self, "_maxent_weight_buffer", []) or [])
        if not buffer:
            return
        if builtin_weight_transfer is None:
            raise RuntimeError("Built-in vLLM weight transfer is unavailable")
        if getattr(self, "pynccl_comm", None) is None:
            raise RuntimeError(
                "Communicator not initialized. Call `init_communicator` first."
            )
        names = [str(name) for name, _ in buffer]
        dtypes = [str(weight.dtype).split(".")[-1] for _, weight in buffer]
        shapes = [list(tuple(weight.shape)) for _, weight in buffer]
        url = f"{self.base_url}/update_named_param/"
        response_holder: Dict[str, Any] = {}

        def _post_update() -> None:
            try:
                response_holder["resp"] = self.session.post(
                    url,
                    json=_encode_vllm_batched_update(names, dtypes, shapes),
                )
            except Exception as exc:
                response_holder["error"] = exc

        post_thread = threading.Thread(target=_post_update, daemon=True)
        post_thread.start()
        builtin_weight_transfer.trainer_send_weights(
            iter(buffer),
            self.pynccl_comm,
            src=0,
            packed=True,
        )
        post_thread.join()
        post_error = response_holder.get("error")
        if post_error is not None:
            raise RuntimeError(f"vLLM update_named_param POST failed: {post_error}")
        response = response_holder.get("resp")
        if response is None:
            raise RuntimeError("vLLM update_named_param POST returned no response")
        if response.status_code != 200:
            raise RuntimeError(f"Request failed: {response.status_code}, {response.text}")
        _clear_vllm_client_buffer(self)

    def _patched_update_named_param(self: Any, name: str, weights: Any) -> None:
        if builtin_weight_transfer is None:
            dtype, shape = str(weights.dtype), tuple(weights.shape)
            url = f"{self.base_url}/update_named_param/"
            response = self.session.post(
                url,
                json={"name": name, "dtype": dtype, "shape": shape},
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Request failed: {response.status_code}, {response.text}"
                )

            # vLLM's NCCL broadcast is launched asynchronously on the current
            # CUDA stream. Replacing the broken store-backed barrier with a
            # stream sync keeps the source buffer valid until the transfer is
            # complete.
            self.pynccl_comm.broadcast(weights, src=self.rank)
            require_torch("baseline_vllm_weight_sync").cuda.current_stream(
                device=weights.device
            ).synchronize()
            return

        if weights is None:
            return
        tensor = getattr(weights, "detach", None)
        tensor = tensor() if callable(tensor) else weights
        weight_buffer = list(getattr(self, "_maxent_weight_buffer", []) or [])
        weight_buffer.append((str(name), tensor))
        setattr(self, "_maxent_weight_buffer", weight_buffer)
        total_bytes = int(getattr(self, "_maxent_weight_buffer_bytes", 0) or 0)
        total_bytes += _tensor_nbytes(tensor)
        setattr(self, "_maxent_weight_buffer_bytes", total_bytes)
        chunk_bytes = int(
            getattr(self, "_maxent_weight_chunk_bytes", _vllm_sync_chunk_bytes()) or 0
        )
        if chunk_bytes > 0 and total_bytes >= chunk_bytes:
            _flush_weight_buffer(self)

    def _patched_update_model_params(self: Any, model: Any) -> None:
        if builtin_weight_transfer is None or not callable(original_update_model_params):
            if callable(original_update_model_params):
                original_update_model_params(self, model)
            return
        original_update_model_params(self, model)
        _flush_weight_buffer(self)

    def _patched_reset_prefix_cache(self: Any) -> Any:
        if builtin_weight_transfer is not None:
            _flush_weight_buffer(self)
        if callable(original_reset_prefix_cache):
            return original_reset_prefix_cache(self)
        return None

    def _patched_close_communicator(self: Any) -> Any:
        if builtin_weight_transfer is None:
            if callable(original_close_communicator):
                return original_close_communicator(self)
            return None
        try:
            _flush_weight_buffer(self)
        except Exception:
            LOG.debug("Failed to flush pending vLLM weights during shutdown.")
        _clear_vllm_client_buffer(self)
        if getattr(self, "pynccl_comm", None) is not None:
            try:
                delattr(self, "pynccl_comm")
            except Exception:
                setattr(self, "pynccl_comm", None)
        session = getattr(self, "session", None)
        close = getattr(session, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                LOG.debug("Failed to close vLLM client session cleanly.")
        return None

    setattr(client_cls, "__init__", _patched_ctor)
    setattr(client_cls, "generate", _patched_generate)
    setattr(client_cls, "init_communicator", _patched_init_communicator)
    setattr(client_cls, "update_named_param", _patched_update_named_param)
    setattr(client_cls, "flush", _flush_weight_buffer)
    if callable(original_update_model_params):
        setattr(client_cls, "update_model_params", _patched_update_model_params)
    if callable(original_reset_prefix_cache):
        setattr(client_cls, "reset_prefix_cache", _patched_reset_prefix_cache)
    if callable(original_close_communicator):
        setattr(client_cls, "close_communicator", _patched_close_communicator)
    setattr(client_cls, "_maxent_async_init_patch", True)

    try:  # Keep GRPOTrainer's module-local alias in sync if it was imported earlier.
        import trl.trainer.grpo_trainer as trl_grpo_mod

        if getattr(trl_grpo_mod, "VLLMClient", None) is not client_cls:
            setattr(trl_grpo_mod, "VLLMClient", client_cls)
    except Exception:
        pass

    LOG.info("Applied async vLLM communicator patch to TRL VLLMClient.")


def run_baseline_training(
    script_args: GRPOScriptArguments,
    training_args: GRPOConfig,
    model_args: "ModelConfig",
) -> None:
    """Entrypoint that loads data/model, builds trainer, and runs GRPO.

    The function also performs a small eval subsample for speed if
    ``training_args.do_eval`` is enabled and an eval split exists.

    :param script_args: Script configuration including dataset and rewards.
    :type script_args: GRPOScriptArguments
    :param training_args: GRPO trainer arguments from TRL.
    :type training_args: GRPOConfig
    :param model_args: Model configuration for TRL/transformers.
    :type model_args: ``trl.ModelConfig``
    :returns: ``None``. Side effects include training, evaluation, and checkpointing.
    :rtype: None
    """
    # Ensure logs directory exists for any file redirections by launchers
    os.makedirs(os.environ.get("LOG_DIR", "var/artifacts/logs"), exist_ok=True)

    ensure_real_dependencies(context="baseline GRPO training")
    ensure_hf_repo_ready(training_args)
    if getattr(training_args, "controller_meta_enabled", False):
        LOG.info(
            "controller_meta_enabled is set; CustomGRPOTrainer will handle controller/meta updates."
        )

    # Import selected pieces lazily to keep module import light-weight
    if bool(getattr(training_args, "use_vllm", False)):
        _patch_vllm_guided_decoding_compat()
    from transformers.trainer_utils import get_last_checkpoint
    from trl import (
        GRPOTrainer as _GRPOTrainer,
        get_peft_config as _get_peft_config,
    )
    from trl.data_utils import maybe_apply_chat_template
    if bool(getattr(training_args, "use_vllm", False)):
        _patch_vllm_guided_decoding_compat()

    override = getattr(sys.modules[__name__], "GRPOTrainerOverride", None)
    if override is not None:
        trainer_cls = wrap_trl_trainer(override)
    else:
        trainer_cls = build_custom_grpo_trainer(_GRPOTrainer)
    # Avoid leaking overrides across calls/tests.
    setattr(sys.modules[__name__], "GRPOTrainerOverride", None)
    peft_factory = get_peft_config_override or _get_peft_config

    # Keep custom communicator patch opt-in to preserve open-r1 parity by default.
    if bool(getattr(training_args, "use_vllm", False)):
        patch_vllm_client = str(
            os.getenv("MAXENT_TRL_VLLM_CLIENT_PATCH", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        if patch_vllm_client:
            _patch_trl_vllm_client_init()
        else:
            LOG.info(
                "Skipping custom TRL vLLM communicator patch "
                "(MAXENT_TRL_VLLM_CLIENT_PATCH=0)."
            )

    transformers_mod = transformers
    set_seed_fn = getattr(transformers_mod, "set_seed", None)
    if callable(set_seed_fn):
        set_seed_fn(training_args.seed)
    if not getattr(training_args, "return_reward", False):
        setattr(training_args, "return_reward", True)
    # Keep stop sequences aligned across train/eval and vLLM/HF generation.
    vllm_stops = getattr(training_args, "vllm_stop_sequences", None)
    if getattr(training_args, "gen_stop_sequences", None) in (None, []):
        setattr(training_args, "gen_stop_sequences", vllm_stops)
    if getattr(training_args, "eval_stop_sequences", None) in (None, []):
        setattr(training_args, "eval_stop_sequences", vllm_stops)

    # Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logging.getLogger(__name__).setLevel(log_level)
    log_run_header(training_args)
    dl_kwargs = resolve_dataloader_kwargs(training_args)
    if dl_kwargs:
        # Normalize dataloader settings onto training_args for TRL/Trainer usage.
        try:
            training_args.dataloader_num_workers = int(
                dl_kwargs.get(
                    "num_workers", getattr(training_args, "dataloader_num_workers", 0)
                )
            )
        except (AttributeError, TypeError, ValueError) as exc:
            LOG.debug("Failed to set dataloader_num_workers: %s", exc)
        if "pin_memory" in dl_kwargs:
            try:
                training_args.dataloader_pin_memory = bool(dl_kwargs["pin_memory"])
            except (AttributeError, TypeError, ValueError) as exc:
                LOG.debug("Failed to set dataloader_pin_memory: %s", exc)
        if getattr(training_args, "dataloader_num_workers", 0) > 0:
            if "prefetch_factor" in dl_kwargs:
                try:
                    training_args.dataloader_prefetch_factor = int(
                        dl_kwargs["prefetch_factor"]
                    )
                except (AttributeError, TypeError, ValueError) as exc:
                    LOG.debug("Failed to set dataloader_prefetch_factor: %s", exc)
            if "persistent_workers" in dl_kwargs:
                try:
                    training_args.dataloader_persistent_workers = bool(
                        dl_kwargs["persistent_workers"]
                    )
                except (AttributeError, TypeError, ValueError) as exc:
                    LOG.debug("Failed to set dataloader_persistent_workers: %s", exc)
        else:
            # Avoid invalid prefetch/persistent settings when workers are disabled.
            try:
                training_args.dataloader_prefetch_factor = None
            except (AttributeError, TypeError, ValueError) as exc:
                LOG.debug("Failed to clear dataloader_prefetch_factor: %s", exc)
            try:
                training_args.dataloader_persistent_workers = None
            except (AttributeError, TypeError, ValueError) as exc:
                LOG.debug("Failed to clear dataloader_persistent_workers: %s", exc)
        LOG.info(
            "Baseline dataloader settings | num_workers=%s | pin_memory=%s | prefetch_factor=%s | persistent_workers=%s",
            getattr(training_args, "dataloader_num_workers", None),
            getattr(training_args, "dataloader_pin_memory", None),
            getattr(training_args, "dataloader_prefetch_factor", None),
            getattr(training_args, "dataloader_persistent_workers", None),
        )
    # Optional: datasets logging if available
    try:  # pragma: no cover - environment dependent
        import datasets as _hf_datasets

        datasets_utils = getattr(_hf_datasets, "utils", None)
        datasets_logging = getattr(datasets_utils, "logging", None)
        set_verbosity = getattr(datasets_logging, "set_verbosity", None)
        if callable(set_verbosity):
            set_verbosity(log_level)
    except (
        ImportError,
        ModuleNotFoundError,
        AttributeError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        LOG.debug("Skipping datasets logging setup: %s", exc)
    tf_logging_module = getattr(
        getattr(transformers_mod, "utils", None), "logging", None
    )
    if tf_logging_module is not None:
        set_verbosity = getattr(tf_logging_module, "set_verbosity", None)
        if callable(set_verbosity):
            set_verbosity(log_level)
        enable_default_handler = getattr(
            tf_logging_module, "enable_default_handler", None
        )
        if callable(enable_default_handler):
            enable_default_handler()
        enable_explicit_format = getattr(
            tf_logging_module, "enable_explicit_format", None
        )
        if callable(enable_explicit_format):
            enable_explicit_format()

    # Data / model
    raw_ds = get_dataset(script_args)
    pc = getattr(script_args, "dataset_prompt_column", "problem")
    pc = _resolve_prompt_column(raw_ds, pc)
    sc = getattr(script_args, "dataset_solution_column", "answer")
    dataset_label = getattr(script_args, "dataset_name", None) or getattr(
        script_args, "dataset_mixture", None
    )
    _validate_dataset_columns(
        raw_ds,
        prompt_column=pc,
        solution_column=sc,
        label=f"training dataset {dataset_label or ''}".strip(),
    )
    tokenizer = get_tokenizer(model_args, training_args)
    model = get_model(model_args, training_args)
    ensure_real_dependencies(
        context="baseline GRPO training",
        require_torch=False,
        require_transformers=False,
        require_trl=False,
        require_datasets=False,
        model=model,
        tokenizer=tokenizer,
    )

    # Ensure PAD token exists (left padding recommended for causal LMs)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            eos_token = tokenizer.eos_token
            if isinstance(eos_token, list):
                eos_token = eos_token[0] if eos_token else None
            if isinstance(eos_token, str):
                tokenizer.pad_token = eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            resize_fn = getattr(model, "resize_token_embeddings", None)
            if callable(resize_fn):
                resize_fn(len(tokenizer))
    _maybe_align_model_tokenizer_vocab(model, tokenizer)
    config = getattr(model, "config", None)
    if config is not None and getattr(config, "pad_token_id", None) is None:
        setattr(config, "pad_token_id", tokenizer.pad_token_id)
    try:
        tokenizer.padding_side = "left"
    except AttributeError as exc:
        LOG.debug("Unable to set tokenizer.padding_side: %s", exc)

    # Map dataset → prompt text + gold answer
    char_limit = _prompt_char_limit_from_tokens(
        getattr(training_args, "max_prompt_length", 0)
    )
    # Keep prompt mapping identical for GRPO and MaxEnt so startup prompt
    # preprocessing and chat-template behavior stay aligned.
    use_prompt_messages = True

    def _make_conversation(ex: Dict[str, Any]) -> Dict[str, Any]:
        if pc not in ex:
            raise ValueError(f"Dataset Question Field Error: {pc} is not supported.")
        prompt: List[Dict[str, str]] = []
        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})
        prompt.append({"role": "user", "content": str(ex[pc])})
        return {"prompt": prompt, "answer": str(ex.get(sc, ex.get("solution", "")))}

    def _map_fn(ex: Dict[str, Any]) -> Dict[str, Any]:
        """Map a training split example to prompt/answer text.

        :param ex: Dataset row containing prompt/answer fields.
        :type ex: dict[str, Any]
        :returns: Mapping with ``prompt``/``answer`` keys for training.
            :rtype: dict[str, Any]
        """
        if use_prompt_messages:
            return _make_conversation(ex)
        prompt_col = pc
        if prompt_col not in ex and prompt_col == "problem" and "prompt" in ex:
            prompt_col = "prompt"
        out = _to_prompt(
            ex,
            cast(Any, tokenizer),
            prompt_col,
            training_args.system_prompt,
            char_limit=char_limit,
            return_messages=use_prompt_messages,
        )
        out["answer"] = str(ex.get(sc, out.get("answer", "")))
        return out

    dataset: MutableMapping[str, Any]
    map_fn = getattr(raw_ds, "map", None)
    dataset: MutableMapping[str, Any]
    if callable(map_fn):
        with _main_process_first(training_args, "dataset prompt mapping"):
            dataset = _ensure_split_mapping(map_fn(_map_fn))
    else:

        class _Split:
            def __init__(self, rows: List[Any]) -> None:
                self._rows = rows

            @property
            def column_names(self) -> List[str]:
                return []

            def remove_columns(self, *_cols: Any) -> "_Split":
                return self

            def shuffle(self, seed: Any = None) -> "_Split":
                _ = seed
                return self

            def select(self, _indices: Any) -> "_Split":
                return self

            def __len__(self) -> int:
                return len(self._rows)

        class _DictDataset(dict):
            def map(self, fn: Callable[[Any], Any]) -> "_DictDataset":
                return _DictDataset(
                    {k: _Split([fn(ex) for ex in v]) for k, v in self.items()}
                )

        raw_splits = raw_ds if isinstance(raw_ds, dict) else {"train": raw_ds}
        dataset = _ensure_split_mapping(_DictDataset(raw_splits).map(_map_fn))
    for split in list(dataset):
        split_ds = dataset[split]
        if "messages" in _get_column_names(split_ds):
            remove_columns = getattr(split_ds, "remove_columns", None)
            if callable(remove_columns):
                dataset[split] = remove_columns("messages")

    try:
        rank = int(getattr(training_args, "local_rank", -1) or -1)
    except (TypeError, ValueError):
        rank = -1
    if rank in (-1, 0):
        try:
            sample = dataset[getattr(script_args, "dataset_train_split", "train")][0]
            rendered = maybe_apply_chat_template(sample, cast(Any, tokenizer)).get(
                "prompt"
            )
            if isinstance(rendered, str):
                preview = rendered[:400].replace("\n", "\\n")
                LOG.info(
                    "Prompt preview (chat template applied): %s%s",
                    preview,
                    "..." if len(rendered) > 400 else "",
                )
        except Exception as exc:
            LOG.debug("Failed to render prompt preview: %s", exc)

    # Resolve splits
    train_split = getattr(script_args, "dataset_train_split", "train")
    test_split = getattr(script_args, "dataset_test_split", None)
    if test_split is None:
        # prefer 'validation' then 'test' if present
        if "validation" in dataset:
            test_split = "validation"
        elif "test" in dataset:
            test_split = "test"

    train_ds = dataset[train_split]
    eval_ds = None
    eval_benchmark_name_to_id: Dict[str, int] = {}
    eval_benchmark_id_to_name: Dict[int, str] = {}
    eval_dataset_name = getattr(script_args, "eval_dataset_name", None)
    eval_prompt_col = getattr(script_args, "eval_dataset_prompt_column", None) or pc
    eval_solution_col = getattr(script_args, "eval_dataset_solution_column", None) or sc

    if training_args.do_eval:
        if eval_dataset_name:
            eval_split = getattr(script_args, "eval_dataset_split", "validation")
            eval_specs = _split_eval_dataset_specs(eval_dataset_name)
            if not eval_specs:
                eval_specs = [str(eval_dataset_name)]
            eval_dataset_parts: List[Any] = []

            for spec in eval_specs:
                (
                    spec_dataset_name,
                    spec_dataset_config,
                    spec_split,
                    spec_prompt_col,
                    spec_solution_col,
                    spec_benchmark,
                ) = _resolve_eval_dataset_spec(
                    spec,
                    default_dataset_config=getattr(
                        script_args, "eval_dataset_config", None
                    ),
                    default_split=eval_split,
                    default_prompt_column=eval_prompt_col,
                    default_solution_column=eval_solution_col,
                )

                eval_ds_raw = load_dataset_split(
                    spec_dataset_name,
                    spec_dataset_config,
                    spec_split,
                )
                spec_prompt_col = _resolve_prompt_column(eval_ds_raw, spec_prompt_col)
                _validate_dataset_columns(
                    eval_ds_raw,
                    prompt_column=spec_prompt_col,
                    solution_column=spec_solution_col,
                    label=f"eval dataset {spec_dataset_name}:{spec_split}",
                )

                benchmark_id = eval_benchmark_name_to_id.setdefault(
                    spec_benchmark, len(eval_benchmark_name_to_id)
                )
                eval_benchmark_id_to_name.setdefault(benchmark_id, spec_benchmark)

                def _map_eval_fn(
                    ex: Dict[str, Any],
                    *,
                    prompt_col: str = spec_prompt_col,
                    solution_col: str = spec_solution_col,
                    benchmark_label: str = spec_benchmark,
                    benchmark_idx: int = benchmark_id,
                ) -> Dict[str, Any]:
                    """Convert evaluation dataset rows into prompt/answer pairs."""

                    if use_prompt_messages:
                        if prompt_col not in ex:
                            raise ValueError(
                                f"Dataset Question Field Error: {prompt_col} is not supported."
                            )
                        prompt: List[Dict[str, str]] = []
                        if training_args.system_prompt is not None:
                            prompt.append(
                                {
                                    "role": "system",
                                    "content": training_args.system_prompt,
                                }
                            )
                        prompt.append({"role": "user", "content": str(ex[prompt_col])})
                        return {
                            "prompt": prompt,
                            "answer": str(ex.get(solution_col, ex.get("solution", ""))),
                            "eval_benchmark": benchmark_label,
                            "eval_benchmark_id": int(benchmark_idx),
                        }
                    resolved_prompt_col = prompt_col
                    if (
                        resolved_prompt_col not in ex
                        and resolved_prompt_col == "problem"
                        and "prompt" in ex
                    ):
                        resolved_prompt_col = "prompt"
                    out = _to_prompt(
                        ex,
                        cast(Any, tokenizer),
                        resolved_prompt_col,
                        training_args.system_prompt,
                        char_limit=char_limit,
                        return_messages=use_prompt_messages,
                    )
                    out["answer"] = str(ex.get(solution_col, out.get("answer", "")))
                    out["eval_benchmark"] = benchmark_label
                    out["eval_benchmark_id"] = int(benchmark_idx)
                    return out

                with _main_process_first(training_args, "eval dataset prompt mapping"):
                    mapped_eval = eval_ds_raw.map(_map_eval_fn)
                if "messages" in _get_column_names(mapped_eval):
                    remove_columns = getattr(mapped_eval, "remove_columns", None)
                    if callable(remove_columns):
                        mapped_eval = remove_columns("messages")
                eval_dataset_parts.append(mapped_eval)

            if len(eval_dataset_parts) == 1:
                eval_ds = eval_dataset_parts[0]
            elif eval_dataset_parts:
                try:
                    from datasets import concatenate_datasets as _hf_concat

                    eval_ds = _hf_concat(eval_dataset_parts)
                except Exception:
                    merged_rows: List[Any] = []
                    for part in eval_dataset_parts:
                        merged_rows.extend(list(part))
                    try:
                        from datasets import Dataset as _HFDataset

                        eval_ds = _HFDataset.from_list(merged_rows)
                    except Exception:
                        eval_ds = merged_rows
            if eval_benchmark_id_to_name:
                LOG.info(
                    "Configured eval benchmarks: %s",
                    ", ".join(
                        f"{idx}:{name}"
                        for idx, name in sorted(eval_benchmark_id_to_name.items())
                    ),
                )
        elif test_split is not None and test_split in dataset:
            full_eval = dataset[test_split]
            eval_ds = full_eval

    # Rewards
    reward_funcs, reward_weights = load_reward_functions(
        script_args, tokenizer, training_args
    )
    # Keep TRL args aligned with the resolved reward spec so GRPOTrainer's
    # validation (length match) succeeds even when recipes store rewards on
    # script_args only.
    try:
        setattr(training_args, "reward_weights", reward_weights)
    except (AttributeError, TypeError) as exc:
        LOG.debug("Failed to attach reward_weights to training_args: %s", exc)

    # Trainer
    with _force_vllm_dtype(training_args):
        trainer = trainer_cls(
            model=model,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            peft_config=peft_factory(model_args),
            processing_class=tokenizer,
        )
        # Expose trainer kwargs for tests that introspect trainer construction.
        setattr(
            trainer,
            "_init_kwargs",
            dict(
                model=model,
                reward_funcs=reward_funcs,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                peft_config=peft_factory(model_args),
                processing_class=tokenizer,
            ),
        )
        if eval_benchmark_id_to_name:
            setattr(
                trainer,
                "eval_benchmark_id_to_name",
                dict(eval_benchmark_id_to_name),
            )
            setattr(
                trainer,
                "eval_benchmark_name_to_id",
                dict(eval_benchmark_name_to_id),
            )
        if bool(getattr(training_args, "seed_paper_eval_enabled", False)) and hasattr(
            trainer, "add_callback"
        ):
            trainer.add_callback(SeedPaperEvalCallback(training_args))

    # Train
    logger = logging.getLogger(__name__)
    resume_request = getattr(training_args, "resume_from_checkpoint", None)
    last_ckpt: Optional[str] = None
    if isinstance(resume_request, str) and resume_request:
        if os.path.isdir(resume_request):
            last_ckpt = resume_request
        else:
            logger.warning(
                "resume_from_checkpoint=%s was provided but the path does not exist; "
                "starting from scratch.",
                resume_request,
            )
    elif resume_request is None:
        # Backward compatible behavior: if the output directory already contains a
        # checkpoint and the user did not explicitly opt out of resuming, prefer
        # picking up from the latest checkpoint.
        output_dir = getattr(training_args, "output_dir", None)
        if output_dir and os.path.isdir(output_dir):
            last_ckpt = get_last_checkpoint(output_dir)
    elif resume_request:
        output_dir = getattr(training_args, "output_dir", None)
        if output_dir and os.path.isdir(output_dir):
            last_ckpt = get_last_checkpoint(output_dir)
        if last_ckpt is None:
            logger.warning(
                "resume_from_checkpoint was requested but no checkpoint was found under %s; "
                "starting from scratch.",
                output_dir or "<unspecified>",
            )
    else:
        last_ckpt = None

    if last_ckpt is not None:
        training_args.resume_from_checkpoint = last_ckpt
    else:
        training_args.resume_from_checkpoint = None
    if bool(getattr(training_args, "seed_paper_eval_enabled", False)):
        eval_strategy = str(getattr(training_args, "eval_strategy", "") or "").strip().lower()
        built_in_eval_enabled = bool(getattr(training_args, "do_eval", False)) and eval_strategy not in {
            "",
            "no",
            "none",
        }
        if not built_in_eval_enabled and hasattr(training_args, "eval_on_start"):
            setattr(
                training_args,
                "seed_paper_eval_on_start",
                bool(getattr(training_args, "eval_on_start", False)),
            )
            setattr(training_args, "eval_on_start", False)
    train_result = trainer.train(resume_from_checkpoint=last_ckpt)
    if hasattr(trainer, "log_metrics"):
        trainer.log_metrics("train", train_result.metrics)
    if hasattr(trainer, "save_metrics"):
        trainer.save_metrics("train", train_result.metrics)
    if hasattr(trainer, "save_state"):
        trainer.save_state()

    # Save
    if bool(getattr(training_args, "final_model_save_enabled", True)):
        try:
            trainer.save_model(training_args.output_dir)
        except TypeError:
            trainer.save_model()
        if getattr(trainer, "accelerator", None) is not None and getattr(
            trainer.accelerator, "is_main_process", False
        ):
            if hasattr(trainer, "create_model_card"):
                trainer.create_model_card(
                    dataset_name=script_args.dataset_name, tags=["open-r1"]
                )
            if hasattr(trainer, "model") and hasattr(trainer.model, "config"):
                trainer.model.config.use_cache = True
                if hasattr(trainer.model.config, "save_pretrained"):
                    trainer.model.config.save_pretrained(training_args.output_dir)

    # Eval
    if training_args.do_eval and eval_ds is not None:
        if hasattr(trainer, "evaluate"):
            metrics = trainer.evaluate()
            if hasattr(trainer, "log_metrics"):
                trainer.log_metrics("eval", metrics)
            if hasattr(trainer, "save_metrics"):
                trainer.save_metrics("eval", metrics)

    # Hub
    if getattr(training_args, "push_to_hub", False):
        trainer.push_to_hub(dataset_name=script_args.dataset_name, tags=["open-r1"])
