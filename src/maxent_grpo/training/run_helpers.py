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

"""Shared helper utilities for the MaxEnt-GRPO training pipeline."""

from __future__ import annotations

import importlib
import logging
from contextlib import nullcontext
import os
from dataclasses import dataclass, field
from functools import lru_cache
from types import ModuleType, SimpleNamespace
import sys
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Union,
    TYPE_CHECKING,
    runtime_checkable,
)

from maxent_grpo.config import GRPOConfig
from maxent_grpo.telemetry.wandb import init_wandb_training

if TYPE_CHECKING:
    from accelerate import Accelerator
    from accelerate.utils import DeepSpeedPlugin
    from torch import Tensor
    from transformers import PreTrainedTokenizer
else:  # pragma: no cover - typing fallbacks
    Tensor = Any
    PreTrainedTokenizer = Any

try:  # Optional dependency for reading accelerate config files
    import yaml
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional
    yaml = None

LOG = logging.getLogger(__name__)
PROMPT_CHAR_LIMIT = int(os.environ.get("MAX_PROMPT_CHARS", "2048"))
_TRUNC_STATE = {"warned": False}
_FIRST_WANDB_LOGGED_RUNS: Set[Any] = set()


@lru_cache(maxsize=None)
def _import_module(module_name: str) -> ModuleType:
    """Cache module imports to avoid repeated dynamic lookups."""
    return importlib.import_module(module_name)


def _require_dependency(module_name: str, context_hint: str) -> ModuleType:
    """Import a dependency or raise a helpful error when it is missing."""
    try:
        return _import_module(module_name)
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise ImportError(context_hint) from exc


def _optional_dependency(module_name: str) -> Optional[ModuleType]:
    """Attempt to import an optional dependency."""
    try:
        return _import_module(module_name)
    except ModuleNotFoundError:
        return None


@lru_cache(maxsize=None)
def _wandb_error_types() -> Tuple[type, ...]:
    """Return exception types that should be suppressed during W&B logging."""
    base_exceptions: Tuple[type, ...] = (RuntimeError, ValueError)
    errors_module = _optional_dependency("wandb.errors")
    if errors_module is None:
        return base_exceptions
    wandb_error = getattr(errors_module, "Error", None)
    if isinstance(wandb_error, type) and issubclass(wandb_error, BaseException):
        return (wandb_error,) + base_exceptions
    return base_exceptions


def _report_to_contains(
    report_to: Union[str, Sequence[str], None], target: str
) -> bool:
    """Case-insensitive membership check for TrainingArguments.report_to."""
    if report_to is None:
        return False
    if isinstance(report_to, str):
        entries = [report_to]
    else:
        entries = list(report_to)
    target = target.lower()
    return any(str(item).lower() == target for item in entries)


def _maybe_init_wandb_run(
    accelerator: Accelerator,
    training_args: GRPOConfig,
    wandb_config: Dict[str, Any],
) -> Optional[Any]:
    """Initialize a W&B run when report_to includes wandb."""
    if not _report_to_contains(getattr(training_args, "report_to", None), "wandb"):
        return None
    init_wandb_training(training_args)
    if not accelerator.is_main_process:
        os.environ.setdefault("WANDB_MODE", os.environ.get("WANDB_MODE", "offline"))
        return None
    wandb = _optional_dependency("wandb")
    if wandb is None:
        LOG.warning(
            "report_to includes wandb but the wandb package is not installed; skipping logging."
        )
        return None

    run_name = getattr(training_args, "run_name", None)
    wandb_kwargs: Dict[str, Any] = {
        "config": wandb_config,
        "dir": os.environ.get("WANDB_DIR") or os.getcwd(),
    }
    if run_name:
        wandb_kwargs["name"] = run_name
    project = os.environ.get("WANDB_PROJECT")
    if project:
        wandb_kwargs["project"] = project
    entity = os.environ.get("WANDB_ENTITY")
    if entity:
        wandb_kwargs["entity"] = entity
    group = os.environ.get("WANDB_RUN_GROUP")
    if group:
        wandb_kwargs["group"] = group
    return wandb.init(**wandb_kwargs)


def _log_wandb(run: Optional[Any], metrics: Dict[str, Any], step: int) -> None:
    """Safely log metrics to a W&B run."""
    if run is None or not metrics:
        return
    run_key = getattr(run, "id", None) or id(run)
    if run_key not in _FIRST_WANDB_LOGGED_RUNS:
        LOG.info(
            "Logging first metrics to W&B | step=%d | keys=%s",
            step,
            ",".join(sorted(metrics.keys())[:5]) if metrics else "",
        )
        _FIRST_WANDB_LOGGED_RUNS.add(run_key)
    error_types = _wandb_error_types()
    try:
        run.log(metrics, step=step)
    except error_types as exc:  # pragma: no cover - defensive logging
        LOG.warning("Failed to log metrics to W&B: %s", exc)


def _maybe_create_deepspeed_plugin() -> Optional[DeepSpeedPlugin]:
    """Construct a DeepSpeedPlugin from Accelerate env/config when available."""
    if os.environ.get("ACCELERATE_USE_DEEPSPEED", "false").lower() != "true":
        return None
    ds_module = _require_dependency(
        "accelerate.utils",
        (
            "DeepSpeed integration requires the Accelerate package. "
            "Install it via `pip install accelerate[deepspeed]`."
        ),
    )
    try:
        ds_class = getattr(ds_module, "DeepSpeedPlugin")
    except AttributeError as exc:  # pragma: no cover - defensive guard
        raise ImportError(
            "accelerate.utils does not expose DeepSpeedPlugin; update Accelerate."
        ) from exc

    ds_cfg: Dict[str, Any] = {}
    cfg_path = os.environ.get("ACCELERATE_CONFIG_FILE")
    if cfg_path and yaml is not None and os.path.isfile(cfg_path):
        handled_exceptions: Tuple[type, ...] = (OSError, ValueError)
        yaml_error = getattr(yaml, "YAMLError", None)
        if isinstance(yaml_error, type):
            handled_exceptions = handled_exceptions + (yaml_error,)
        try:
            with open(cfg_path, "r", encoding="utf-8") as cfg_file:
                raw = yaml.safe_load(cfg_file) or {}
            ds_cfg = raw.get("deepspeed_config") or {}
        except handled_exceptions:
            ds_cfg = {}
    zero_stage_raw = ds_cfg.get("zero_stage", 3)
    zero_stage = int(zero_stage_raw) if zero_stage_raw is not None else None
    offload_param = ds_cfg.get("offload_param_device")
    offload_optim = ds_cfg.get("offload_optimizer_device")
    zero3_init_flag = ds_cfg.get("zero3_init_flag")
    zero3_save = ds_cfg.get("zero3_save_16bit_model")

    kwargs = {
        "zero_stage": zero_stage,
        "offload_param_device": offload_param,
        "offload_optimizer_device": offload_optim,
        "zero3_init_flag": zero3_init_flag,
        "zero3_save_16bit_model": zero3_save,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if not kwargs:
        return None
    return ds_class(**kwargs)


def _build_torch_stub() -> Any:
    """Return a lightweight torch stub for test environments."""
    class _Tensor:
        def __init__(self, data=None, dtype=None):
            self.data = list(data) if data is not None else []
            self.dtype = dtype

        def __iter__(self):
            return iter(self.data) if self.data is not None else iter([])

        def __len__(self):
            return len(self.data) if self.data is not None else 0

        @property
        def shape(self):
            if self.data and hasattr(self.data[0], "__len__"):
                return (len(self.data), len(self.data[0]))
            return (len(self.data),)

        def __getitem__(self, key):
            if isinstance(key, tuple) and len(key) == 2:
                rows, cols = key
                selected = []
                row_indices = range(len(self.data)) if isinstance(rows, slice) else [rows]
                for r in row_indices:
                    row_val = self.data[r]
                    if isinstance(row_val, list):
                        selected.append(row_val[cols])
                return _Tensor(selected, self.dtype)
            return _Tensor(self.data[key], self.dtype)

        def __setitem__(self, key, value):
            val = value.data if isinstance(value, _Tensor) else value
            if isinstance(key, tuple) and len(key) == 2 and isinstance(key[1], slice):
                row, sl = key
                if isinstance(row, int):
                    # ensure row exists
                    while len(self.data) <= row:
                        self.data.append([])
                    # slice assign into row list
                    row_data = list(self.data[row])
                    start, stop, step = sl.indices(len(row_data))
                    if isinstance(val, list):
                        row_data[start:stop:step] = val
                    else:
                        row_data[start:stop:step] = [val] * len(range(start, stop, step))
                    self.data[row] = row_data
                else:
                    self.data[key] = val
            else:
                self.data[key] = val

        def tolist(self):
            return list(self.data)

        def long(self):
            return self

        def float(self):
            return self

        def numel(self):
            return len(self.data)

        def sum(self, dim=None):
            if dim is None:
                return _Tensor([sum(self.data)], self.dtype)
            if self.data and isinstance(self.data[0], list):
                return _Tensor([sum(row) for row in self.data], self.dtype)
            return _Tensor([sum(self.data)], self.dtype)

        def item(self):
            try:
                return self.data[0]
            except (IndexError, TypeError):
                return self.data

        def _binary(self, other, op):
            other_data = other.data if isinstance(other, _Tensor) else other
            if isinstance(self.data, list) and self.data and isinstance(self.data[0], list):
                res = [[op(a, b if isinstance(other_data, list) else other_data) for a, b in zip(row, other_data[row_idx] if isinstance(other_data, list) else [other_data]*len(row))] for row_idx, row in enumerate(self.data)]
            else:
                if isinstance(other_data, list):
                    res = [op(a, b) for a, b in zip(self.data, other_data)]
                else:
                    res = [op(a, other_data) for a in self.data]
            return _Tensor(res, self.dtype)

        def __add__(self, other):
            return self._binary(other, lambda a, b: a + b)

        def __sub__(self, other):
            return self._binary(other, lambda a, b: a - b)

        def __rsub__(self, other):
            return _Tensor([other], self.dtype)._binary(self, lambda a, b: a - b)

        def __truediv__(self, other):
            return self._binary(other, lambda a, b: a / b)

        def cpu(self):
            return self

        def __eq__(self, other):
            return self._binary(other, lambda a, b: a == b)

        def __ge__(self, other):
            return self._binary(other, lambda a, b: a >= b)

    def _tensor(data=None, dtype=None):
        return _Tensor(data, dtype)

    def _zeros(shape, *_args, **_kwargs):
        n = int(shape[0]) if shape else 0
        if len(shape) > 1:
            m = int(shape[1])
            return _Tensor([[0] * m for _ in range(n)])
        return _Tensor([0] * n)

    def _ones_like(arr, *_args, **_kwargs):
        try:
            return _Tensor([1 for _ in arr])
        except (TypeError, ValueError):
            return _Tensor([])

    def _full(shape, fill_value, *_args, **_kwargs):
        n = int(shape[0]) if shape else 0
        if len(shape) > 1:
            m = int(shape[1])
            return _Tensor([[fill_value] * m for _ in range(n)])
        return _Tensor([fill_value] * n)

    def _cat(seq, dim=0):
        out: list[Any] = []
        for item in seq:
            if isinstance(item, _Tensor):
                out.extend(item.data)
            elif isinstance(item, list):
                out.extend(item)
        return _Tensor(out)

    def _size(arr, dim=None):
        try:
            return len(arr) if dim is None else len(arr[dim])
        except (TypeError, AttributeError):
            return 0

    def _to(self, *_args, **_kwargs):
        return self

    def _autocast(**_kwargs):
        return nullcontext()

    stub = SimpleNamespace(
        Tensor=_Tensor,
        tensor=_tensor,
        full=_full,
        ones_like=_ones_like,
        zeros=_zeros,
        cat=_cat,
        size=_size,
        autocast=_autocast,
    )
    stub.device = lambda *args, **kwargs: SimpleNamespace(type=str(args[0]) if args else "cpu")
    stub.nn = SimpleNamespace(functional=SimpleNamespace(log_softmax=lambda *a, **k: None))
    stub.autograd = SimpleNamespace(no_grad=lambda: nullcontext())
    stub.no_grad = lambda: nullcontext()

    def _all(x):
        data = x.data if isinstance(x, _Tensor) else x
        def _flatten(val):
            for item in val:
                if isinstance(item, list):
                    yield from _flatten(item)
                else:
                    yield item
        return all(_flatten(data))

    stub.all = _all
    stub.ones = _ones_like
    stub.zeros_like = _ones_like
    stub.long = int
    stub.float32 = float
    stub.int64 = int
    _Tensor.to = _to
    _Tensor.detach = lambda self: self
    _Tensor.clone = lambda self: _Tensor(list(self.data), self.dtype)
    _Tensor.size = lambda self, dim=None: _size(self.data, dim)
    def _tensor_clamp(self, min=None, max=None):
        result = []
        for v in self.data:
            val = v
            if min is not None and val < min:
                val = min
            if max is not None and val > max:
                val = max
            result.append(val)
        return _Tensor(result, self.dtype)

    _Tensor.clamp = _tensor_clamp
    return stub


def require_torch(context: str) -> Any:
    """Return the torch module or a stub for test environments."""
    existing = sys.modules.get("torch")
    if existing is not None:
        return existing
    try:
        torch_mod = _import_module("torch")
    except (ModuleNotFoundError, RuntimeError):  # pragma: no cover - import guard
        torch_mod = None
        try:
            import ops.sitecustomize as _bootstrap

            installer = getattr(_bootstrap, "_install_torch_stub", None)
            if callable(installer):
                installer()
                _import_module.cache_clear()
                torch_mod = _import_module("torch")
        except (ImportError, AttributeError, RuntimeError):
            torch_mod = None
        if torch_mod is None:
            torch_mod = _build_torch_stub()
    required_attrs = ("tensor", "full", "ones_like", "zeros")
    if torch_mod is not None and any(
        not hasattr(torch_mod, attr) for attr in required_attrs
    ):
        try:
            import ops.sitecustomize as _bootstrap

            installer = getattr(_bootstrap, "_install_torch_stub", None)
            if callable(installer):
                installer()
                _import_module.cache_clear()
                torch_mod = _import_module("torch")
        except (ImportError, AttributeError, RuntimeError):
            if torch_mod is None:
                torch_mod = _build_torch_stub()
    if torch_mod is None:
        torch_mod = _build_torch_stub()
    return torch_mod


def require_dataloader(context: str) -> Any:
    """Return torch.utils.data.DataLoader with a descriptive error on failure."""
    hint = (
        f"Torch's DataLoader is required for MaxEnt-GRPO {context}. "
        "Install torch first."
    )
    try:
        torch_data = _import_module("torch.utils.data")
    except ModuleNotFoundError:  # pragma: no cover - import guard
        torch_data = None
        try:
            import ops.sitecustomize as _bootstrap

            installer = getattr(_bootstrap, "_install_torch_stub", None)
            if callable(installer):
                installer()
                _import_module.cache_clear()
                torch_data = _import_module("torch.utils.data")
        except Exception:
            torch_data = None

        if torch_data is None:
            # Minimal stub for environments running without torch (e.g., perf scripts).
            torch_mod = sys.modules.get("torch")
            if torch_mod is None:
                torch_mod = _build_torch_stub()
                sys.modules["torch"] = torch_mod
            utils_mod = getattr(torch_mod, "utils", None)
            if utils_mod is None:
                utils_mod = ModuleType("torch.utils")
                sys.modules["torch.utils"] = utils_mod
                torch_mod.utils = utils_mod
            data_mod = ModuleType("torch.utils.data")
            data_mod.DataLoader = type("DataLoader", (), {})
            data_mod.Sampler = type("Sampler", (), {})
            sys.modules["torch.utils.data"] = data_mod
            utils_mod.data = data_mod
            torch_data = data_mod
    try:
        return getattr(torch_data, "DataLoader")
    except AttributeError as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            "torch.utils.data.DataLoader is missing; update your torch installation."
        ) from exc


def require_accelerator(context: str) -> Any:
    """Return accelerate.Accelerator or raise a helpful RuntimeError."""
    hint = (
        f"Accelerate is required for MaxEnt-GRPO {context}. "
        "Install it with `pip install accelerate`."
    )
    try:
        accelerate_mod = _import_module("accelerate")
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise RuntimeError(hint) from exc
    try:
        return getattr(accelerate_mod, "Accelerator")
    except AttributeError as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            "accelerate.Accelerator is unavailable; upgrade the accelerate package."
        ) from exc


def require_transformer_base_classes(context: str) -> Tuple[Any, Any]:
    """Return (PreTrainedModel, PreTrainedTokenizer) with clear failure messages."""
    hint = (
        f"Transformers is required for MaxEnt-GRPO {context}. "
        "Install it with `pip install transformers`."
    )
    try:
        transformers_mod = _import_module("transformers")
    except ModuleNotFoundError as exc:  # pragma: no cover - import guard
        raise RuntimeError(hint) from exc
    try:
        model_cls = getattr(transformers_mod, "PreTrainedModel")
        tokenizer_cls = getattr(transformers_mod, "PreTrainedTokenizer")
    except AttributeError as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            "Transformers does not expose PreTrainedModel/Tokenizer; upgrade transformers."
        ) from exc
    return model_cls, tokenizer_cls


def require_deepspeed(context: str, module: str = "deepspeed") -> ModuleType:
    """Return a DeepSpeed module import or raise a contextual RuntimeError."""
    hint = (
        f"DeepSpeed is required for MaxEnt-GRPO {context}. "
        "Install it with `pip install deepspeed`."
    )
    try:
        return _require_dependency(module, hint)
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(hint) from exc


def get_trl_prepare_deepspeed() -> Optional[Any]:
    """Return TRL's prepare_deepspeed helper when available."""
    utils_module = _optional_dependency("trl.trainer.utils")
    if utils_module is None:
        return None
    prepare = getattr(utils_module, "prepare_deepspeed", None)
    if not callable(prepare):
        return None
    return prepare


@runtime_checkable
class ChatTokenizer(Protocol):
    """Protocol for tokenizers with chat template capabilities."""

    def apply_chat_template(
        self,
        conversation: List[Dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = True,
    ) -> Union[str, List[int]]:
        """Render a conversation into a model-ready prompt."""
        raise NotImplementedError

    @property
    def eos_token_id(self) -> Optional[int]:
        """Expose the EOS token id used by the tokenizer."""
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Allow chat tokenizers to be invoked like standard HF tokenizers."""
        raise NotImplementedError


def truncate_prompt(prompt: str, char_limit: Optional[int] = None) -> str:
    """Clamp prompt strings to a safe length for vLLM/http payloads (shared warning state)."""
    limit = char_limit if char_limit is not None else PROMPT_CHAR_LIMIT
    if limit <= 0 or len(prompt) <= limit:
        return prompt
    if not _TRUNC_STATE["warned"]:
        LOG.warning(
            "Prompt length exceeded %d characters; truncating. "
            "Override via MAX_PROMPT_CHARS if needed.",
            limit,
        )
        _TRUNC_STATE["warned"] = True
    return prompt[:limit]


# Backwards compatibility for existing imports.
_truncate_prompt = truncate_prompt


def _prompt_char_limit_from_tokens(max_prompt_len: int) -> int:
    """Derive a character cap from the token cap (≈4 chars/token) with env floor."""
    approx_char_limit = (
        int(max_prompt_len * 4) if max_prompt_len and max_prompt_len > 0 else 0
    )
    if approx_char_limit <= 0:
        return PROMPT_CHAR_LIMIT
    return max(PROMPT_CHAR_LIMIT, approx_char_limit)


def _to_prompt(
    example: Dict[str, Any],
    tokenizer: Union["PreTrainedTokenizer", ChatTokenizer],
    prompt_column: str,
    system_prompt: Optional[str],
    char_limit: Optional[int] = None,
) -> Dict[str, str]:
    """Light copy of src/maxent_grpo/grpo.py:_to_prompt (kept local to avoid circular import)."""
    user = str(example.get(prompt_column, example.get("prompt", "")))
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user})

    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except (AttributeError, TypeError, ValueError, RuntimeError):
        prompt = (
            "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
            + "\nASSISTANT:"
        )
    prompt = truncate_prompt(prompt, char_limit)
    return {
        "prompt": prompt,
        "answer": str(example.get("answer", example.get("solution", ""))),
    }


@dataclass
class MaxEntOptions:
    """Lightweight knobs specific to MaxEnt sequence-level updates."""

    tau: float = field(default_factory=lambda: float(os.environ.get("MAXENT_TAU", 0.2)))
    q_temperature: float = field(
        default_factory=lambda: float(os.environ.get("MAXENT_Q_TEMPERATURE", 1.0))
    )
    q_epsilon: float = field(
        default_factory=lambda: float(os.environ.get("MAXENT_Q_EPS", 1e-6))
    )
    length_normalize_ref: bool = field(
        default_factory=lambda: os.environ.get("MAXENT_LENGTH_NORM_REF", "1")
        not in {"0", "false", "False"}
    )


@dataclass
class VLLMClientConfig:
    """Configuration for vLLM-backed completion generation with all exposed knobs."""

    url: str
    rounds_cfg: int
    retry_sleep: float
    backfill_local: bool
    request_logprobs: bool
    best_of: Optional[int] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    timeout: float = 120.0
    max_retries: int = 3
    backoff: float = 1.0
    guided_json: Optional[str] = None
    guided_regex: Optional[str] = None
    logit_bias: Optional[Dict[str, float]] = None
    request_id_prefix: Optional[str] = None
    sync_weights: bool = False


@dataclass
class GenerationPenaltyConfig:
    """Shared penalty/stop sequence overrides for completion sampling."""

    gen_top_k: Optional[int] = None
    gen_best_of: Optional[int] = None
    gen_frequency_penalty: float = 0.0
    gen_presence_penalty: float = 0.0
    gen_stop_sequences: Optional[List[str]] = None


@dataclass
class GenerationSamplingConfig:
    """Shared completion sampling knobs (HF + vLLM)."""

    max_prompt_len: int
    max_completion_len: int
    gen_temperature: float
    gen_top_p: float
    use_vllm: bool
    vllm: VLLMClientConfig

    @property
    def vllm_url(self) -> str:
        """Backward-compatible accessor for the vLLM endpoint URL."""
        return self.vllm.url

    @property
    def vllm_rounds_cfg(self) -> int:
        """Backward-compatible accessor for the maximum vLLM retry rounds."""
        return self.vllm.rounds_cfg

    @property
    def vllm_retry_sleep(self) -> float:
        """Backward-compatible accessor for the per-round retry sleep."""
        return self.vllm.retry_sleep

    @property
    def vllm_backfill_local(self) -> bool:
        """Backward-compatible accessor for local fallback behavior."""
        return self.vllm.backfill_local

    @property
    def vllm_request_logprobs(self) -> bool:
        """Backward-compatible accessor for whether to request logprobs."""
        return self.vllm.request_logprobs

    @property
    def vllm_best_of(self) -> Optional[int]:
        """Backward-compatible accessor for the best-of sampling count."""
        return self.vllm.best_of

    @property
    def vllm_frequency_penalty(self) -> float:
        """Backward-compatible accessor for the frequency penalty value."""
        return self.vllm.frequency_penalty

    @property
    def vllm_presence_penalty(self) -> float:
        """Backward-compatible accessor for the presence penalty value."""
        return self.vllm.presence_penalty

    @property
    def vllm_top_k(self) -> Optional[int]:
        """Backward-compatible accessor for the top-k sampling limit."""
        return self.vllm.top_k

    @property
    def vllm_stop_sequences(self) -> Optional[List[str]]:
        """Backward-compatible accessor for stop sequences."""
        return self.vllm.stop_sequences

    @property
    def vllm_timeout(self) -> float:
        """Backward-compatible accessor for request timeout."""
        return self.vllm.timeout

    @property
    def vllm_max_retries(self) -> int:
        """Backward-compatible accessor for maximum request retries."""
        return self.vllm.max_retries

    @property
    def vllm_backoff(self) -> float:
        """Backward-compatible accessor for exponential backoff factor."""
        return self.vllm.backoff

    @property
    def vllm_guided_json(self) -> Optional[str]:
        """Backward-compatible accessor for JSON schema-guided decoding."""
        return self.vllm.guided_json

    @property
    def vllm_guided_regex(self) -> Optional[str]:
        """Backward-compatible accessor for regex-guided decoding."""
        return self.vllm.guided_regex

    @property
    def vllm_logit_bias(self) -> Optional[Dict[str, float]]:
        """Backward-compatible accessor for logit bias configuration."""
        return self.vllm.logit_bias

    @property
    def vllm_request_id_prefix(self) -> Optional[str]:
        """Backward-compatible accessor for request-id prefixes."""
        return self.vllm.request_id_prefix

    @property
    def vllm_sync_weights(self) -> bool:
        """Whether to push model weights to the vLLM server before generation."""
        return bool(getattr(self.vllm, "sync_weights", False))


def _group_softmax(
    values: List[float],
    temperature: float = 1.0,
    eps: float = 1e-6,
) -> List[float]:
    """Numerically stable softmax with optional temperature and epsilon floor."""
    if len(values) == 0:
        return []
    torch_module = _require_dependency(
        "torch",
        (
            "MaxEnt softmax weighting requires PyTorch. "
            "Install it via `pip install torch`."
        ),
    )
    value_tensor = torch_module.tensor(values, dtype=torch_module.float32)
    value_tensor = value_tensor / max(temperature, 1e-8)
    value_tensor = value_tensor - value_tensor.max()
    probs = torch_module.softmax(value_tensor, dim=0)
    probs = probs * (1.0 - eps * len(values)) + eps
    probs = probs / probs.sum()
    return probs.tolist()


def _prepare_labels_for_ce(
    input_ids: "Tensor",
    prompt_lengths: List[int],
) -> "Tensor":
    """Create labels tensor with prompt tokens masked as -100 for CE."""
    labels = input_ids.clone()
    for i, plen in enumerate(prompt_lengths):
        labels[i, :plen] = -100
    return labels


def _batch_tokenize_pairs(
    tokenizer: "PreTrainedTokenizer",
    prompts: List[str],
    completions: List[str],
) -> Tuple["Tensor", "Tensor", List[int]]:
    """Tokenize prompt+completion pairs and return tensors + prompt lengths."""
    pairs = [p + c for p, c in zip(prompts, completions)]
    enc_prompts = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    prompt_lengths = (
        enc_prompts["input_ids"].ne(tokenizer.pad_token_id).sum(dim=1).tolist()
    )
    enc = tokenizer(
        pairs,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]
    return input_ids, attn, prompt_lengths


__all__ = [
    "ChatTokenizer",
    "GenerationSamplingConfig",
    "GenerationPenaltyConfig",
    "MaxEntOptions",
    "VLLMClientConfig",
    "require_accelerator",
    "require_dataloader",
    "require_torch",
    "require_transformer_base_classes",
    "require_deepspeed",
    "get_trl_prepare_deepspeed",
    "_batch_tokenize_pairs",
    "_group_softmax",
    "_log_wandb",
    "_maybe_create_deepspeed_plugin",
    "_maybe_init_wandb_run",
    "_prepare_labels_for_ce",
    "_to_prompt",
]
