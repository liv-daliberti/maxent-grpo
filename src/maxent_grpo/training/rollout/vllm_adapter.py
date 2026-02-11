"""vLLM-focused helpers split away from the local generation path."""

from __future__ import annotations

import importlib
import sys
import logging
import os
import time
import numbers
from contextlib import AbstractContextManager, contextmanager
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    cast,
)

from maxent_grpo.generation.common import (
    AggregatedGenerationState as _AggregatedGenerationState,
    retry_incomplete_prompts as _retry_incomplete_prompts_impl,
    seed_generation_groups as _seed_generation_groups_impl,
)
from maxent_grpo.generation.vllm import (
    VLLMGenerationHelper,
    _VLLMGenerationState as _BaseVLLMGenerationState,
)
from maxent_grpo.generation.vllm_utils import (
    import_vllm_client_cls as _shared_import_vllm_client_cls,
    init_vllm_client_communicator as _shared_init_vllm_client_communicator,
    zero3_gather_factory as _shared_zero3_gather_factory,
)
from maxent_grpo.patches.vllm import VLLMLogprobResult, safe_generate
from maxent_grpo.training.runtime import require_accelerator, require_torch
from maxent_grpo.training.runtime.prompts import _truncate_prompt

from .context import GenerationContext

torch = require_torch("generation")
Accelerator = require_accelerator("generation")
dist = getattr(torch, "distributed", None)
LOG = logging.getLogger(__name__)


def _progress_log_enabled() -> bool:
    raw = os.getenv("MAXENT_PROGRESS_LOG")
    if raw is None or not str(raw).strip():
        return False
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}

if TYPE_CHECKING:
    from accelerate import Accelerator as AcceleratorLike
else:
    AcceleratorLike = Any


def _optional_import(module_name: str) -> Any:
    """Import a module if available without triggering import errors."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _zero3_gather_factory(
    accelerator: AcceleratorLike,
) -> Callable[[Sequence[Any]], AbstractContextManager[Any]]:
    return _shared_zero3_gather_factory(accelerator, import_fn=_optional_import)


def _import_vllm_client_cls(
    import_fn: Optional[Callable[[str], Any]] = None,
) -> Optional[type]:
    """Return TRL's VLLMClient using the provided import fn (defaults to optional_import)."""

    return _shared_import_vllm_client_cls(import_fn or _optional_import)


def _resolve_vllm_group_port() -> Optional[int]:
    for key in ("VLLM_GROUP_PORT", "PORT_FOR_COMMUNICATION"):
        value = os.environ.get(key)
        if not value:
            continue
        try:
            return int(value)
        except ValueError:
            LOG.warning("Invalid %s=%r; expected an integer.", key, value)
    return None


@contextmanager
def _temporary_env(overrides: Dict[str, str]) -> Iterable[None]:
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
        from urllib.parse import urlparse

        parsed = urlparse(base_url)
        host = parsed.hostname or ""
    except Exception:
        host = ""
    if not host:
        host = base_url
    host = host.strip().lower()
    return host in {"localhost", "127.0.0.1", "::1"}


def _vllm_client_nccl_overrides(base_url: str) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
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


def _is_peft_model_safe(target: Any) -> bool:
    """Return True if accelerate.utils reports that the model uses PEFT adapters."""
    accel_utils = _optional_import("accelerate.utils")
    if accel_utils is None:
        return False
    is_peft_model = getattr(accel_utils, "is_peft_model", None)
    if not callable(is_peft_model):
        return False
    try:
        return bool(is_peft_model(target))
    except (TypeError, AttributeError, ValueError):
        return False


_VLLMGenerationState = _BaseVLLMGenerationState


class VLLMGenerationMixin:
    """All vLLM-specific plumbing extracted from the main generator."""

    # Access to helper internals is intentional for tests/patching.

    ctx: GenerationContext

    def __init__(self, ctx: GenerationContext) -> None:
        self.ctx = ctx
        self._vllm_helper = VLLMGenerationHelper(ctx, self._generate_local)
        self._vllm_colocate_engine = None
        # Surface patchable hooks for tests so monkeypatched helpers.* propagate.
        if hasattr(self._vllm_helper, "set_safe_generate"):
            self._vllm_helper.set_safe_generate(safe_generate)
        else:
            setattr(self._vllm_helper, "_safe_generate", safe_generate)
        setattr(self._vllm_helper, "_scatter_object", _scatter_object)
        if hasattr(self._vllm_helper, "set_time_provider"):
            self._vllm_helper.set_time_provider(time)
        else:
            setattr(self._vllm_helper, "_time", time)  # pragma: no cover - legacy stubs
        setattr(self._vllm_helper, "_is_peft_model_safe", _is_peft_model_safe)
        if hasattr(self._vllm_helper, "set_fallback_generate"):
            self._vllm_helper.set_fallback_generate(self._generate_local)
        else:
            setattr(self._vllm_helper, "_fallback_generate", self._generate_local)
        self._configure_vllm_mode()

    def _configure_vllm_mode(self) -> None:
        if not getattr(self.ctx, "use_vllm", False):
            return
        mode = str(getattr(self.ctx, "vllm_mode", "server") or "server").strip().lower()
        if mode in {"inline", "local", "inprocess", "in-process"}:
            mode = "colocate"
        if mode != "colocate":
            return
        try:
            from .vllm_colocate import ColocateVLLMEngine
        except ImportError as exc:
            LOG.warning("vLLM colocate requested but unavailable: %s", exc)
            return
        is_main = bool(
            getattr(getattr(self.ctx, "accelerator", None), "is_main_process", True)
        )
        if is_main:
            device_hint = os.getenv("MAXENT_VLLM_COLOCATE_DEVICE", "")
            if not device_hint:
                local_rank = os.getenv("LOCAL_RANK") or os.getenv("SLURM_LOCALID")
                if local_rank is not None and str(local_rank).isdigit():
                    device_hint = f"cuda:{int(local_rank)}"
                else:
                    try:
                        import torch

                        if torch.cuda.is_available():
                            device_hint = f"cuda:{torch.cuda.current_device()}"
                    except Exception:
                        device_hint = ""
            LOG.info(
                "vLLM pre-init | CUDA_VISIBLE_DEVICES=%s MAXENT_VLLM_COLOCATE_DEVICE=%s "
                "LOCAL_RANK=%s SLURM_LOCALID=%s",
                os.getenv("CUDA_VISIBLE_DEVICES"),
                device_hint or "auto",
                os.getenv("LOCAL_RANK"),
                os.getenv("SLURM_LOCALID"),
            )
        sync_enabled = _env_flag("MAXENT_VLLM_COLOCATE_SYNC", False)
        if not sync_enabled and getattr(self.ctx, "vllm_sync_weights", False):
            LOG.warning(
                "vLLM colocate sync disabled; set MAXENT_VLLM_COLOCATE_SYNC=1 to enable."
            )
            try:
                setattr(self.ctx, "vllm_sync_weights", False)
            except Exception:
                pass
        self._vllm_colocate_engine = ColocateVLLMEngine(self.ctx, self._generate_local)
        batcher = getattr(self._vllm_helper, "set_request_batcher", None)
        if callable(batcher):
            batcher(self._vllm_colocate_engine.request_batch)
        else:
            setattr(
                self._vllm_helper,
                "_request_vllm_batch",
                self._vllm_colocate_engine.request_batch,
            )
        if sync_enabled:
            if not getattr(self.ctx, "vllm_sync_weights", False):
                setattr(self.ctx, "vllm_sync_weights", True)
                LOG.info("vLLM colocate sync enabled via MAXENT_VLLM_COLOCATE_SYNC=1.")
            sync_interval = _env_int("MAXENT_VLLM_COLOCATE_SYNC_INTERVAL")
            if sync_interval is not None:
                if sync_interval < 0:
                    LOG.warning(
                        "Invalid MAXENT_VLLM_COLOCATE_SYNC_INTERVAL=%s; ignoring.",
                        sync_interval,
                    )
                else:
                    setattr(self.ctx, "vllm_sync_interval_steps", sync_interval)
            else:
                current_interval = getattr(self.ctx, "vllm_sync_interval_steps", None)
                if current_interval in (None, 1):
                    setattr(self.ctx, "vllm_sync_interval_steps", 10)
                    LOG.info(
                        "vLLM colocate sync interval defaulted to 10 steps. "
                        "Override via MAXENT_VLLM_COLOCATE_SYNC_INTERVAL."
                    )
            is_main = bool(
                getattr(getattr(self.ctx, "accelerator", None), "is_main_process", True)
            )
            if is_main:
                try:
                    client = self._vllm_colocate_engine.sync_client()
                    self._vllm_client = client
                    self._vllm_sync_ready = True
                except Exception as exc:
                    LOG.warning("vLLM colocate sync client unavailable: %s", exc)
                    self._vllm_client = None
                    self._vllm_sync_ready = False
            else:
                # Avoid initializing colocate sync clients on non-main ranks.
                self._vllm_client = None
                self._vllm_sync_ready = False
                LOG.info(
                    "Skipping vLLM colocate sync client init on non-main rank."
                )
        LOG.info("vLLM colocate mode enabled; using in-process vLLM engine.")

    def _generate_local(
        self,
        prompts: List[str],
        num_samples: int,
        per_prompt_counts: Optional[List[int]] = None,
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
        raise NotImplementedError("Subclasses must implement _generate_local().")

    def _prompt_char_limit(self) -> int:
        raise NotImplementedError("Subclasses must implement _prompt_char_limit().")

    @property
    def _vllm_client(self) -> Any:
        client = getattr(self._vllm_helper, "vllm_client", None)
        if client is None:
            client = getattr(self._vllm_helper, "_vllm_client", None)
        return client

    @_vllm_client.setter
    def _vllm_client(self, value: Any) -> None:
        setattr(self._vllm_helper, "vllm_client", value)
        setattr(self._vllm_helper, "_vllm_client", value)

    @property
    def _vllm_sync_ready(self) -> bool:
        if hasattr(self._vllm_helper, "vllm_sync_ready"):
            return bool(getattr(self._vllm_helper, "vllm_sync_ready"))
        return bool(getattr(self._vllm_helper, "_vllm_sync_ready", False))

    @_vllm_sync_ready.setter
    def _vllm_sync_ready(self, value: bool) -> None:
        setattr(self._vllm_helper, "vllm_sync_ready", value)
        setattr(self._vllm_helper, "_vllm_sync_ready", value)

    @property
    def _last_vllm_synced_step(self) -> Optional[int]:
        step = getattr(self._vllm_helper, "last_vllm_synced_step", None)
        if step is None:
            step = getattr(self._vllm_helper, "_last_vllm_synced_step", None)
        return step

    @_last_vllm_synced_step.setter
    def _last_vllm_synced_step(self, value: Optional[int]) -> None:
        setattr(self._vllm_helper, "last_vllm_synced_step", value)
        setattr(self._vllm_helper, "_last_vllm_synced_step", value)

    @property
    def _fsdp_cls(self) -> Any:
        fsdp = getattr(self._vllm_helper, "fsdp_cls", None)
        if fsdp is None:
            fsdp = getattr(self._vllm_helper, "_fsdp_cls", None)
        return fsdp

    @_fsdp_cls.setter
    def _fsdp_cls(self, value: Any) -> None:
        setattr(self._vllm_helper, "fsdp_cls", value)
        setattr(self._vllm_helper, "_fsdp_cls", value)

    def _vllm_base_url(self, url: str) -> str:
        """Delegate to the shared vLLM helper to normalize the base URL."""
        base_url_fn_obj = getattr(self._vllm_helper, "vllm_base_url", None)
        def _fallback_normalized(value: str) -> str:
            resolved = self._invoke_helper("_vllm_base_url", value)
            return str(resolved) if resolved is not None else value

        normalized_fn: Callable[[str], str] = (
            cast(Callable[[str], str], base_url_fn_obj)
            if callable(base_url_fn_obj)
            else _fallback_normalized
        )
        return normalized_fn(url)

    def _ensure_vllm_client(self) -> bool:
        """Instantiate the TRL VLLMClient when weight sync is enabled."""
        try:
            helpers_mod = sys.modules.get(
                type(self).__module__
            ) or importlib.import_module("maxent_grpo.training.rollout.helpers")
        except ImportError:
            helpers_mod = None
        import_fn = getattr(
            self,
            "_import_vllm_client_cls",
            getattr(helpers_mod, "_import_vllm_client_cls", _import_vllm_client_cls),
        )
        ctx = self.ctx
        if not getattr(ctx, "vllm_sync_weights", False) or not getattr(
            ctx.accelerator, "is_main_process", False
        ):
            return False
        if self._vllm_client is not None and self._vllm_sync_ready:
            return True
        client_cls = import_fn()
        if client_cls is None:
            try:
                client_cls = import_fn(_optional_import)
            except TypeError:
                client_cls = None
        if client_cls is None or not callable(client_cls):
            self._vllm_sync_ready = False
            return False
        try:
            base_url = self._vllm_base_url(ctx.vllm_url)
            LOG.info(
                "vLLM client NCCL config | base_url=%s | vllm_url=%s | group_port=%s | NCCL_SOCKET_IFNAME=%s | MAXENT_VLLM_CLIENT_NCCL_SOCKET_IFNAME=%s | NCCL_P2P_DISABLE=%s | NCCL_IB_DISABLE=%s",
                base_url,
                ctx.vllm_url,
                _resolve_vllm_group_port(),
                os.getenv("NCCL_SOCKET_IFNAME", ""),
                os.getenv("MAXENT_VLLM_CLIENT_NCCL_SOCKET_IFNAME", ""),
                os.getenv("NCCL_P2P_DISABLE", ""),
                os.getenv("NCCL_IB_DISABLE", ""),
            )
            try:
                group_port = _resolve_vllm_group_port()
                client_kwargs = {"base_url": base_url}
                if group_port is not None:
                    client_kwargs["group_port"] = group_port
                client = client_cls(**client_kwargs)
            except TypeError:
                try:
                    client = client_cls(base_url=base_url)
                except TypeError:
                    client = client_cls()
            init = getattr(client, "init_communicator", None)
            if callable(init):
                overrides = _vllm_client_nccl_overrides(base_url)
                if overrides:
                    LOG.info(
                        "vLLM client NCCL overrides applied | %s",
                        ", ".join(f"{k}={v}" for k, v in overrides.items()),
                    )
                else:
                    LOG.info("vLLM client NCCL overrides applied | none")
                with _temporary_env(overrides):
                    LOG.info(
                        "vLLM client NCCL env effective | NCCL_SOCKET_IFNAME=%s | NCCL_P2P_DISABLE=%s | NCCL_IB_DISABLE=%s",
                        os.getenv("NCCL_SOCKET_IFNAME", ""),
                        os.getenv("NCCL_P2P_DISABLE", ""),
                        os.getenv("NCCL_IB_DISABLE", ""),
                    )
                    _shared_init_vllm_client_communicator(client, log=LOG.info)
            self._vllm_client = client
            self._vllm_sync_ready = True
            return True
        except (
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
        ) as exc:  # pragma: no cover - defensive
            LOG.warning("Failed to initialize vLLMClient for weight sync: %s", exc)
            self._vllm_client = None
            self._vllm_sync_ready = False
            helper = getattr(self, "_vllm_helper", None)
            if helper is not None:
                try:
                    setattr(helper, "_vllm_last_sync_ok", False)
                except Exception:
                    pass
            return False

    def _maybe_sync_vllm_weights(self) -> None:
        """Push current model weights to the vLLM server."""
        accelerator = self.ctx.accelerator
        progress_log = _progress_log_enabled()
        rank = getattr(accelerator, "process_index", None)
        world = getattr(accelerator, "num_processes", None)
        sync_start = time.monotonic()
        if progress_log:
            LOG.info("vLLM weight sync start | rank=%s/%s", rank, world)
        try:
            try:
                self._vllm_helper.maybe_sync_weights(
                    ensure_client=self._ensure_vllm_client,
                    sync_model=lambda model: self._sync_model_params_to_vllm(
                        model, accelerator
                    ),
                )
            except TypeError:
                # Allow lightweight stubs without keyword support.
                self._vllm_helper.maybe_sync_weights()
        except Exception as exc:
            if progress_log:
                LOG.warning(
                    "vLLM weight sync failed | rank=%s/%s | seconds=%.2f | error=%s",
                    rank,
                    world,
                    time.monotonic() - sync_start,
                    exc,
                )
            raise
        if progress_log:
            LOG.info(
                "vLLM weight sync done | rank=%s/%s | seconds=%.2f",
                rank,
                world,
                time.monotonic() - sync_start,
            )

    def _invoke_helper(self, attr: str, *args: Any, **kwargs: Any) -> Any:
        """Call a helper attribute if present, preferring public names when available."""
        helper = getattr(self, "_vllm_helper", None)
        if helper is None:
            return None
        fn = getattr(helper, attr, None)
        if not callable(fn) and attr.startswith("_"):
            fn = getattr(helper, attr.lstrip("_"), None)
        if callable(fn):
            return fn(*args, **kwargs)
        return None

    def _sync_model_params_to_vllm(
        self,
        model: Any,
        accelerator: AcceleratorLike,
    ) -> None:
        """Best-effort parameter broadcast mirroring HF GRPO's vLLM path."""
        del accelerator  # handled internally by the shared helper
        result = self._invoke_helper("sync_model_params_to_vllm", model)
        if result is None:
            self._invoke_helper("_sync_model_params_to_vllm", model)

    def _push_param_to_vllm(self, name: str, param: Any) -> None:
        """Send a single parameter tensor to the vLLM client."""
        self._invoke_helper("_push_param_to_vllm", name, param)

    def _reset_vllm_cache(self) -> None:
        """Reset prefix caches when the vLLM client exposes the helper."""
        self._invoke_helper("_reset_vllm_cache")

    def _sync_fsdp_params(self, model: Any) -> None:
        """Iterate FSDP shards and push full parameters to vLLM."""
        self._invoke_helper("_sync_fsdp_params", model)

    def _sync_peft_params(
        self,
        model: Any,
        gather_factory: Callable[[Sequence[Any]], AbstractContextManager[Any]],
    ) -> None:
        """Push merged PEFT adapter weights to vLLM."""
        self._invoke_helper("_sync_peft_params", model, gather_factory)

    def _sync_standard_params(
        self,
        model: Any,
        gather_factory: Callable[[Sequence[Any]], AbstractContextManager[Any]],
    ) -> None:
        """Push standard (non-PEFT/FSDP) parameters to vLLM."""
        self._invoke_helper("_sync_standard_params", model, gather_factory)

    def _resolve_vllm_round_limit(self, requested_n: int) -> int:
        """Decide how many vLLM rounds to run for the current request."""
        result = self._invoke_helper("_resolve_vllm_round_limit", requested_n)
        if isinstance(result, numbers.Real):
            try:
                return int(float(result))
            except (TypeError, ValueError):
                return requested_n
        return requested_n

    @staticmethod
    def _seed_generation_groups(
        prompt_count: int,
        grouped_comps: Optional[List[List[str]]],
        grouped_meta: Optional[List[List[Optional[Any]]]],
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[Any]]]]]:
        """Compatibility wrapper for older tests expecting this helper."""
        return _seed_generation_groups_impl(prompt_count, grouped_comps, grouped_meta)

    @staticmethod
    def _retry_incomplete_prompts(
        helper: "VLLMGenerationMixin",
        prompts: List[str],
        generator: Callable[
            [List[str], int, Optional[List[int]]],
            Tuple[List[List[str]], Optional[List[List[Optional[Any]]]]],
        ],
        expected_generations: int,
        aggregated_comps: List[List[str]],
        aggregated_meta: Optional[List[List[Optional[Any]]]],
        max_retry_rounds: Optional[int],
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[Any]]]]]:
        """Retry helpers retained for backwards compatibility with older tests."""
        del helper  # helper is unused but kept for signature compatibility.
        state = _AggregatedGenerationState(aggregated_comps, aggregated_meta)
        updated = _retry_incomplete_prompts_impl(
            prompts,
            generator,
            expected_generations,
            state,
            max_retry_rounds,
        )
        return updated.completions, updated.metadata

    @staticmethod
    def _summarize_grouped(groups: List[List[str]], limit: int = 8) -> str:
        """Return a compact preview of grouped completions."""
        summary_fn = getattr(VLLMGenerationHelper, "_summarize_grouped", None)
        if callable(summary_fn):
            return str(summary_fn(groups, limit))
        truncated = groups[:limit]
        parts = [f"{idx}:{len(group)}" for idx, group in enumerate(truncated)]
        if len(groups) > limit:
            parts.append(f"+{len(groups) - limit} more")
        return " | ".join(parts)

    def _request_vllm_batch(
        self,
        pending_prompts: List[str],
        request_count: int,
    ) -> Tuple[
        Optional[List[List[str]]],
        Optional[List[List[Optional[VLLMLogprobResult]]]],
    ]:
        """Request completions from vLLM for a subset of prompts."""
        char_limit = self._prompt_char_limit()
        tokenizer = getattr(self.ctx, "tokenizer", None)
        max_tokens = getattr(self.ctx, "max_prompt_len", None)
        truncated = [
            _truncate_prompt(
                prompt,
                char_limit,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
            )
            for prompt in pending_prompts
        ]
        response = self._invoke_vllm_requests(truncated, request_count)
        if response is None:
            return None, None
        grouped, grouped_meta, latency_ms = response
        self._record_vllm_latency(latency_ms)
        pending_count = len(pending_prompts)
        raw_group_count = len(grouped)
        if raw_group_count != pending_count:
            LOG.warning(
                "vLLM raw groups=%d for %d prompts (req_n=%d) | per-group preview: %s",
                raw_group_count,
                pending_count,
                request_count,
                self._summarize_grouped(grouped),
            )
        coalesce_fn = getattr(
            self._vllm_helper,
            "_coalesce_grouped_outputs",
            self._coalesce_grouped_outputs,
        )
        grouped, grouped_meta = coalesce_fn(
            grouped,
            pending_count,
            request_count,
            meta=grouped_meta,
        )
        if len(grouped) == pending_count:
            LOG.warning(
                (
                    "vLLM grouped outputs normalized to %d prompts "
                    "(req_n=%d) | per-prompt lengths=%s"
                ),
                len(grouped),
                request_count,
                [len(entry) for entry in grouped],
            )
            return grouped, grouped_meta
        LOG.warning(
            "vLLM grouped outputs len=%d vs pending=%d | per-prompt lengths=%s",
            len(grouped),
            pending_count,
            [len(entry) for entry in grouped],
        )
        return None, None

    def _record_vllm_latency(self, latency_ms: float) -> None:
        """Track latency metrics for successful vLLM invocations."""
        self._invoke_helper("_record_vllm_latency", latency_ms)

    def _build_vllm_request_kwargs(
        self,
        prompts: List[str],
        request_count: int,
    ) -> Dict[str, Any]:
        """Assemble keyword arguments for ``safe_generate`` requests."""
        kwargs = self._invoke_helper(
            "_build_vllm_request_kwargs", prompts, request_count
        )
        return kwargs if isinstance(kwargs, dict) else {}

    def _invoke_vllm_requests(
        self,
        prompts: List[str],
        request_count: int,
    ) -> Optional[
        Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]], float]
    ]:
        """Call vLLM with retries by splitting large prompt batches."""
        try:
            helpers_mod = sys.modules.get(
                type(self).__module__
            ) or importlib.import_module("maxent_grpo.training.rollout.helpers")
        except ImportError:
            helpers_mod = None
        safe_gen = getattr(helpers_mod, "safe_generate", safe_generate)
        set_safe = getattr(self._vllm_helper, "set_safe_generate", None)
        if callable(set_safe):
            set_safe(safe_gen)
        else:
            setattr(self._vllm_helper, "_safe_generate", safe_gen)
        set_time = getattr(self._vllm_helper, "set_time_provider", None)
        if callable(set_time):
            set_time(getattr(helpers_mod, "time", time))
        else:
            setattr(
                self._vllm_helper,
                "_time",
                getattr(helpers_mod, "time", time),
            )
        result = self._invoke_helper("_invoke_vllm_requests", prompts, request_count)
        return cast(
            Optional[
                Tuple[
                    List[List[str]],
                    Optional[List[List[Optional[VLLMLogprobResult]]]],
                    float,
                ]
            ],
            result,
        )

    def _merge_vllm_results(
        self,
        state: _VLLMGenerationState,
        grouped: List[List[str]],
        grouped_meta: Optional[List[List[Optional[VLLMLogprobResult]]]],
        pending_indices: List[int],
    ) -> None:
        """Append vLLM outputs into the shared state aggregates."""
        self._vllm_helper.merge_vllm_results(
            state,
            grouped,
            grouped_meta,
            pending_indices,
        )

    def _backfill_missing(
        self,
        state: _VLLMGenerationState,
        missing_indices: List[int],
    ) -> None:
        """Generate missing completions locally when vLLM under-delivers."""
        self._vllm_helper.set_fallback_generate(self._generate_local)
        self._vllm_helper.backfill_missing(state, missing_indices)

    def _record_vllm_failure(
        self,
        state: _VLLMGenerationState,
        missing_indices: List[int],
    ) -> None:
        """Log a warning when vLLM fails to deliver even after retries/backfill."""
        self._vllm_helper.record_vllm_failure(state, missing_indices)

    @staticmethod
    def _coalesce_grouped_outputs(
        groups: List[List[str]],
        prompt_count: int,
        requested_n: int,
        meta: Optional[List[List[Optional[VLLMLogprobResult]]]] = None,
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
        """Normalize grouped outputs when vLLM returns per-sample lists."""
        return VLLMGenerationHelper.coalesce_grouped_outputs(
            groups, prompt_count, requested_n, meta
        )

    @staticmethod
    def _merge_group_chunk(
        chunk: List[List[str]],
        meta_chunk: Optional[List[List[Optional[VLLMLogprobResult]]]],
        requested_n: int,
    ) -> Tuple[List[str], Optional[List[Optional[VLLMLogprobResult]]]]:
        """Merge consecutive micro-groups back into per-prompt lists."""
        return VLLMGenerationHelper.merge_group_chunk(chunk, meta_chunk, requested_n)

    def _prepare_vllm_targets(
        self,
        prompts: List[str],
        num_samples: int,
        per_prompt_counts: Optional[List[int]],
    ) -> Tuple[List[str], List[int], Optional[List[int]]]:
        """Resolve target counts and optional dedup mapping for vLLM."""
        return self._vllm_helper.prepare_vllm_targets(
            prompts, num_samples, per_prompt_counts
        )

    def _run_vllm_rounds(self, state: _VLLMGenerationState) -> None:
        """Iteratively request completions until targets are satisfied."""
        try:
            helpers_mod = sys.modules.get(type(self).__module__)
            if helpers_mod is None or not hasattr(helpers_mod, "time"):
                helpers_mod = importlib.import_module(
                    "maxent_grpo.training.rollout.helpers"
                )
        except ImportError:
            helpers_mod = helpers_mod if "helpers_mod" in locals() else None
        set_time = getattr(self._vllm_helper, "set_time_provider", None)
        if callable(set_time):
            set_time(getattr(helpers_mod, "time", time))
        else:
            setattr(self._vllm_helper, "_time", getattr(helpers_mod, "time", time))
        # Allow monkeypatched generator hooks to propagate into the helper.
        helper_exec = getattr(self._vllm_helper, "_execute_vllm_request", None)
        helper_exec_name = getattr(
            getattr(helper_exec, "__func__", helper_exec), "__name__", ""
        )
        if not callable(helper_exec) or helper_exec_name == "_execute_vllm_request":
            set_exec = getattr(self._vllm_helper, "set_request_executor", None)
            if callable(set_exec):
                set_exec(self._execute_vllm_request)
            else:
                setattr(
                    self._vllm_helper,
                    "_execute_vllm_request",
                    self._execute_vllm_request,
                )
        helper_batch = getattr(self._vllm_helper, "_request_vllm_batch", None)
        helper_batch_name = getattr(
            getattr(helper_batch, "__func__", helper_batch), "__name__", ""
        )
        if not callable(helper_batch) or helper_batch_name == "_request_vllm_batch":
            set_batcher = getattr(self._vllm_helper, "set_request_batcher", None)
            if callable(set_batcher):
                set_batcher(self._request_vllm_batch)
            else:
                setattr(
                    self._vllm_helper, "_request_vllm_batch", self._request_vllm_batch
                )
        set_fallback = getattr(self._vllm_helper, "set_fallback_generate", None)
        if callable(set_fallback):
            set_fallback(self._generate_local)
        else:
            setattr(self._vllm_helper, "_fallback_generate", self._generate_local)
        run_rounds = getattr(self._vllm_helper, "run_vllm_rounds", None)
        if callable(run_rounds):
            run_rounds(state)
        else:
            self._invoke_helper("_run_vllm_rounds", state)

    @staticmethod
    def _expand_dedup_results(
        grouped: List[List[str]],
        meta: Optional[List[List[Optional[VLLMLogprobResult]]]],
        mapping: Optional[List[int]],
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
        """Expand de-duplicated prompts back to the original ordering."""
        return VLLMGenerationHelper.expand_dedup_results(grouped, meta, mapping)

    def _generate_with_vllm(
        self,
        prompts: List[str],
        num_samples: int,
        per_prompt_counts: Optional[List[int]] = None,
        *,
        skip_sync: bool = False,
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
        """Generate completions via vLLM, with dedupe/backoff handling."""
        if not prompts:
            return [], None
        # Keep prompt truncation aligned with the legacy helper implementation.
        self.ctx.prompt_char_limit = self._prompt_char_limit()
        accelerator = self.ctx.accelerator
        set_fallback = getattr(self._vllm_helper, "set_fallback_generate", None)
        if callable(set_fallback):
            set_fallback(self._generate_local)
        else:
            setattr(self._vllm_helper, "_fallback_generate", self._generate_local)
        generate_fn = getattr(self._vllm_helper, "generate", None)
        if not callable(generate_fn):
            return [], None
        if skip_sync:
            helper = getattr(self, "_vllm_helper", None)
            sentinel = object()
            prev_sync = sentinel
            swapped = False
            if helper is not None:
                prev_sync = getattr(helper, "__dict__", {}).get(
                    "maybe_sync_weights", sentinel
                )
                if callable(getattr(helper, "maybe_sync_weights", None)):
                    setattr(helper, "maybe_sync_weights", lambda *args, **kwargs: None)
                    swapped = True
            try:
                result = generate_fn(
                    prompts,
                    num_samples,
                    per_prompt_counts,
                    ensure_client=None,
                    sync_model=None,
                )
            finally:
                if swapped and helper is not None:
                    if prev_sync is sentinel:
                        try:
                            delattr(helper, "maybe_sync_weights")
                        except AttributeError:
                            pass
                    else:
                        setattr(helper, "maybe_sync_weights", prev_sync)
        else:
            result = generate_fn(
                prompts,
                num_samples,
                per_prompt_counts,
                ensure_client=self._ensure_vllm_client,
                sync_model=lambda model: self._sync_model_params_to_vllm(
                    model, accelerator
                ),
            )
        if isinstance(result, tuple) and len(result) == 2:
            grouped, meta = result
            if grouped is None:
                grouped = []
            if isinstance(grouped, list):
                return cast(
                    List[List[str]], grouped
                ), cast(Optional[List[List[Optional[VLLMLogprobResult]]]], meta)
        return [], None

    def _execute_vllm_request(
        self,
        state: _VLLMGenerationState,
        pending_indices: List[int],
    ) -> bool:
        """Request completions for specific prompts, grouped by need bucket."""
        exec_fn = getattr(self._vllm_helper, "_execute_vllm_request", None)
        if callable(exec_fn):
            return bool(exec_fn(state, pending_indices))
        return False

    def _flatten_prompts_for_broadcast(
        self,
        prompts: List[str],
        per_prompt_counts: Optional[List[int]] = None,
    ) -> Tuple[List[str], List[int], Optional[List[int]]]:
        result = self._invoke_helper(
            "_flatten_prompts_for_broadcast", prompts, per_prompt_counts
        )
        if isinstance(result, tuple) and len(result) == 3:
            flat_prompts, offsets, flat_counts = result
            if isinstance(flat_prompts, list) and isinstance(offsets, list):
                if flat_counts is None or isinstance(flat_counts, list):
                    return (
                        cast(List[str], flat_prompts),
                        cast(List[int], offsets),
                        cast(Optional[List[int]], flat_counts),
                    )
        return prompts, [], None

    def _broadcast_vllm_payload(
        self,
        flat_prompts: List[str],
        payload: List[Any],
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
        _broadcast_object_list(self.ctx.accelerator, payload, src=0)
        grouped_all, meta_all = payload
        if grouped_all is None:
            grouped_all = [[] for _ in flat_prompts]
        return grouped_all, meta_all

    def _scatter_vllm_payload(
        self,
        flat_prompts: List[str],
        offsets: List[int],
        grouped_all: Optional[List[List[str]]],
        meta_all: Optional[List[List[Optional[VLLMLogprobResult]]]],
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
        """Scatter per-rank slices instead of broadcasting full completions."""
        result = self._invoke_helper(
            "_scatter_vllm_payload", flat_prompts, offsets, grouped_all, meta_all
        )
        if isinstance(result, tuple) and len(result) == 2:
            grouped, meta = result
            if grouped is None:
                return [], cast(Optional[List[List[Optional[VLLMLogprobResult]]]], meta)
            if isinstance(grouped, list):
                return cast(
                    List[List[str]], grouped
                ), cast(Optional[List[List[Optional[VLLMLogprobResult]]]], meta)
            return grouped, meta
        return [], None

    def _pluck_rank_outputs(
        self,
        grouped_all: List[List[str]],
        meta_all: Optional[List[List[Optional[VLLMLogprobResult]]]],
        offsets: List[int],
        prompts: List[str],
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
        result = self._invoke_helper(
            "_pluck_rank_outputs", grouped_all, meta_all, offsets, prompts
        )
        if isinstance(result, tuple) and len(result) == 2:
            grouped, meta = result
            if grouped is None:
                return [], cast(Optional[List[List[Optional[VLLMLogprobResult]]]], meta)
            if isinstance(grouped, list):
                return cast(
                    List[List[str]], grouped
                ), cast(Optional[List[List[Optional[VLLMLogprobResult]]]], meta)
            return grouped, meta
        return [], None

    def _generate_vllm_collective(
        self,
        prompts: List[str],
        num_samples: int,
        per_prompt_counts: Optional[List[int]] = None,
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
        """Run vLLM once on rank 0 and scatter results back to all ranks."""
        self.ctx.prompt_char_limit = self._prompt_char_limit()
        accelerator = self.ctx.accelerator
        if getattr(accelerator, "num_processes", 1) <= 1:
            return self._generate_with_vllm(prompts, num_samples, per_prompt_counts)
        prev_disable_fallback = getattr(self.ctx, "vllm_disable_local_fallback", None)
        try:
            setattr(self.ctx, "vllm_disable_local_fallback", True)
        except Exception:
            prev_disable_fallback = None
        try:
            # Weight sync under ZeRO-3 uses collective gathers; run it on all ranks
            # before non-main processes block waiting for the scatter payload.
            vllm_helper = getattr(self, "_vllm_helper", None)
            maybe_sync = getattr(vllm_helper, "maybe_sync_weights", None) if vllm_helper else None
            sync_weights = bool(getattr(self.ctx, "vllm_sync_weights", False))
            dist = getattr(torch, "distributed", None)
            if (
                getattr(accelerator, "num_processes", 1) > 1
                and dist is not None
                and callable(getattr(dist, "is_available", None))
                and callable(getattr(dist, "is_initialized", None))
                and dist.is_available()
                and dist.is_initialized()
                and callable(getattr(dist, "broadcast_object_list", None))
            ):
                payload = [sync_weights] if accelerator.is_main_process else [False]
                dist.broadcast_object_list(payload, src=0)
                sync_weights = bool(payload[0])
                if sync_weights != bool(getattr(self.ctx, "vllm_sync_weights", False)):
                    try:
                        setattr(self.ctx, "vllm_sync_weights", sync_weights)
                    except Exception:
                        pass
            if callable(maybe_sync) and sync_weights:
                try:
                    maybe_sync(
                        ensure_client=self._ensure_vllm_client,
                        sync_model=lambda model: self._sync_model_params_to_vllm(
                            model, accelerator
                        ),
                    )
                except TypeError:
                    maybe_sync()
            flat_prompts, offsets, flat_counts = self._flatten_prompts_for_broadcast(
                prompts,
                per_prompt_counts,
            )
            grouped_all = None
            meta_all = None
            status_ok = True
            status_err = None
            if bool(getattr(accelerator, "is_main_process", False)):
                try:
                    grouped_all, meta_all = self._generate_with_vllm(
                        flat_prompts,
                        num_samples,
                        flat_counts,
                        # Skip helper-side sync in collective mode; we already handle
                        # weight sync (or intentionally skip it) above on all ranks.
                        skip_sync=True,
                    )
                except Exception as exc:
                    status_ok = False
                    status_err = str(exc)
            dist = getattr(torch, "distributed", None)
            if (
                getattr(accelerator, "num_processes", 1) > 1
                and dist is not None
                and callable(getattr(dist, "is_available", None))
                and callable(getattr(dist, "is_initialized", None))
                and dist.is_available()
                and dist.is_initialized()
                and callable(getattr(dist, "broadcast_object_list", None))
            ):
                payload = (
                    [{"ok": status_ok, "error": status_err}]
                    if accelerator.is_main_process
                    else [{"ok": False, "error": None}]
                )
                dist.broadcast_object_list(payload, src=0)
                status_ok = bool(payload[0].get("ok", False))
                status_err = payload[0].get("error")
            if not status_ok:
                if bool(getattr(self.ctx, "vllm_disable_local_fallback", False)):
                    msg = (
                        "vLLM collective generate failed and local fallback disabled: "
                        f"{status_err}"
                    )
                    if bool(getattr(accelerator, "is_main_process", False)):
                        LOG.error(msg)
                    raise RuntimeError(msg)
                if bool(getattr(accelerator, "is_main_process", False)):
                    LOG.warning(
                        "vLLM collective generate failed on rank 0; "
                        "falling back to local generation on all ranks: %s",
                        status_err,
                    )
                return self._generate_local(
                    prompts, num_samples, per_prompt_counts
                )
            scatter_result = self._scatter_vllm_payload(
                flat_prompts, offsets, grouped_all, meta_all
            )
            if isinstance(scatter_result, tuple):
                if len(scatter_result) != 2:
                    grouped_res, meta_res = [], None
                else:
                    grouped_res = scatter_result[0]
                    meta_res = scatter_result[1]
            else:
                grouped_res, meta_res = scatter_result, None
            if grouped_res is None:
                grouped_res = [[] for _ in prompts]
            return grouped_res, meta_res
        finally:
            if prev_disable_fallback is None:
                try:
                    delattr(self.ctx, "vllm_disable_local_fallback")
                except AttributeError:
                    pass
            else:
                try:
                    setattr(self.ctx, "vllm_disable_local_fallback", prev_disable_fallback)
                except Exception:
                    pass

    def generate(
        self,
        prompts: List[str],
        num_samples: int,
        per_prompt_counts: Optional[List[int]] = None,
    ) -> Tuple[List[List[str]], Optional[List[List[Optional[VLLMLogprobResult]]]]]:
        """Produce completions, preferring vLLM when configured."""
        if per_prompt_counts is not None and len(per_prompt_counts) != len(prompts):
            raise ValueError(
                "per_prompt_counts length must match prompts length in generate()"
            )
        if self.ctx.use_vllm:
            return self._generate_vllm_collective(
                prompts, num_samples, per_prompt_counts
            )
        if not prompts:
            return [], None
        return self._generate_local(prompts, num_samples, per_prompt_counts)


def _gather_object_list(
    accelerator: AcceleratorLike, value: List[Any]
) -> List[List[Any]]:
    """Gather Python lists across ranks with graceful Accelerate fallbacks."""
    gather_fn = getattr(accelerator, "gather_object", None)
    if callable(gather_fn):
        gathered_obj: Any = gather_fn(value)
        if isinstance(gathered_obj, list):
            return cast(List[List[Any]], gathered_obj)
        return [value]
    if dist is not None and dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        gathered: List[List[str]] = [[] for _ in range(world_size)]
        dist.all_gather_object(gathered, value)
        return gathered
    # Single-process fallback
    return [value]


def _broadcast_object_list(
    accelerator: AcceleratorLike, payload: List[Any], *, src: int = 0
) -> None:
    """Broadcast python objects even when Accelerate lacks the helper."""
    broadcast_fn = getattr(accelerator, "broadcast_object_list", None)
    if callable(broadcast_fn):
        broadcast_fn(payload, src)
        return
    if dist is not None and dist.is_available() and dist.is_initialized():
        broadcast = getattr(dist, "broadcast_object_list", None)
        if callable(broadcast):
            broadcast(payload, src)


def _scatter_object(
    accelerator: AcceleratorLike,
    input_list: Optional[List[Any]],
    *,
    src: int = 0,
) -> Any:
    """Scatter python objects from src to all ranks."""
    mode = os.getenv("MAXENT_SCATTER_MODE", "").strip().lower()
    prefer_broadcast = mode in {"broadcast", "bcast", "broadcast_object_list"}
    if accelerator.num_processes <= 1:
        if input_list is None:
            return None
        return input_list[0]
    idx = getattr(accelerator, "process_index", None)
    try:
        if input_list is not None and isinstance(idx, int) and idx >= len(input_list):
            return None
    except (TypeError, ValueError):
        return None
    if dist is not None and dist.is_available() and dist.is_initialized():
        # Prefer broadcast-based scatter when possible. Some environments
        # intermittently hang inside scatter_object_list; broadcasting the full
        # payload is slower but tends to be more reliable.
        broadcast_fn = getattr(dist, "broadcast_object_list", None)
        if prefer_broadcast and callable(broadcast_fn):
            try:
                world_size = int(dist.get_world_size())
            except (RuntimeError, TypeError, ValueError):
                world_size = int(getattr(accelerator, "num_processes", 1) or 1)
            list_ok = input_list is None or (
                isinstance(input_list, list) and len(input_list) == world_size
            )
            if world_size > 0 and list_ok:
                payload = (
                    input_list
                    if accelerator.process_index == src and input_list is not None
                    else [None for _ in range(world_size)]
                )
                try:
                    broadcast_fn(payload, src)
                    return payload[int(getattr(accelerator, "process_index", 0))]
                except (RuntimeError, TypeError, ValueError):
                    return None
    scatter_fn = getattr(accelerator, "scatter_object", None)
    if callable(scatter_fn):
        return scatter_fn(
            input_list if accelerator.process_index == src else None,
            src=src,
        )
    if dist is not None and dist.is_available() and dist.is_initialized():
        broadcast_fn = getattr(dist, "broadcast_object_list", None)
        if callable(broadcast_fn):
            try:
                world_size = int(dist.get_world_size())
            except (RuntimeError, TypeError, ValueError):
                world_size = int(getattr(accelerator, "num_processes", 1) or 1)
            list_ok = input_list is None or (
                isinstance(input_list, list) and len(input_list) == world_size
            )
            if world_size > 0 and list_ok:
                payload = (
                    input_list
                    if accelerator.process_index == src and input_list is not None
                    else [None for _ in range(world_size)]
                )
                try:
                    broadcast_fn(payload, src)
                    return payload[int(getattr(accelerator, "process_index", 0))]
                except (RuntimeError, TypeError, ValueError):
                    return None
        scatter_fn = getattr(dist, "scatter_object_list", None)
        if callable(scatter_fn):
            output: List[Any] = [None]
            try:
                scatter_fn(
                    output,
                    input_list if accelerator.process_index == src else None,
                    src,
                )
            except (RuntimeError, ValueError, TypeError):
                return None
            return output[0]
        return None
    # Fallback to best-effort local selection if no distributed backend is initialized.
    if input_list is None:
        return None
    if idx is None:
        return None
    try:
        if idx >= len(input_list):
            return None
    except (TypeError, ValueError):
        return None
    try:
        return input_list[idx]
    except (IndexError, TypeError):
        return None


def gather_object_list(
    accelerator: AcceleratorLike, value: List[Any]
) -> List[List[Any]]:
    """Public alias for gathering Python objects across ranks."""
    return _gather_object_list(accelerator, value)


def broadcast_object_list(
    accelerator: AcceleratorLike, payload: List[Any], *, src: int = 0
) -> None:
    """Public alias for broadcasting Python objects across ranks."""
    _broadcast_object_list(accelerator, payload, src=src)
    return None


def scatter_object(
    accelerator: AcceleratorLike,
    input_list: Optional[List[Any]],
    *,
    src: int = 0,
) -> Any:
    """Public alias for scattering Python objects across ranks."""
    return _scatter_object(accelerator, input_list, src=src)


__all__ = [
    "VLLMGenerationMixin",
    "_VLLMGenerationState",
    "_broadcast_object_list",
    "broadcast_object_list",
    "_gather_object_list",
    "gather_object_list",
    "_import_vllm_client_cls",
    "_is_peft_model_safe",
    "dist",
    "_optional_import",
    "_scatter_object",
    "scatter_object",
    "_zero3_gather_factory",
]
