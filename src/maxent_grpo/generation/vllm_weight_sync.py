"""Weight synchronization helpers split out from the main vLLM helper."""

from __future__ import annotations

import inspect
import logging
from contextlib import nullcontext
from types import SimpleNamespace as _SimpleNamespace
from typing import Any, Callable, Optional, Sequence, cast

from maxent_grpo.generation import vllm_utils as _vllm_utils
from maxent_grpo.utils.fallbacks import optional_import

from maxent_grpo.training.runtime import require_accelerator, require_torch

torch = require_torch("generation_vllm")
Accelerator = require_accelerator("generation_vllm")
LOG = logging.getLogger(__name__)
SimpleNamespace = _SimpleNamespace  # Exposed for tests that monkeypatch this module


def _optional_import(module_name: str) -> Any:
    """Import a module using the shared optional import helper.

    :param module_name: Dotted module path to import.
    :type module_name: str
    :returns: Imported module or ``None`` when unavailable.
    :rtype: Any
    """
    return optional_import(module_name)


def _import_vllm_client_cls(
    import_fn: Optional[Callable[[str], Any]] = None,
) -> Optional[type]:
    """Return TRL's VLLMClient using the provided import helper.

    :param import_fn: Optional import helper; defaults to ``_optional_import``.
    :type import_fn: Callable[[str], Any] | None
    :returns: VLLMClient class when import succeeds, otherwise ``None``.
    :rtype: type | None
    """

    return _vllm_utils.import_vllm_client_cls(import_fn or _optional_import)


def _zero3_gather_factory(
    accelerator: Accelerator,
) -> Callable[[Sequence[Any]], Any]:
    """Return a callable that gathers parameters when ZeRO-3 is active.

    :param accelerator: Accelerate instance exposing ``state.deepspeed_plugin``.
    :type accelerator: accelerate.Accelerator
    :returns: Callable producing a gather context manager for ZeRO-3, or a
        no-op when ZeRO-3 is not enabled.
    :rtype: Callable[[Sequence[Any]], Any]
    """

    return _vllm_utils.zero3_gather_factory(accelerator, import_fn=_optional_import)


def _is_peft_model_safe(target: Any) -> bool:
    """Return ``True`` if accelerate.utils reports that the model uses PEFT adapters.

    :param target: Model instance to inspect.
    :type target: Any
    :returns: Whether the model appears to be PEFT-wrapped.
    :rtype: bool
    """
    accelerate_utils = _optional_import("accelerate.utils")
    if accelerate_utils is None:
        return False
    is_peft_model = getattr(accelerate_utils, "is_peft_model", None)
    if not callable(is_peft_model):
        return False
    try:
        return bool(is_peft_model(target))
    except (TypeError, AttributeError, ValueError):
        return False


class _ClientCallable:
    """Lightweight callable wrapper to keep static analyzers satisfied."""

    def __init__(self, func: Callable[..., Any]) -> None:
        """Wrap a callable to guard attribute access in static typing.

        :param func: Callable to wrap.
        :type func: Callable[..., Any]
        """
        self._func = func

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Forward the call to the wrapped function.

        :param args: Positional arguments forwarded to ``func``.
        :type args: tuple
        :param kwargs: Keyword arguments forwarded to ``func``.
        :type kwargs: dict
        :returns: Result of the wrapped function.
        :rtype: Any
        """
        return self._func(*args, **kwargs)


class VLLMWeightSyncMixin:
    """Group weight sync helpers separately from retry/resilience logic."""

    _vllm_client: Any
    _vllm_sync_ready: bool
    _last_vllm_synced_step: Optional[int]
    _fsdp_cls: Any
    _gather_factory: Any
    ctx: Any

    def _vllm_base_url(self, url: str) -> str:
        """Strip common ``/generate`` suffixes from the vLLM endpoint.

        :param url: Full URL configured for the vLLM server.
        :type url: str
        :returns: Base URL without trailing ``/generate`` paths.
        :rtype: str
        """
        from urllib.parse import urlparse

        try:
            parsed = urlparse(url)
        except ValueError:
            parsed = None
        if parsed is not None and parsed.scheme and parsed.netloc:
            base = f"{parsed.scheme}://{parsed.netloc}"
            return base.rstrip("/")
        if "/generate" in url:
            return url.split("/generate", 1)[0].rstrip("/")
        return url.rstrip("/")

    def _ensure_vllm_client(
        self, import_vllm_client_cls: Optional[Callable[[], Any]] = None
    ) -> bool:
        """Return True when the TRL VLLMClient is ready for weight sync.

        :param import_vllm_client_cls: Optional callable that imports and
            returns the TRL ``VLLMClient`` class.
        :type import_vllm_client_cls: Callable[[], Any] | None
        :returns: Whether weight sync is ready to proceed on this rank.
        :rtype: bool
        """
        ctx = self.ctx
        if not ctx.vllm_sync_weights or not ctx.accelerator.is_main_process:
            return False
        if self._vllm_client is not None and self._vllm_sync_ready:
            return True
        import_fn = import_vllm_client_cls or getattr(
            self, "_import_vllm_client_cls", _import_vllm_client_cls
        )
        client_cls = import_fn()
        if client_cls is None:
            LOG.warning(
                "vLLM weight sync requested but TRL VLLMClient is unavailable; skipping."
            )
            self._vllm_sync_ready = False
            return False
        if not callable(client_cls):
            self._vllm_client = None
            self._vllm_sync_ready = False
            return False
        try:
            base_url = self._vllm_base_url(ctx.vllm_url)
            try:
                self._vllm_client = client_cls(base_url=base_url)
            except TypeError:
                self._vllm_client = client_cls()
            init_attr = getattr(self._vllm_client, "init_communicator", None)
            if not callable(init_attr):
                LOG.warning(
                    "vLLM weight sync requested but TRL VLLMClient is unavailable; skipping."
                )
                self._vllm_client = None
                self._vllm_sync_ready = False
                return False
            init_fn = cast(Callable[..., Any], init_attr)

            def _call_init(fn: Callable[..., Any]) -> None:
                # Some stubs may expect different signatures; try common variants explicitly.
                try:
                    fn()
                except TypeError:
                    try:
                        fn(self._vllm_client)
                    except TypeError:
                        fn(self._vllm_client, base_url)

            _call_init(init_fn)
            self._vllm_sync_ready = True
            return True
        except (
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:  # pragma: no cover - network dependent
            LOG.warning("Failed to initialize vLLMClient for weight sync: %s", exc)
            self._vllm_client = None
            self._vllm_sync_ready = False
            return False

    def maybe_sync_weights(
        self,
        ensure_client: Optional[Callable[[], bool]] = None,
        sync_model: Optional[Callable[[Any], None]] = None,
    ) -> None:
        """Synchronize weights to the vLLM server if configured.

        :param ensure_client: Optional callable that prepares the vLLM client.
        :type ensure_client: Callable[[], bool] | None
        :param sync_model: Optional callable invoked to push model weights.
        :type sync_model: Callable[[Any], None] | None
        """
        ensure_fn = ensure_client or self._ensure_vllm_client
        if not ensure_fn():
            return
        current_step = self.ctx.generation_stats.get("current_step")
        if current_step is not None and self._last_vllm_synced_step == int(
            current_step
        ):
            return
        accelerator = self.ctx.accelerator
        try:
            model = accelerator.unwrap_model(self.ctx.model)
        except (AttributeError, TypeError):
            model = self.ctx.model
        sync_fn = sync_model or self._sync_model_params_to_vllm
        visited: set[str] = set()
        try:
            try:
                sig = inspect.signature(sync_fn)
                accepts_visited = "visited" in sig.parameters
            except (TypeError, ValueError):
                accepts_visited = False
            if accepts_visited:
                sync_fn(model, visited=visited)
            else:
                sync_fn(model)
            stats = self.ctx.generation_stats
            stats["vllm_weight_syncs"] = int(stats.get("vllm_weight_syncs", 0)) + 1
            if current_step is not None:
                self._last_vllm_synced_step = int(current_step)
        except (
            RuntimeError,
            ValueError,
        ) as exc:  # pragma: no cover - runtime dependent
            LOG.warning("Skipping vLLM weight sync due to error: %s", exc)
        wait_for_all = getattr(accelerator, "wait_for_everyone", None)
        if callable(wait_for_all):
            wait_for_all()

    def _client_callable(self, attr_name: str) -> Optional[_ClientCallable]:
        """Return a callable attribute from the vLLM client if available.

        :param attr_name: Attribute name to fetch from the client.
        :type attr_name: str
        :returns: Wrapped callable or ``None`` when missing.
        :rtype: _ClientCallable | None
        """
        client = self._vllm_client
        if client is None:
            return None
        candidate = getattr(client, attr_name, None)
        if not callable(candidate):
            return None
        return _ClientCallable(candidate)

    def _sync_model_params_to_vllm(
        self,
        model: Any,
        visited: Optional[set[str]] = None,
    ) -> None:
        """Push model parameters to the vLLM side, handling FSDP/PEFT cases.

        :param model: Model instance whose parameters should be synchronized.
        :type model: Any
        :param visited: Optional set of parameter names that have already been
            synchronized (used for recursion).
        :type visited: set[str] | None
        """
        fsdp_cls = self._fsdp_cls
        visited = visited if visited is not None else set()

        def _has_summon_full_params(target: Any) -> bool:
            try:
                return callable(getattr(target, "summon_full_params"))
            except (AttributeError, RuntimeError, TypeError, ValueError):
                return False

        class_summon = getattr(type(model), "summon_full_params", None)
        if fsdp_cls is None:
            fsdp_mod = getattr(getattr(torch, "distributed", None), "fsdp", None)
            fsdp_cls = (
                getattr(fsdp_mod, "FullyShardedDataParallel", None)
                if fsdp_mod
                else None
            )
        has_summon = _has_summon_full_params(model)
        if has_summon and callable(class_summon):
            if fsdp_cls is None or not isinstance(model, fsdp_cls):
                fsdp_cls = type(model)
        if fsdp_cls is not None and (
            self._fsdp_cls is None or not isinstance(model, self._fsdp_cls)
        ):
            self._fsdp_cls = fsdp_cls
        if fsdp_cls is not None and isinstance(model, fsdp_cls):
            children = list(getattr(model, "named_children", None)())  # type: ignore[arg-type]
            modules_to_sync = children or [("", model)]
            for base_name, base_module in modules_to_sync:
                named_params = getattr(base_module, "named_parameters", None)
                if not callable(named_params):
                    continue
                for pname, param in named_params():
                    full_name = f"{base_name}.{pname}" if base_name else pname
                    for extra in (
                        "_fsdp_wrapped_module.",
                        "_checkpoint_wrapped_module.",
                    ):
                        full_name = full_name.replace(extra, "")
                    if full_name in visited:
                        continue
                    visited.add(full_name)
                    self._push_param_to_vllm(full_name, param)
            self._reset_vllm_cache()
            return
        if not has_summon:
            has_summon = _has_summon_full_params(model)
            if callable(class_summon) and (
                fsdp_cls is None or not isinstance(model, fsdp_cls)
            ):
                fsdp_cls = type(model)
                if self._fsdp_cls is None or not isinstance(model, self._fsdp_cls):
                    self._fsdp_cls = fsdp_cls
        if fsdp_cls is not None and isinstance(model, fsdp_cls):
            children = list(getattr(model, "named_children", None)())  # type: ignore[arg-type]
            modules_to_sync = children or [("", model)]
            for base_name, base_module in modules_to_sync:
                named_params = getattr(base_module, "named_parameters", None)
                if not callable(named_params):
                    continue
                for pname, param in named_params():
                    full_name = f"{base_name}.{pname}" if base_name else pname
                    for extra in (
                        "_fsdp_wrapped_module.",
                        "_checkpoint_wrapped_module.",
                    ):
                        full_name = full_name.replace(extra, "")
                    if full_name in visited:
                        continue
                    visited.add(full_name)
                    self._push_param_to_vllm(full_name, param)
            self._reset_vllm_cache()
            return
        if has_summon:

            def _walk(module: Any, prefix: str = "") -> None:
                named_children = getattr(module, "named_children", None)
                children = list(named_children()) if callable(named_children) else []
                named_params = getattr(module, "named_parameters", None)
                if callable(named_params):
                    for raw_name, param in named_params():
                        if param is None:
                            continue
                        clean = raw_name
                        for extra in (
                            "_fsdp_wrapped_module.",
                            "_checkpoint_wrapped_module.",
                        ):
                            clean = clean.replace(extra, "")
                        full_name = f"{prefix}.{clean}" if prefix else clean
                        if children and any(
                            raw_name.startswith(extra)
                            for extra in (
                                "_fsdp_wrapped_module.",
                                "_checkpoint_wrapped_module.",
                            )
                        ):
                            continue
                        if full_name in visited:
                            continue
                        visited.add(full_name)
                        self._push_param_to_vllm(full_name, param)
                for child_name, child in children:
                    child_prefix = f"{prefix}.{child_name}" if prefix else child_name
                    _walk(child, child_prefix)

            _walk(model)
            root_params = getattr(model, "named_parameters", None)
            if callable(root_params):
                for raw_name, param in root_params():
                    clean = raw_name
                    for extra in (
                        "_fsdp_wrapped_module.",
                        "_checkpoint_wrapped_module.",
                    ):
                        clean = clean.replace(extra, "")
                    if any(
                        raw_name.startswith(extra)
                        for extra in (
                            "_fsdp_wrapped_module.",
                            "_checkpoint_wrapped_module.",
                        )
                    ):
                        continue
                    if clean in visited:
                        continue
                    visited.add(clean)
                    self._push_param_to_vllm(clean, param)
            self._reset_vllm_cache()
            return
        is_peft_fn = getattr(self, "_is_peft_model_safe", _is_peft_model_safe)
        if is_peft_fn(model):
            self._sync_peft_params(model)
            self._reset_vllm_cache()
            return
        self._sync_standard_params(model)
        self._reset_vllm_cache()

    def _push_param_to_vllm(self, name: str, param: Any) -> None:
        """Send a single parameter tensor to the vLLM client if available.

        :param name: Fully qualified parameter name.
        :type name: str
        :param param: Tensor to push to the vLLM server.
        :type param: Any
        """
        if param is None:
            return
        update_fn = self._client_callable("update_named_param")
        if update_fn is None:
            return
        try:
            update_fn(name, param.data)
        except (
            RuntimeError,
            ValueError,
        ) as exc:  # pragma: no cover - network dependent
            LOG.warning("Failed to push param %s to vLLM: %s", name, exc)

    def _reset_vllm_cache(self) -> None:
        """Reset the vLLM prefix cache if the client exposes the hook."""
        reset_fn = self._client_callable("reset_prefix_cache")
        if reset_fn is None:
            return
        try:
            reset_fn()
        except (RuntimeError, ValueError, AttributeError):
            return

    def _sync_standard_params(
        self,
        model: Any,
        gather_factory: Optional[Callable[[Sequence[Any]], Any]] = None,
        prefix: str = "",
        visited: Optional[set[str]] = None,
    ) -> None:
        """Synchronize standard (non-FSDP/PEFT) model parameters.

        :param model: Model instance whose parameters are being pushed.
        :type model: Any
        :param gather_factory: Optional context manager factory for ZeRO-3.
        :type gather_factory: Callable[[Sequence[Any]], Any] | None
        :param prefix: Name prefix to prepend to parameters.
        :type prefix: str
        :param visited: Parameter names already synced to avoid duplicates.
        :type visited: set[str] | None
        """
        params = list(model.parameters()) if hasattr(model, "parameters") else []
        factory = gather_factory or self._gather_factory or (lambda _p: nullcontext())
        try:
            ctx = factory(params)
        except NameError:
            ctx = nullcontext()
        visited = visited if visited is not None else set()
        named_params = getattr(model, "named_parameters", None)
        with ctx:
            if callable(named_params):
                for name, param in named_params():
                    if param is None:
                        continue
                    clean = f"{prefix}{name}" if prefix else name
                    for extra in (
                        "_fsdp_wrapped_module.",
                        "_checkpoint_wrapped_module.",
                    ):
                        clean = clean.replace(extra, "")
                    if clean in visited:
                        continue
                    visited.add(clean)
                    self._push_param_to_vllm(clean, param)
        named_children = getattr(model, "named_children", None)
        if callable(named_children):
            for child_name, child in named_children():
                child_prefix = f"{prefix}{child_name}."
                self._sync_standard_params(
                    child, gather_factory, child_prefix, visited=visited
                )

    def _sync_peft_params(
        self,
        model: Any,
        gather_factory: Optional[Callable[[Sequence[Any]], Any]] = None,
    ) -> None:
        """Synchronize PEFT adapter parameters to the vLLM server.

        :param model: PEFT model instance.
        :type model: Any
        :param gather_factory: Optional context manager factory for ZeRO-3.
        :type gather_factory: Callable[[Sequence[Any]], Any] | None
        """
        merge_fn = getattr(model, "merge_adapter", None)
        unmerge_fn = getattr(model, "unmerge_adapter", None)
        params = list(model.parameters())
        factory = gather_factory or self._gather_factory or (lambda _p: nullcontext())
        with factory(params):
            if callable(merge_fn):
                merge_fn()
            for name, param in model.named_parameters():
                clean = (
                    name.replace("modules_to_save.default.", "")
                    .replace("base_model.model.", "")
                    .replace(".base_layer", "")
                )
                if getattr(model, "prefix", None) and str(model.prefix) in clean:
                    continue
                if "original_module" in clean:
                    continue
                self._push_param_to_vllm(clean, param)
            if callable(unmerge_fn):
                unmerge_fn()

    def _sync_fsdp_params(
        self,
        module: Any,
        gather_factory: Optional[Callable[[Sequence[Any]], Any]] = None,
        prefix: str = "",
        fsdp_cls: Any = None,
        visited: Optional[set[str]] = None,
    ) -> None:
        """Synchronize parameters for FSDP-wrapped modules.

        :param module: Module wrapped by FullyShardedDataParallel.
        :type module: Any
        :param gather_factory: Optional context manager factory for ZeRO-3.
        :type gather_factory: Callable[[Sequence[Any]], Any] | None
        :param prefix: Prefix to prepend to parameter names.
        :type prefix: str
        :param fsdp_cls: FSDP class used to detect wrapped modules.
        :type fsdp_cls: Any
        :param visited: Parameter names already synced to avoid duplicates.
        :type visited: set[str] | None
        """
        fsdp_cls = fsdp_cls or self._fsdp_cls
        if fsdp_cls is None:
            return
        try:
            params = list(module.parameters()) if hasattr(module, "parameters") else []
        except AttributeError:
            params = []
        visited = visited or set()
        factory = gather_factory or self._gather_factory or (lambda _p: nullcontext())
        with factory(params):
            named_params = getattr(module, "named_parameters", None)
            if callable(named_params):
                for name, param in named_params():
                    full_name = f"{prefix}{name}" if prefix else name
                    for extra in (
                        "_fsdp_wrapped_module.",
                        "_checkpoint_wrapped_module.",
                    ):
                        full_name = full_name.replace(extra, "")
                    if full_name in visited:
                        continue
                    visited.add(full_name)
                    self._push_param_to_vllm(full_name, param)
            named_children = getattr(module, "named_children", None)
            if callable(named_children):
                for child_name, child in named_children():
                    child_prefix = f"{prefix}{child_name}."
                    self._sync_fsdp_params(
                        child,
                        gather_factory=gather_factory,
                        prefix=child_prefix,
                        fsdp_cls=fsdp_cls,
                        visited=visited,
                    )
        if isinstance(module, fsdp_cls) and callable(
            getattr(module, "named_parameters", None)
        ):
            with fsdp_cls.summon_full_params(module, recurse=False, writeback=False):
                for pname, param in module.named_parameters():
                    full_name = f"{prefix}{pname}" if prefix else pname
                    for extra in (
                        "_fsdp_wrapped_module.",
                        "_checkpoint_wrapped_module.",
                    ):
                        full_name = full_name.replace(extra, "")
                    if full_name in visited:
                        continue
                    visited.add(full_name)
                    self._push_param_to_vllm(full_name, param)


__all__ = [
    "Accelerator",
    "VLLMWeightSyncMixin",
    "_ClientCallable",
    "_import_vllm_client_cls",
    "_is_peft_model_safe",
    "_optional_import",
    "_zero3_gather_factory",
]
