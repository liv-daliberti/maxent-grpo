"""Shared vLLM helper utilities reused across generation modules."""

from __future__ import annotations
# pylint: disable=broad-exception-caught

from contextlib import AbstractContextManager, nullcontext
import logging
import os
import threading
import time
from urllib.parse import urlparse
from typing import Any, Callable, Optional, Sequence

from maxent_grpo.utils.imports import optional_import

LOG = logging.getLogger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _is_loopback_host(base_url: Optional[str]) -> bool:
    if not base_url:
        return False
    try:
        parsed = urlparse(base_url)
        host = parsed.hostname or ""
    except Exception:
        host = ""
    if not host:
        host = base_url
    host = host.strip().lower()
    return host in {"localhost", "127.0.0.1", "::1"}


def _resolve_async_mode(async_mode: Optional[bool], base_url: Optional[str]) -> bool:
    if async_mode is not None:
        return async_mode
    raw = os.getenv("MAXENT_VLLM_ASYNC_INIT")
    if raw is None:
        # Match open-r1 behavior by default; async init must be explicitly opted in.
        return False
    raw = raw.strip().lower()
    if raw == "auto":
        return _is_loopback_host(base_url)
    return raw not in {"0", "false", "no", "off"}


def _is_already_initialized_error(exc: BaseException) -> bool:
    """Return True when vLLM reports an already-initialized weight-sync group."""
    text = str(exc).strip().lower()
    if not text:
        return False
    return "weight update group already initialized" in text or (
        "already initialized" in text and "communicator" in text
    )


def zero3_gather_factory(
    accelerator: Any,
    import_fn: Optional[Callable[[str], Any]] = None,
) -> Callable[[Sequence[Any]], AbstractContextManager[Any]]:
    """Return a callable that gathers parameters when ZeRO-3 is active.

    :param accelerator: Accelerate object exposing ``state.deepspeed_plugin``.
    :type accelerator: Any
    :param import_fn: Optional import helper used to lazily import deepspeed.
    :type import_fn: Callable[[str], Any] | None
    :returns: Callable that wraps a parameter sequence in a gather context
        manager, or a no-op ``nullcontext`` when ZeRO-3 is not active.
    :rtype: Callable[[Sequence[Any]], contextlib.AbstractContextManager[Any]]
    """

    importer = import_fn or optional_import
    ds_plugin = getattr(getattr(accelerator, "state", None), "deepspeed_plugin", None)
    zero_stage = getattr(ds_plugin, "zero_stage", 0) or 0
    gather_cls = None
    if zero_stage == 3:
        deepspeed_mod = importer("deepspeed")
        zero_mod = getattr(deepspeed_mod, "zero", None) if deepspeed_mod else None
        gather_cls = getattr(zero_mod, "GatheredParameters", None)

    if gather_cls is None:
        return lambda _params: nullcontext()

    def _factory(params: Sequence[Any]) -> AbstractContextManager[Any]:
        # Prefer limiting full-parameter materialization to rank 0 when the
        # Deepspeed signature supports it, but fall back to the simplest call
        # for older versions and for unit-test stubs.
        try:
            return gather_cls(params, modifier_rank=0)
        except TypeError:
            try:
                return gather_cls(params, 0)
            except TypeError:
                return gather_cls(params)

    return _factory


def import_vllm_client_cls(
    import_fn: Optional[Callable[[str], Any]] = None,
) -> Optional[type]:
    """Return TRL's VLLMClient class if available.

    :param import_fn: Optional import helper to load TRL modules.
    :type import_fn: Callable[[str], Any] | None
    :returns: VLLMClient class when import succeeds, otherwise ``None``.
    :rtype: type | None
    """

    importer = import_fn or optional_import
    vllm_module = importer("trl.extras.vllm_client")
    if vllm_module is None:
        return None
    return getattr(vllm_module, "VLLMClient", None)


def init_vllm_client_communicator(
    client: Any,
    *,
    async_mode: Optional[bool] = None,
    timeout_s: Optional[float] = None,
    log: Optional[Callable[[str], None]] = None,
    init_fn: Optional[Callable[[], Any]] = None,
) -> None:
    """Initialize the vLLM weight-sync communicator with an async-safe handshake.

    The TRL client performs a blocking POST before joining the NCCL group, which
    can deadlock when the server waits for the client to join. This helper sends
    the POST in a background thread, then joins the NCCL group immediately.

    :param client: TRL VLLMClient instance.
    :type client: Any
    :param async_mode: Whether to use the async handshake. When ``None``, the
        ``MAXENT_VLLM_ASYNC_INIT`` env var controls the behavior (default False).
    :type async_mode: bool | None
    :param timeout_s: Timeout for the POST and join wait. Defaults to
        ``MAXENT_VLLM_INIT_TIMEOUT_S`` or 60 seconds.
    :type timeout_s: float | None
    :param log: Optional logger callback for info messages.
    :type log: Callable[[str], None] | None
    """
    resolved_init_fn = init_fn or getattr(client, "init_communicator", None)
    if not callable(resolved_init_fn):
        raise TypeError("VLLMClient.init_communicator is unavailable")
    base_url_hint = getattr(client, "base_url", None)
    async_mode = _resolve_async_mode(async_mode, base_url_hint)

    def _invoke_init_with_fallback() -> None:
        """Invoke ``init_communicator`` across known TRL signature variants."""

        try:
            resolved_init_fn()
            return
        except TypeError as first_exc:
            call_patterns = (
                (client, getattr(client, "base_url", None)),
                (client,),
                (getattr(client, "base_url", None),),
            )
            for args in call_patterns:
                try:
                    resolved_init_fn(*args)
                    return
                except TypeError:
                    continue
            raise first_exc

    def _log(msg: str) -> None:
        if log is not None:
            log(msg)
        else:
            LOG.info(msg)

    def _close_client() -> None:
        close_fn = getattr(client, "close_communicator", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception as exc:
                _log(f"vLLM close_communicator failed (ignored): {exc}")

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

    def _init_once() -> None:
        if not async_mode:
            _invoke_init_with_fallback()
            return

        base_url = getattr(client, "base_url", None)
        host = getattr(client, "host", None)
        group_port = getattr(client, "group_port", None)
        session = getattr(client, "session", None)
        if not base_url or not host or not group_port or session is None:
            _log(
                "Async vLLM init unavailable; falling back to blocking init_communicator."
            )
            _invoke_init_with_fallback()
            return

        timeout = float(
            timeout_s
            if timeout_s is not None
            else os.getenv("MAXENT_VLLM_INIT_TIMEOUT_S", "60")
        )

        requests_mod = optional_import("requests")
        vllm_utils_mod = optional_import("vllm.distributed.utils")
        pynccl_mod = optional_import("vllm.distributed.device_communicators.pynccl")
        if requests_mod is None or vllm_utils_mod is None or pynccl_mod is None:
            _log(
                "Async vLLM init missing dependencies; falling back to blocking init_communicator."
            )
            _invoke_init_with_fallback()
            return

        try:
            resp = session.get(f"{base_url}/get_world_size/")
            resp.raise_for_status()
            vllm_world_size = int(resp.json()["world_size"])
        except Exception as exc:
            raise RuntimeError(f"Failed to query vLLM world size: {exc}") from exc

        world_size = vllm_world_size + 1
        try:
            setattr(client, "rank", vllm_world_size)
        except Exception:
            pass

        init_url = f"{base_url}/init_communicator/"
        payload = {"host": host, "port": int(group_port), "world_size": world_size}
        _log(
            f"Async vLLM init_communicator | url={init_url} | host={host} | "
            f"port={group_port} | world_size={world_size}"
        )

        response_holder: dict[str, Any] = {}

        def _post_init() -> None:
            try:
                response_holder["resp"] = session.post(
                    init_url, json=payload, timeout=timeout
                )
            except Exception as exc:
                response_holder["error"] = exc

        thread = threading.Thread(target=_post_init, daemon=True)
        thread.start()
        try:
            pg = vllm_utils_mod.StatelessProcessGroup.create(
                host=host,
                port=int(group_port),
                rank=vllm_world_size,
                world_size=world_size,
            )
            client.pynccl_comm = pynccl_mod.PyNcclCommunicator(pg, device=0)
        except Exception as exc:
            raise RuntimeError(f"Failed to join vLLM NCCL group: {exc}") from exc

        thread.join(timeout=timeout)
        if "error" in response_holder:
            raise RuntimeError(
                f"vLLM init_communicator POST failed: {response_holder['error']}"
            )
        if "resp" not in response_holder:
            raise TimeoutError(
                f"vLLM init_communicator POST timed out after {timeout:.1f}s"
            )
        resp = response_holder["resp"]
        if getattr(resp, "status_code", None) != 200:
            raise RuntimeError(
                f"vLLM init_communicator POST failed: {resp.status_code} {getattr(resp, 'text', '')}"
            )
        _log("Async vLLM init_communicator completed.")

        close_fn = getattr(client, "close_communicator", None)
        if callable(close_fn):
            import atexit

            atexit.register(close_fn)

    last_exc: Optional[BaseException] = None
    attempt = 0
    used_recoverable_retry = False
    while True:
        attempt += 1
        try:
            _init_once()
            return
        except Exception as exc:
            last_exc = exc
            recoverable = _is_already_initialized_error(exc)
            _log(f"vLLM init_communicator failed (attempt {attempt}): {exc}")
            _close_client()
            should_retry = attempt < retries
            if recoverable and not used_recoverable_retry:
                used_recoverable_retry = True
                should_retry = True
                _log("Detected already-initialized communicator; forced close + retry.")
            if should_retry and backoff_s > 0:
                time.sleep(backoff_s)
            if should_retry:
                continue
            break
    if last_exc is not None:
        raise last_exc


__all__ = [
    "import_vllm_client_cls",
    "init_vllm_client_communicator",
    "zero3_gather_factory",
]
