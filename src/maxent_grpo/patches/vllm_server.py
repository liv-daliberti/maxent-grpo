"""
Server-side helpers for sharding vLLM responses by ``client_tag``.

The MaxEnt trainer forwards a ``client_tag`` per Accelerate rank via both
the JSON payload and the ``X-VLLM-Client-Tag`` header.  Shared vLLM
servers (or lightweight reverse proxies) must echo this tag inside each
result group and completion output so downstream clients can filter
responses per rank.  Without the echo every rank receives the union of
all completions, so the trainer discards them and the loss falls to zero.

This module provides a FastAPI middleware that injects the ``client_tag``
header into every JSON response emitted by ``/generate``.  The middleware
is attached automatically from ``sitecustomize.py`` when the vLLM
OpenAI server is imported, but can also be installed manually when
constructing a custom proxy.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, Optional

LOG = logging.getLogger(__name__)

# Keys indicating nested result groups inside vLLM responses.
_GROUP_KEYS: tuple[str, ...] = ("results", "outputs", "choices", "data")
# Additional hints that mark a payload as a completion object.
_PAYLOAD_HINTS: tuple[str, ...] = (
    "metadata",
    "text",
    "content",
    "response",
    "message",
    "messages",
)


def _ensure_metadata(entry: Dict[str, Any], client_tag: str) -> bool:
    """
    Guarantee that ``entry['metadata']['client_tag']`` is set.

    :param entry: Completion/result dictionary.
    :param client_tag: Tag associated with the inbound request.
    :returns: ``True`` when the entry was updated, ``False`` otherwise.
    """

    metadata = entry.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        entry["metadata"] = metadata
        changed = True
    else:
        changed = False
    if metadata.get("client_tag") != client_tag:
        metadata["client_tag"] = client_tag
        changed = True
    return changed


def _should_tag(entry: Dict[str, Any]) -> bool:
    """Return True when ``entry`` looks like a completion or result block."""

    for key in _PAYLOAD_HINTS:
        if key in entry:
            return True
    for key in _GROUP_KEYS:
        if key in entry:
            return True
    return False


def _iter_nested_entries(entry: Dict[str, Any]) -> Iterable[Any]:
    """Yield nested structures that may hold completions."""

    for key in _GROUP_KEYS:
        value = entry.get(key)
        if isinstance(value, list):
            yield from value
        elif isinstance(value, dict):
            yield value


def _propagate_client_tag(payload: Any, client_tag: str, depth: int = 0) -> bool:
    """
    Recursively attach ``client_tag`` to result groups and outputs.

    :param payload: Parsed JSON response emitted by vLLM.
    :param client_tag: Tag extracted from the incoming request/header.
    :param depth: Recursion guard to avoid pathological payloads.
    :returns: ``True`` when any portion of the payload was updated.
    """

    if depth > 10:
        return False
    changed = False
    if isinstance(payload, dict):
        if _should_tag(payload):
            if _ensure_metadata(payload, client_tag):
                changed = True
        for nested in _iter_nested_entries(payload):
            if _propagate_client_tag(nested, client_tag, depth + 1):
                changed = True
    elif isinstance(payload, list):
        for entry in payload:
            if _propagate_client_tag(entry, client_tag, depth + 1):
                changed = True
    return changed


def _build_passthrough_response(response: Any, content: bytes) -> Any:
    """Clone *response* while replacing the payload with *content*."""

    try:
        from starlette.responses import Response
    except (ImportError, ModuleNotFoundError):  # pragma: no cover - FastAPI/Starlette missing
        return response

    headers = dict(getattr(response, "headers", {}))
    headers["content-length"] = str(len(content))
    background = getattr(response, "background", None)
    media_type = getattr(response, "media_type", None)
    return Response(
        content=content,
        status_code=getattr(response, "status_code", 200),
        headers=headers,
        media_type=media_type or "application/json",
        background=background,
    )


def install_vllm_client_tag_middleware(app: Optional[Any] = None) -> bool:
    """
    Attach the middleware that copies ``client_tag`` into vLLM responses.

    :param app: Optional FastAPI app instance. When absent the helper
        attempts to import ``vllm.entrypoints.openai.api_server`` and
        patch its global ``app`` reference.
    :returns: ``True`` when middleware was installed, ``False`` otherwise.
    """

    if app is None:
        try:  # pragma: no cover - exercised in runtime, not unit tests
            from vllm.entrypoints.openai import api_server as vllm_api

            app = getattr(vllm_api, "app", None)
        except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - optional dependency
            LOG.debug("vLLM OpenAI server unavailable; skip client_tag middleware: %s", exc)
            return False
    if app is None:
        return False

    state = getattr(app, "state", None)
    if state is not None and getattr(state, "_maxent_client_tag_installed", False):
        return False

    try:
        from starlette.types import ASGIApp, Message, Receive, Scope, Send
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - FastAPI/Starlette missing
        LOG.warning(
            "Starlette unavailable; cannot install client_tag middleware: %s", exc
        )
        return False

    def _header_map(headers: list[tuple[bytes, bytes]]) -> dict[str, str]:
        mapped: dict[str, str] = {}
        for key, value in headers:
            try:
                mapped[key.decode("latin-1").lower()] = value.decode("latin-1")
            except (AttributeError, TypeError, UnicodeDecodeError):
                continue
        return mapped

    class _ClientTagMiddleware:
        """ASGI middleware that injects ``client_tag`` into JSON responses.

        This is implemented as raw ASGI middleware (not BaseHTTPMiddleware)
        because Starlette's BaseHTTPMiddleware can deadlock on streaming
        responses when the middleware consumes ``body_iterator``.
        """

        def __init__(self, inner: ASGIApp) -> None:
            self._app = inner

        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            if scope.get("type") != "http":
                await self._app(scope, receive, send)
                return

            method = (scope.get("method") or "").upper()
            path = str(scope.get("path") or "")
            if method != "POST" or not path.endswith("/generate"):
                await self._app(scope, receive, send)
                return

            headers = _header_map(list(scope.get("headers") or []))
            client_tag = headers.get("x-vllm-client-tag", "").strip()
            if not client_tag:
                await self._app(scope, receive, send)
                return

            start_message: Message | None = None
            body_chunks: list[bytes] = []
            response_sent = False

            async def send_wrapper(message: Message) -> None:
                nonlocal start_message, body_chunks, response_sent

                if message.get("type") == "http.response.start":
                    start_message = message
                    return
                if message.get("type") != "http.response.body":
                    await send(message)
                    return

                body_chunks.append(message.get("body", b"") or b"")
                if message.get("more_body"):
                    return

                # Final body chunk: rewrite (if possible) and flush.
                if start_message is None:
                    await send(message)
                    return

                start_headers = list(start_message.get("headers") or [])
                header_lookup = _header_map(start_headers)
                content_type = (header_lookup.get("content-type") or "").lower()
                body = b"".join(body_chunks)

                new_body = body
                if body and "application/json" in content_type:
                    try:
                        payload = json.loads(body)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        payload = None
                    if payload is not None:
                        updated = _propagate_client_tag(payload, client_tag)
                        if updated:
                            try:
                                new_body = json.dumps(payload).encode("utf-8")
                            except (TypeError, ValueError):
                                new_body = body

                # Update content-length for the buffered payload.
                filtered_headers = [
                    (k, v)
                    for k, v in start_headers
                    if k.lower() != b"content-length"
                ]
                filtered_headers.append(
                    (b"content-length", str(len(new_body)).encode("ascii"))
                )
                start_message = dict(start_message)
                start_message["headers"] = filtered_headers
                await send(start_message)
                await send({"type": "http.response.body", "body": new_body, "more_body": False})
                response_sent = True
                start_message = None
                body_chunks = []

            await self._app(scope, receive, send_wrapper)
            if not response_sent and start_message is not None:
                # Some responses may not send a body (e.g., HEAD). Ensure the
                # start message isn't swallowed.
                await send(start_message)
                await send({"type": "http.response.body", "body": b"", "more_body": False})

    app.add_middleware(_ClientTagMiddleware)
    if state is not None:
        setattr(state, "_maxent_client_tag_installed", True)
    LOG.info("Installed vLLM client_tag middleware for response sharding.")
    return True


__all__ = ["install_vllm_client_tag_middleware"]
