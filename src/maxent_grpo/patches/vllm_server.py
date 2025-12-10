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
is attached automatically from ``ops/sitecustomize.py`` when the vLLM
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
    except Exception:  # pragma: no cover - FastAPI/Starlette missing
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
        except Exception as exc:  # pragma: no cover - optional dependency
            LOG.debug("vLLM OpenAI server unavailable; skip client_tag middleware: %s", exc)
            return False
    if app is None:
        return False

    state = getattr(app, "state", None)
    if state is not None and getattr(state, "_maxent_client_tag_installed", False):
        return False

    try:
        from starlette.middleware.base import BaseHTTPMiddleware
    except Exception as exc:  # pragma: no cover - FastAPI/Starlette missing
        LOG.warning("Starlette unavailable; cannot install client_tag middleware: %s", exc)
        return False

    class _ClientTagMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):  # type: ignore[override]
            client_tag = request.headers.get("X-VLLM-Client-Tag", "")
            response = await call_next(request)
            if (
                not client_tag
                or request.method != "POST"
                or not str(getattr(request.url, "path", "")).endswith("/generate")
            ):
                return response
            content_type = (response.headers.get("content-type") or "").lower()
            if "application/json" not in content_type:
                return response

            body = b""
            try:
                async for chunk in response.body_iterator:
                    body += chunk
            except Exception as exc:  # pragma: no cover - defensive
                LOG.warning("Failed to read vLLM response body for tag propagation: %s", exc)
                return response

            if not body:
                return _build_passthrough_response(response, body)
            try:
                payload = json.loads(body)
            except Exception:
                return _build_passthrough_response(response, body)

            updated = _propagate_client_tag(payload, client_tag)
            new_body = json.dumps(payload).encode("utf-8") if updated else body
            return _build_passthrough_response(response, new_body)

    app.add_middleware(_ClientTagMiddleware)
    if state is not None:
        setattr(state, "_maxent_client_tag_installed", True)
    LOG.info("Installed vLLM client_tag middleware for response sharding.")
    return True


__all__ = ["install_vllm_client_tag_middleware"]
