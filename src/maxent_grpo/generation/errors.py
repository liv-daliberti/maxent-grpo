"""Shared service error payloads for external generation backends."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ServiceErrorPayload:
    """Structured metadata capturing a failed external service request."""

    service: str
    endpoint: str
    model: Optional[str]
    prompt_count: int
    payload_chars: int
    payload_size_bytes: Optional[int]
    status_code: Optional[int]
    attempt: int
    max_attempts: int
    exception_type: str
    exception_message: str
    request_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dictionary for structured logging."""

        trimmed_message = self.exception_message or ""
        if len(trimmed_message) > 512:
            trimmed_message = trimmed_message[:512]
        return {
            "service": self.service,
            "endpoint": self.endpoint,
            "model": self.model,
            "prompt_count": self.prompt_count,
            "payload_chars": self.payload_chars,
            "payload_size_bytes": self.payload_size_bytes,
            "status_code": self.status_code,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "exception_type": self.exception_type,
            "exception_message": trimmed_message,
            "request_id": self.request_id,
            "extra": dict(self.extra),
        }

    def to_json(self) -> str:
        """Serialize the payload to JSON for log aggregation."""

        return json.dumps(self.to_dict(), sort_keys=True)

    def copy_with(self, **updates: Any) -> "ServiceErrorPayload":
        """Return a shallow copy with the provided field overrides."""

        data = asdict(self)
        extra_updates = updates.pop("extra", None)
        if extra_updates:
            merged_extra = dict(data.get("extra", {}))
            merged_extra.update(extra_updates)
            data["extra"] = merged_extra
        elif "extra" not in data:
            data["extra"] = {}
        data.update(updates)
        return ServiceErrorPayload(**data)


class GenerationServiceError(RuntimeError):
    """Raised when external generation services exhaust their retries."""

    def __init__(self, message: str, payload: ServiceErrorPayload) -> None:
        super().__init__(message)
        self.payload = payload

    def to_dict(self) -> Dict[str, Any]:
        """Structured representation including the human-readable message."""

        info = self.payload.to_dict()
        info["message"] = str(self)
        return info

    def to_json(self) -> str:
        """JSON string suitable for structured log/event ingestion."""

        return json.dumps(self.to_dict(), sort_keys=True)


def log_generation_service_error(
    logger: logging.Logger, stage: str, error: GenerationServiceError
) -> None:
    """Emit a structured log entry for a failed generation call."""

    try:
        payload = error.payload.to_json()
    except (TypeError, ValueError):  # pragma: no cover - fallback
        payload = str(error.payload.to_dict())
    logger.error(
        "Generation service failure (%s) | payload=%s",
        stage,
        payload,
    )


__all__ = ["GenerationServiceError", "ServiceErrorPayload", "log_generation_service_error"]
