"""Detect and classify vLLM server startup stalls from log text."""

from __future__ import annotations

from argparse import ArgumentParser
from enum import Enum
from pathlib import Path


class StartupStatus(str, Enum):
    """High-level startup state derived from vLLM log lines."""

    STARTING = "starting"
    HEALTHY = "healthy"
    CORE_ENGINE_STALL = "core_engine_stall"
    ERROR = "error"


_READY_MARKERS = (
    "Application startup complete.",
    "Uvicorn running on",
)
_CORE_STALL_MARKER = "Waiting for 1 local, 0 remote core engine proc(s) to start."
_ENGINE_READY_HINTS = (
    "Started server process",
    "Waiting for application startup.",
    "Model loading took",
)
_POST_INIT_STALL_HINTS = (
    "GPU KV cache size:",
    "Maximum concurrency for",
)
_ERROR_MARKERS = (
    "Traceback (most recent call last):",
    "RuntimeError:",
    "ERROR ",
)


def classify_vllm_startup_log(log_text: str, stall_threshold: int = 3) -> StartupStatus:
    """Classify startup progress using marker patterns in ``log_text``."""
    if any(marker in log_text for marker in _READY_MARKERS):
        return StartupStatus.HEALTHY

    if any(marker in log_text for marker in _ERROR_MARKERS):
        return StartupStatus.ERROR

    stall_count = log_text.count(_CORE_STALL_MARKER)
    has_boot_hints = all(marker in log_text for marker in _ENGINE_READY_HINTS)
    if stall_count >= stall_threshold and has_boot_hints:
        return StartupStatus.CORE_ENGINE_STALL
    # vLLM 0.8.x can hang after cache/profile init without emitting the
    # repeated "core engine proc(s)" line; treat that signature as stalled.
    if has_boot_hints and all(marker in log_text for marker in _POST_INIT_STALL_HINTS):
        return StartupStatus.CORE_ENGINE_STALL

    return StartupStatus.STARTING


def should_trigger_v0_fallback(
    log_text: str,
    attempt: int,
    min_attempts: int = 20,
    stall_threshold: int = 3,
) -> bool:
    """Return True when vLLM startup appears stuck and should be relaunched in V0 mode."""
    if attempt < min_attempts:
        return False
    status = classify_vllm_startup_log(log_text, stall_threshold=stall_threshold)
    return status is StartupStatus.CORE_ENGINE_STALL


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Classify vLLM startup health from log content.")
    parser.add_argument("--log", type=Path, required=True, help="Path to the vLLM log file.")
    parser.add_argument("--attempt", type=int, default=0, help="Current health-check attempt index.")
    parser.add_argument(
        "--min-attempts",
        type=int,
        default=20,
        help="Minimum attempt before core-stall fallback is allowed.",
    )
    parser.add_argument(
        "--stall-threshold",
        type=int,
        default=3,
        help="Minimum repeated core-stall lines required for classification.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    text = args.log.read_text(encoding="utf-8", errors="ignore") if args.log.exists() else ""
    if should_trigger_v0_fallback(
        text,
        attempt=args.attempt,
        min_attempts=args.min_attempts,
        stall_threshold=args.stall_threshold,
    ):
        print("fallback_v0")
        return 0
    status = classify_vllm_startup_log(text, stall_threshold=args.stall_threshold)
    print(status.value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
