"""CLI helper utilities shared across entrypoints."""

from __future__ import annotations

from .generate import build_generate_parser

__all__ = ["parse_grpo_args", "build_generate_parser"]


def parse_grpo_args():
    """Lazily import the TRL parser to avoid heavyweight deps at import time."""
    from training.cli import parse_grpo_args as _parse_grpo_args

    return _parse_grpo_args()
