"""Compatibility shim that re-exports vLLM helpers from :mod:`generation.vllm`."""

from __future__ import annotations

from generation.vllm import VLLMGenerationHelper, _VLLMGenerationState

__all__ = ["VLLMGenerationHelper", "_VLLMGenerationState"]
