"""Configuration dataclasses for the training runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


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


__all__ = ["GenerationSamplingConfig", "MaxEntOptions", "VLLMClientConfig"]
