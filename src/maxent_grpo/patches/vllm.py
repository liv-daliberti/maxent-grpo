"""
Robust helpers for a vLLM ``/generate`` server.

This module provides a resilient ``safe_generate`` that handles transient
errors with retries/backoff and decodes multiple response schemas across vLLM
versions (OpenAI‑compatible ``choices``, ``results``, batched ``text``, and
legacy ``completion_ids`` when a tokenizer is provided). It also supports
streaming responses by collecting chunked JSON lines.

Key functions

* ``safe_request``: Simple GET with retries/backoff.
* ``safe_generate``: POST to ``/generate`` with schema‑agnostic decoding and
  optional streaming support.

License
Copyright 2025 Liv d'Aliberti

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the specific language governing permissions and
limitations under the License.
"""

# Helpers for talking to an external ``trl vllm-serve`` instance.
#
# Notes
# -----
# - vLLM ≤ 0.8.5 sometimes returns token IDs by default, for example:
#
#     {"completion_ids": [[ids...]], "prompt_ids": [[ids...]]}
#
#   This module detects that schema and decodes it if a tokenizer is provided.
from __future__ import annotations

import json
import time
from dataclasses import dataclass
import hashlib
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

try:
    import requests
except ImportError:  # pragma: no cover - optional dependency

    class _RequestsStub:
        """Minimal stub to avoid hard dependency when vLLM is unused."""

        class ConnectionError(RuntimeError):
            pass

        class Timeout(RuntimeError):
            pass

        def _raise(self, *_args, **_kwargs):
            raise ImportError("requests is required for vLLM helpers")

        get = _raise
        post = _raise

    requests = _RequestsStub()

# Type aliases for JSON responses
JsonDict = Dict[str, Any]
GenerationResults = List[List[str]]
GenerationLogprobGroups = Optional[List[List[Optional["VLLMLogprobResult"]]]]


@dataclass
class VLLMLogprobResult:
    """Aggregate (and optionally raw) log-probability info for one completion."""

    logprob_sum: float
    token_count: int
    token_logprobs: Optional[List[float]] = None
    raw_output: Optional[Dict[str, Any]] = None

    def to_trl_payload(self) -> Dict[str, Any]:
        """Return a dict compatible with TRL's refinement metadata.

        :returns: Dictionary describing logprob sums/tokens/raw output.
        :rtype: dict[str, Any]
        """
        payload: Dict[str, Any] = {
            "logprob_sum": float(self.logprob_sum),
            "token_count": int(self.token_count),
        }
        if self.token_logprobs is not None:
            payload["token_logprobs"] = [float(val) for val in self.token_logprobs]
        if self.raw_output is not None:
            payload["raw_output"] = self.raw_output
        return payload


@runtime_checkable
class TokenizerLike(Protocol):
    """Protocol for objects that can decode token IDs to text."""

    def decode(self, token_ids: List[int], **kwargs: Any) -> str:
        """Return the decoded string for ``token_ids``."""
        raise NotImplementedError


# (no generic TypeVar needed here) tokenizer is typed as Optional[TokenizerLike]


# ─────────────────── generic GET helper ──────────────────────────────────────
def safe_request(
    url: str, max_retries: int = 3, backoff: float = 1.0, timeout: float = 10.0
) -> JsonDict:
    """GET JSON with basic retry/backoff.

    :param url: Endpoint to query.
    :type url: str
    :param max_retries: Number of attempts before surfacing the error.
    :type max_retries: int
    :param backoff: Base backoff in seconds; exponential across attempts.
    :type backoff: float
    :param timeout: Per‑request timeout in seconds.
    :type timeout: float
    :returns: Parsed JSON payload.
    :rtype: dict
    :raises RuntimeError: If the request ultimately fails (HTTP error or repeated connection/timeouts).
    """
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
        except (requests.ConnectionError, requests.Timeout):
            if attempt < max_retries - 1:
                time.sleep(backoff * (2**attempt))
            else:
                raise


# ─────────────────── helper to parse non-stream JSON ─────────────────────────
def _clean_logprob_seq(candidate: Any) -> Optional[List[float]]:
    """Normalize various logprob containers into a float list.

    :param candidate: Raw logprob payload (list/dict from vLLM).
    :type candidate: Any
    :returns: Flat list of floats or ``None`` when no values exist.
    :rtype: list[float] | None
    """
    if candidate is None:
        return None
    if isinstance(candidate, dict):
        if "token_logprobs" in candidate:
            return _clean_logprob_seq(candidate.get("token_logprobs"))
        if "logprobs" in candidate:
            return _clean_logprob_seq(candidate.get("logprobs"))
    if isinstance(candidate, list):
        cleaned = [float(val) for val in candidate if val is not None]
        return cleaned if cleaned else None
    return None


def _infer_token_count(entry: Dict[str, Any], seq: Optional[List[float]]) -> int:
    """Best-effort token-count heuristic for varied response schemas.

    :param entry: Raw completion dictionary emitted by vLLM.
    :type entry: dict[str, Any]
    :param seq: Parsed token logprob sequence if already available.
    :type seq: list[float] | None
    :returns: Estimated number of tokens contributing to the result.
    :rtype: int
    """
    tokens = entry.get("token_ids") or entry.get("output_token_ids")
    if isinstance(tokens, list) and tokens:
        return len(tokens)
    if seq is not None:
        return len(seq)
    extra = entry.get("output_token_logprobs")
    cleaned = _clean_logprob_seq(extra)
    if cleaned:
        return len(cleaned)
    if isinstance(entry.get("text"), str):
        return max(1, len(entry["text"].split()))
    return 1


def _extract_logprob_info(entry: Dict[str, Any]) -> Optional[VLLMLogprobResult]:
    """Convert schema-specific logprob payloads into ``VLLMLogprobResult``.

    :param entry: Raw completion block returned by the server.
    :type entry: dict[str, Any]
    :returns: Structured summary or ``None`` when metadata is missing.
    :rtype: VLLMLogprobResult | None
    """
    if not isinstance(entry, dict):
        return None
    if entry.get("cumulative_logprob") is not None:
        seq = _clean_logprob_seq(entry.get("output_token_logprobs"))
        count = _infer_token_count(entry, seq)
        return VLLMLogprobResult(
            logprob_sum=float(entry["cumulative_logprob"]),
            token_count=max(1, int(count)),
            token_logprobs=seq,
            raw_output=dict(entry),
        )
    seq = _clean_logprob_seq(entry.get("logprobs"))
    if seq is None and isinstance(entry.get("logprobs"), dict):
        seq = _clean_logprob_seq(entry["logprobs"].get("token_logprobs"))
    if seq is None:
        seq = _clean_logprob_seq(entry.get("output_token_logprobs"))
    if seq is None:
        return None
    return VLLMLogprobResult(
        logprob_sum=float(sum(seq)),
        token_count=max(1, len(seq)),
        token_logprobs=seq,
        raw_output=dict(entry),
    )


def _parse_nonstream_json(
    data: JsonDict,
    tokenizer: Optional[TokenizerLike] = None,
    *,
    want_logprobs: bool = False,
) -> Tuple[GenerationResults, GenerationLogprobGroups]:
    """Normalize vLLM JSON response into grouped texts (+ optional logprobs).

    :param data: JSON payload returned by vLLM's REST API.
    :type data: dict[str, Any]
    :param tokenizer: Optional tokenizer used to decode completion IDs.
    :type tokenizer: TokenizerLike | None
    :param want_logprobs: Whether to capture logprob metadata per output.
    :type want_logprobs: bool
    :returns: Tuple of grouped completion texts and optional logprob metadata.
    :rtype: tuple[list[list[str]], list[list[Optional[VLLMLogprobResult]]] | None]
    """
    grouped: GenerationResults = []
    logprob_groups: GenerationLogprobGroups = [] if want_logprobs else None

    def _append_group(
        texts: List[str], logs: Optional[List[Optional[VLLMLogprobResult]]]
    ) -> None:
        """Append one prompt's completions into the grouped outputs.

        :param texts: List of decoded completions for a single prompt.
        :type texts: list[str]
        :param logs: Optional log-probability metadata aligned with ``texts``.
        :type logs: list[list[VLLMLogprobResult]] | None
        """
        grouped.append(texts)
        if logprob_groups is not None:
            logprob_groups.append(logs or [])

    # OpenAI route
    if "choices" in data:
        texts = [str(c.get("text", "")) for c in data["choices"]]
        logs = None
        if logprob_groups is not None:
            logs = [_extract_logprob_info(c) for c in data["choices"]]
        _append_group(texts, logs)
        return grouped, logprob_groups
    # Plain /generate route (newer default)
    if "results" in data:
        for item in data["results"]:
            prompt_texts: List[str] = []
            prompt_logs: Optional[List[Optional[VLLMLogprobResult]]] = (
                [] if logprob_groups is not None else None
            )
            if isinstance(item, dict):
                outputs = item.get("outputs")
                if isinstance(outputs, list) and outputs:
                    for out in outputs:
                        if "text" in out:
                            prompt_texts.append(str(out.get("text", "")))
                            if prompt_logs is not None:
                                prompt_logs.append(_extract_logprob_info(out))
                    if prompt_texts:
                        _append_group(prompt_texts, prompt_logs)
                        continue
                if "text" in item:
                    prompt_texts.append(str(item["text"]))
                    _append_group(prompt_texts, prompt_logs)
                    continue
                if "completion_ids" in item and tokenizer is not None:
                    decoded = [
                        tokenizer.decode(ids, skip_special_tokens=True)
                        for ids in item["completion_ids"]
                    ]
                    _append_group(decoded, prompt_logs)
                    continue
            _append_group([str(item)], prompt_logs)
        return grouped, logprob_groups
    # vLLM 0.8.x batched output
    if "text" in data and isinstance(data["text"], list):
        for text in data["text"]:
            _append_group([str(text)], None)
        return grouped, logprob_groups
    # vLLM 0.8.x token-ID output
    if "completion_ids" in data:
        if tokenizer is None:
            raise RuntimeError(
                "Server returned token IDs but no tokenizer was supplied to safe_generate()."
            )
        decoded = [
            tokenizer.decode(ids, skip_special_tokens=True)
            for ids in data["completion_ids"]
        ]
        for text in decoded:
            _append_group([text], None)
        return grouped, logprob_groups
    raise RuntimeError(f"Unknown vLLM response format: {data}")


# ─────────────────── POST /generate helper ────────────────────────────────────
# ─────────────────── logging setup ───────────────────────────────────────────
LOG = logging.getLogger(__name__)


def safe_generate(
    *,
    prompts: List[str],
    url: str = "http://localhost:8000/generate",
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: Optional[int] = None,
    n: int = 1,
    stream: bool = False,
    tokenizer: Optional[TokenizerLike] = None,  # optional tokenizer for decoding ids
    best_of: Optional[int] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    stop: Optional[List[str]] = None,
    logit_bias: Optional[Dict[str, float]] = None,
    guided_json: Optional[str] = None,
    guided_regex: Optional[str] = None,
    request_id: Optional[str] = None,
    request_id_prefix: Optional[str] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    backoff: Optional[float] = None,
    return_logprobs: bool = False,
) -> Tuple[GenerationResults, GenerationLogprobGroups, float]:
    """Robust POST to ``/generate`` with retry + schema-agnostic decoding.

    :param prompts: Input prompts (batch) to generate from.
    :type prompts: list[str]
    :param url: Base URL to the ``/generate`` route.
    :type url: str
    :param max_tokens: Maximum tokens to generate per completion.
    :type max_tokens: int
    :param temperature: Sampling temperature.
    :type temperature: float
    :param top_p: Nucleus sampling p.
    :type top_p: float
    :param top_k: Optional top-k cutoff applied during sampling.
    :type top_k: int | None
    :param n: Number of completions per prompt.
    :type n: int
    :param stream: Whether to use chunked streaming responses.
    :type stream: bool
    :param tokenizer: Optional tokenizer to decode token ID arrays.
    :type tokenizer: Any
    :param best_of: vLLM ``best_of`` parameter to sample more than ``n`` candidates.
    :type best_of: int | None
    :param frequency_penalty: Frequency penalty forwarded to vLLM sampling.
    :type frequency_penalty: float | None
    :param presence_penalty: Presence penalty forwarded to vLLM sampling.
    :type presence_penalty: float | None
    :param stop: Stop sequences used to truncate completions.
    :type stop: list[str] | None
    :param logit_bias: Token-level logit bias forwarded to vLLM.
    :type logit_bias: dict[str, float] | None
    :param guided_json: Optional JSON schema string for constrained decoding.
    :type guided_json: str | None
    :param guided_regex: Optional regex constraint for decoding.
    :type guided_regex: str | None
    :param request_id: Explicit request identifier to forward to vLLM.
    :type request_id: str | None
    :param request_id_prefix: Prefix used when auto-generating ``request_id``.
    :type request_id_prefix: str | None
    :param max_retries: Number of attempts before surfacing the error.
    :type max_retries: int
    :param backoff: Base backoff in seconds; exponential across attempts.
    :type backoff: float
    :param timeout: Per‑request timeout in seconds.
    :type timeout: float
    :param return_logprobs: Whether to request log-prob metadata from vLLM.
    :type return_logprobs: bool
    :returns: Tuple of grouped texts, optional log-prob metadata, and latency in milliseconds.
    :rtype: tuple[list[list[str]], Optional[list[list[VLLMLogprobResult]]], float]
    :raises RuntimeError: When the server returns a non‑200 response or after
        exhausting retries.
    """
    timeout = float(
        timeout if timeout is not None else os.environ.get("VLLM_TIMEOUT", 120.0)
    )
    max_retries = int(
        max_retries
        if max_retries is not None
        else os.environ.get("VLLM_MAX_RETRIES", 3)
    )
    backoff = float(
        backoff if backoff is not None else os.environ.get("VLLM_BACKOFF", 1.0)
    )
    effective_request_id = (
        request_id
        or (f"{request_id_prefix}-" if request_id_prefix else "") + uuid.uuid4().hex
    )
    payload: Dict[str, Any] = {
        "prompts": prompts,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        "max_tokens": max_tokens,
        "stream": stream,
        "request_id": effective_request_id,
    }
    payload["sampling_params"] = {
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        "max_tokens": max_tokens,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
    }
    if top_k is not None:
        payload["top_k"] = top_k
        payload["sampling_params"]["top_k"] = top_k
    if return_logprobs:
        payload["return_logprobs"] = True
        payload["logprobs"] = payload["sampling_params"]["logprobs"] = 1
    if best_of is not None:
        payload["best_of"] = best_of
    if frequency_penalty is not None:
        payload["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        payload["presence_penalty"] = presence_penalty
    if stop:
        payload["stop"] = stop
    if logit_bias:
        payload["logit_bias"] = logit_bias
    if guided_json:
        payload["guided_json"] = guided_json
    if guided_regex:
        payload["guided_regex"] = guided_regex

    headers = _build_vllm_headers(prompts)
    LOG.debug(
        "vLLM request start | url=%s | prompts=%d | n=%d | stream=%s",
        url,
        len(prompts),
        n,
        stream,
    )
    start_time = time.perf_counter()
    for attempt in range(max_retries):
        try:
            r = requests.post(
                url,
                json=payload,
                timeout=timeout,
                stream=stream,
                headers=headers,
            )
            if r.status_code == 200:
                LOG.debug(
                    "vLLM request success | url=%s | prompts=%d | attempt=%d",
                    url,
                    len(prompts),
                    attempt + 1,
                )
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                if stream:
                    return _collect_stream_texts(r, len(prompts)), None, latency_ms
                grouped, meta = _parse_nonstream_json(
                    r.json(),
                    tokenizer,
                    want_logprobs=return_logprobs,
                )
                return grouped, meta, latency_ms
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
        except (requests.ConnectionError, requests.Timeout, RuntimeError) as err:
            LOG.warning(
                "vLLM request failure (%s) | attempt=%d/%d | url=%s",
                err,
                attempt + 1,
                max_retries,
                url,
            )
            if attempt < max_retries - 1:
                time.sleep(backoff * (2**attempt))
            else:
                LOG.error(
                    "vLLM request exhausted retries | url=%s | prompts=%d",
                    url,
                    len(prompts),
                )
                raise RuntimeError(f"safe_generate failed: {err}") from err


def _collect_stream_texts(
    response: requests.Response, num_prompts: int
) -> List[List[str]]:
    """Collect and join streaming response chunks per prompt index.

    :param response: Requests response object streaming chunked JSON lines.
    :type response: ``requests.Response``
    :param num_prompts: Number of prompts in the input batch.
    :type num_prompts: int
    :returns: One concatenated text per prompt.
    :rtype: list[list[str]]
    """
    texts: List[List[str]] = [[] for _ in range(num_prompts)]
    for line in response.iter_lines():
        if not line:
            continue
        row = json.loads(line.decode())
        idx = int(row.get("prompt_index", 0))
        if 0 <= idx < num_prompts:
            texts[idx].append(row.get("text", ""))
        # else: ignore malformed index
    return [["".join(parts)] for parts in texts]


def _build_vllm_headers(prompts: List[str]) -> Dict[str, str]:
    """Construct optional headers used by TRL's vLLM RPC helpers.

    :param prompts: Prompt batch used to derive deterministic group IDs.
    :type prompts: list[str]
    :returns: Header dictionary containing stable request identifiers and
        optional auth/extra headers sourced from the environment.
    :rtype: dict[str, str]
    """
    headers: Dict[str, str] = {}
    group_id = os.environ.get("VLLM_GROUP_REQUEST_ID")
    if not group_id:
        # Stable slug derived from the prompt batch for tracking.
        joined = "|".join(prompts)
        group_id = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    headers["X-VLLM-Group-Request-ID"] = group_id
    api_key = os.environ.get("VLLM_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    custom_headers = os.environ.get("VLLM_EXTRA_HEADERS")
    if custom_headers:
        try:
            parsed = json.loads(custom_headers)
            if isinstance(parsed, dict):
                headers.update({str(k): str(v) for k, v in parsed.items()})
        except json.JSONDecodeError:
            pass
    return headers
