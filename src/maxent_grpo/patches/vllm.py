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
from typing import (
    Any,
    Dict,
    List,
    NoReturn,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
    TYPE_CHECKING,
)

from maxent_grpo.generation.errors import GenerationServiceError, ServiceErrorPayload

try:
    import requests
except ImportError:  # pragma: no cover - optional dependency

    class _RequestsStub:
        """Minimal stub to avoid hard dependency when vLLM is unused."""

        class ConnectionError(RuntimeError):
            pass

        class Timeout(RuntimeError):
            pass

        def _raise(self, *_args: Any, **_kwargs: Any) -> NoReturn:
            raise ImportError("requests is required for vLLM helpers")

        get = _raise
        post = _raise

    requests = _RequestsStub()

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    from requests import Response as RequestsResponse
else:  # pragma: no cover - runtime fallback
    RequestsResponse = Any

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
    raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts")


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
        cleaned: List[float] = []
        for val in candidate:
            if val is None:
                continue
            if isinstance(val, (int, float)):
                cleaned.append(float(val))
                continue
            if isinstance(val, dict):
                extracted = None
                for key in (
                    "logprob",
                    "log_prob",
                    "token_logprob",
                    "token_log_prob",
                    "prob",
                ):
                    if key not in val:
                        continue
                    try:
                        extracted = float(val[key])
                        break
                    except (TypeError, ValueError):
                        extracted = None
                if extracted is not None:
                    cleaned.append(extracted)
                continue
            if isinstance(val, (list, tuple)) and val:
                try:
                    cleaned.append(float(val[0]))
                except (TypeError, ValueError):
                    continue
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
    metadata = entry.get("metadata")
    meta_dict: Optional[Dict[str, Any]] = metadata if isinstance(metadata, dict) else None
    meta_seq = None
    logprob_sum = None
    token_count = None
    if meta_dict is not None:
        meta_seq = _clean_logprob_seq(
            meta_dict.get("token_logprobs") or meta_dict.get("logprobs")
        )
        logprob_sum = (
            meta_dict.get("logprob_sum")
            if meta_dict.get("logprob_sum") is not None
            else meta_dict.get("cumulative_logprob")
        )
        if logprob_sum is None:
            logprob_sum = meta_dict.get("logprob")
        token_count = meta_dict.get("token_count") or meta_dict.get("num_tokens")
        if token_count is None and meta_seq:
            token_count = len(meta_seq)
        if logprob_sum is not None:
            inferred = token_count if token_count is not None else _infer_token_count(entry, meta_seq)
            return VLLMLogprobResult(
                logprob_sum=float(logprob_sum),
                token_count=max(1, int(inferred)),
                token_logprobs=meta_seq,
                raw_output=meta_dict.get("raw_output") or dict(entry),
            )
    if entry.get("cumulative_logprob") is not None:
        seq = _clean_logprob_seq(entry.get("output_token_logprobs"))
        count = _infer_token_count(entry, seq)
        return VLLMLogprobResult(
            logprob_sum=float(entry["cumulative_logprob"]),
            token_count=max(1, int(count)),
            token_logprobs=seq,
            raw_output=dict(entry),
        )
    if entry.get("logprob_sum") is not None:
        seq = _clean_logprob_seq(entry.get("token_logprobs")) or _clean_logprob_seq(
            entry.get("output_token_logprobs")
        )
        count = _infer_token_count(entry, seq)
        return VLLMLogprobResult(
            logprob_sum=float(entry["logprob_sum"]),
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
        logs: Optional[List[Optional[VLLMLogprobResult]]] = None
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
                    completion_ids = item["completion_ids"]
                    decoded = [
                        tokenizer.decode(ids, skip_special_tokens=True)
                        for ids in completion_ids
                    ]
                    if prompt_logs is not None:
                        prompt_logs = [
                            VLLMLogprobResult(
                                logprob_sum=0.0,
                                token_count=len(ids),
                                token_logprobs=None,
                                raw_output={"token_ids": list(ids)},
                            )
                            for ids in completion_ids
                        ]
                    _append_group(decoded, prompt_logs)
                    continue
            _append_group([str(item)], prompt_logs)
        return grouped, logprob_groups
    # vLLM 0.8.x batched output
    if "text" in data and isinstance(data["text"], list):
        text_entries = data["text"]
        # Some vLLM servers (notably older /generate variants) return a flat list of
        # texts, optionally with parallel logprob arrays. Try to reconstruct
        # per-output logprob summaries when requested.
        flat_logprobs = data.get("logprobs")
        if flat_logprobs is None:
            flat_logprobs = data.get("output_token_logprobs")
        flat_cum_logprob = data.get("cumulative_logprob")
        if flat_cum_logprob is None:
            flat_cum_logprob = data.get("cumulative_logprobs")
        flat_token_ids = data.get("token_ids")
        if flat_token_ids is None:
            flat_token_ids = data.get("output_token_ids")
        for idx, text_entry in enumerate(text_entries):
            if isinstance(text_entry, dict):
                payload_entry: Dict[str, Any] = dict(text_entry)
                text = str(payload_entry.get("text", ""))
            else:
                text = str(text_entry)
                payload_entry = {"text": text}
            logs: Optional[List[Optional[VLLMLogprobResult]]] = None
            if logprob_groups is not None:
                if isinstance(flat_logprobs, list) and idx < len(flat_logprobs):
                    payload_entry.setdefault("logprobs", flat_logprobs[idx])
                if isinstance(flat_cum_logprob, list) and idx < len(flat_cum_logprob):
                    payload_entry.setdefault("cumulative_logprob", flat_cum_logprob[idx])
                if isinstance(flat_token_ids, list) and idx < len(flat_token_ids):
                    payload_entry.setdefault("token_ids", flat_token_ids[idx])
                logs = [_extract_logprob_info(payload_entry)]
            _append_group([text], logs)
        return grouped, logprob_groups
    # vLLM 0.8.x token-ID output
    if "completion_ids" in data:
        if tokenizer is None:
            raise RuntimeError(
                "Server returned token IDs but no tokenizer was supplied to safe_generate()."
            )
        completion_ids = data["completion_ids"]
        decoded = [
            tokenizer.decode(ids, skip_special_tokens=True) for ids in completion_ids
        ]
        for ids, text in zip(completion_ids, decoded):
            logs: Optional[List[Optional[VLLMLogprobResult]]] = None
            if logprob_groups is not None:
                logs = [
                    VLLMLogprobResult(
                        logprob_sum=0.0,
                        token_count=len(ids),
                        token_logprobs=None,
                        raw_output={"token_ids": list(ids)},
                    )
                ]
            _append_group([text], logs)
        return grouped, logprob_groups
    raise RuntimeError(f"Unknown vLLM response format: {data}")


# ─────────────────── POST /generate helper ────────────────────────────────────
# ─────────────────── logging setup ───────────────────────────────────────────
LOG = logging.getLogger(__name__)


def _find_client_tag(candidate: Any, depth: int = 0) -> Optional[str]:
    """Traverse a limited portion of ``candidate`` to locate ``client_tag``."""

    if depth > 5:
        return None
    if isinstance(candidate, dict):
        tag = candidate.get("client_tag")
        if isinstance(tag, str) and tag.strip():
            return tag.strip()
        for key in ("metadata", "meta", "context", "extra", "request"):
            nested = candidate.get(key)
            tag = _find_client_tag(nested, depth + 1)
            if tag:
                return tag
    elif isinstance(candidate, list):
        for entry in candidate[:8]:
            tag = _find_client_tag(entry, depth + 1)
            if tag:
                return tag
    return None


def _filter_result_outputs_for_tag(
    entry: Any, client_tag: str
) -> bool:
    """Filter per-output metadata for a single prompt entry."""

    if not isinstance(entry, dict):
        return True
    outputs = entry.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        return True
    filtered: List[Any] = []
    dropped = 0
    LOG.debug(
        "Inspecting %d outputs for client_tag=%s",
        len(outputs),
        client_tag,
    )
    for output in outputs:
        tag = _find_client_tag(output)
        if tag is None or tag == client_tag:
            filtered.append(output)
        else:
            dropped += 1
            LOG.debug(
                "Dropping vLLM output tagged for %s (expecting %s)",
                tag,
                client_tag,
            )
    if dropped and not filtered:
        return False
    if dropped:
        entry["outputs"] = filtered
        LOG.debug(
            "Filtered %d extraneous vLLM outputs for client_tag=%s",
            dropped,
            client_tag,
        )
    return True


def _filter_response_for_client_tag(data: JsonDict, client_tag: Optional[str]) -> JsonDict:
    """Remove prompt groups that do not match ``client_tag`` when provided."""

    if not client_tag or not isinstance(data, dict):
        return data
    results = data.get("results")
    if isinstance(results, list) and results:
        filtered: List[Any] = []
        dropped = 0
        LOG.debug(
            "Evaluating %d vLLM result groups for client_tag=%s",
            len(results),
            client_tag,
        )
        for entry in results:
            tag = _find_client_tag(entry)
            if tag is not None and tag != client_tag:
                dropped += 1
                LOG.debug(
                    "Dropping result group tagged for %s (expecting %s)",
                    tag,
                    client_tag,
                )
                continue
            if not _filter_result_outputs_for_tag(entry, client_tag):
                dropped += 1
                continue
            filtered.append(entry)
        if filtered:
            if dropped:
                LOG.debug(
                    "Filtered %d extraneous vLLM result groups for client_tag=%s",
                    dropped,
                    client_tag,
                )
            data = dict(data)
            data["results"] = filtered
            return data
        if dropped:
            raise RuntimeError(
                f"vLLM response missing completions for client_tag={client_tag}"
            )
        return data
    root_tag = _find_client_tag(data)
    if root_tag and root_tag != client_tag:
        raise RuntimeError(
            f"vLLM response tagged for {root_tag} cannot satisfy client_tag={client_tag}"
        )
    return data


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
    backoff_multiplier: Optional[float] = None,
    return_logprobs: bool = False,
    service_model: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    client_tag: Optional[str] = None,
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
    :param service_model: Optional identifier for the served model (used in error payloads).
    :type service_model: str | None
    :param metadata: Optional structured context (dataset/model) copied into error payloads.
    :type metadata: dict[str, Any] | None
    :param client_tag: Optional client/rank identifier forwarded via headers/payload.
    :type client_tag: str | None
    :returns: Tuple of grouped texts, optional log-prob metadata, and latency in milliseconds.
    :rtype: tuple[list[list[str]], Optional[list[list[VLLMLogprobResult]]], float]
    :raises GenerationServiceError: When the server responds with repeated errors
        after exhausting retries.
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
    backoff_multiplier = float(
        backoff_multiplier
        if backoff_multiplier is not None
        else os.environ.get("VLLM_BACKOFF_MULTIPLIER", 2.0)
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
    if client_tag:
        payload["client_tag"] = client_tag

    headers = _build_vllm_headers(prompts, client_tag)
    prompt_chars = sum(len(prompt) for prompt in prompts)
    prompt_lens_sample = [len(prompt) for prompt in prompts[:8]]
    prompt_hash_sample = [
        hashlib.sha256(prompt.encode("utf-8", errors="ignore")).hexdigest()[:8]
        for prompt in prompts[:8]
    ]
    redacted_headers = {
        key: ("<redacted>" if key.lower() == "authorization" else value)
        for key, value in headers.items()
    }
    try:
        payload_size_bytes = len(json.dumps(payload).encode("utf-8"))
    except (TypeError, ValueError):
        payload_size_bytes = None
    LOG.debug(
        "vLLM request start | url=%s | prompts=%d | n=%d | stream=%s",
        url,
        len(prompts),
        n,
        stream,
    )
    LOG.debug(
        (
            "vLLM request prepared | req_id=%s | prompts=%d | prompt_chars=%d "
            "| prompt_lens_sample=%s | prompt_hash_sample=%s | n=%d | max_tokens=%d "
            "| stream=%s | timeout=%.1fs | max_retries=%d | backoff=%.2f "
            "| backoff_multiplier=%.2f | payload_bytes=%s | client_tag=%s | headers=%s"
        ),
        effective_request_id,
        len(prompts),
        prompt_chars,
        prompt_lens_sample,
        prompt_hash_sample,
        n,
        max_tokens,
        stream,
        timeout,
        max_retries,
        backoff,
        backoff_multiplier,
        payload_size_bytes,
        client_tag,
        redacted_headers,
    )
    start_time = time.perf_counter()
    last_status: Optional[int] = None
    for attempt in range(max_retries):
        attempt_start = time.perf_counter()
        LOG.debug(
            "vLLM attempt start | req_id=%s | attempt=%d/%d | url=%s | stream=%s | timeout=%.2fs | payload_bytes=%s",
            effective_request_id,
            attempt + 1,
            max_retries,
            url,
            stream,
            timeout,
            payload_size_bytes,
        )
        try:
            r = requests.post(
                url,
                json=payload,
                timeout=timeout,
                stream=stream,
                headers=headers,
            )
            attempt_elapsed_ms = (time.perf_counter() - attempt_start) * 1000.0
            LOG.debug(
                (
                    "vLLM response received | req_id=%s | attempt=%d | status=%s "
                    "| attempt_elapsed_ms=%.2f | reported_elapsed_ms=%s | content_length=%s "
                    "| transfer_encoding=%s"
                ),
                effective_request_id,
                attempt + 1,
                getattr(r, "status_code", None),
                attempt_elapsed_ms,
                (
                    r.elapsed.total_seconds() * 1000.0
                    if getattr(r, "elapsed", None) is not None
                    else None
                ),
                r.headers.get("content-length") if hasattr(r, "headers") else None,
                r.headers.get("transfer-encoding") if hasattr(r, "headers") else None,
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
                body_bytes = len(r.content) if hasattr(r, "content") else None
                LOG.debug(
                    "vLLM response decode start | req_id=%s | attempt=%d | body_bytes=%s | content_type=%s",
                    effective_request_id,
                    attempt + 1,
                    body_bytes,
                    r.headers.get("content-type") if hasattr(r, "headers") else None,
                )
                decode_start = time.perf_counter()
                payload = r.json()
                decode_ms = (time.perf_counter() - decode_start) * 1000.0
                LOG.debug(
                    "vLLM response JSON decoded | req_id=%s | attempt=%d | decode_ms=%.2f | payload_keys=%s",
                    effective_request_id,
                    attempt + 1,
                    decode_ms,
                    list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__,
                )
                payload = _filter_response_for_client_tag(payload, client_tag)
                grouped, meta = _parse_nonstream_json(
                    payload,
                    tokenizer,
                    want_logprobs=return_logprobs,
                )
                if return_logprobs:
                    if meta is None:
                        LOG.warning(
                            "vLLM logprobs requested but response missing logprob metadata | req_id=%s | attempt=%d",
                            effective_request_id,
                            attempt + 1,
                        )
                    else:
                        total_completions = sum(len(g or []) for g in meta)
                        with_token_logprobs = sum(
                            1
                            for g in meta
                            for entry in (g or [])
                            if entry is not None and entry.token_logprobs
                        )
                        sample = None
                        for g in meta:
                            if g:
                                for entry in g:
                                    if entry is not None:
                                        sample = (
                                            float(entry.logprob_sum),
                                            int(entry.token_count),
                                        )
                                        break
                                if sample:
                                    break
                        LOG.debug(
                            (
                                "vLLM logprob stats | req_id=%s | attempt=%d | groups=%d | completions=%d "
                                "| with_token_logprobs=%d | sample_sum_count=%s"
                            ),
                            effective_request_id,
                            attempt + 1,
                            len(meta),
                            total_completions,
                            with_token_logprobs,
                            sample,
                        )
                LOG.debug(
                    (
                        "vLLM response parsed | req_id=%s | attempt=%d | grouped_prompts=%d "
                        "| per_prompt_lengths=%s | meta_present=%s | latency_ms=%.2f"
                    ),
                    effective_request_id,
                    attempt + 1,
                    len(grouped) if grouped is not None else 0,
                    [len(entry) for entry in grouped[:8]] if grouped else [],
                    meta is not None,
                    latency_ms,
                )
                return grouped, meta, latency_ms
            last_status = r.status_code
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
        except (requests.ConnectionError, requests.Timeout, RuntimeError) as err:
            LOG.warning(
                (
                    "vLLM request failure (%s) | attempt=%d/%d | url=%s | req_id=%s "
                    "| prompts=%d | payload_bytes=%s | timeout=%.2fs"
                ),
                err,
                attempt + 1,
                max_retries,
                url,
                effective_request_id,
                len(prompts),
                payload_size_bytes,
                timeout,
            )
            if attempt < max_retries - 1:
                delay = backoff * (backoff_multiplier ** attempt)
                time.sleep(delay)
            else:
                LOG.error(
                    "vLLM request exhausted retries | url=%s | prompts=%d",
                    url,
                    len(prompts),
                )
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                payload_extra = {
                    "stream": stream,
                    "timeout": timeout,
                    "request_id_prefix": request_id_prefix,
                    "max_tokens": max_tokens,
                    "completions_per_prompt": n,
                    "backoff_initial": backoff,
                    "backoff_multiplier": backoff_multiplier,
                    "elapsed_ms": elapsed_ms,
                }
                if metadata:
                    for key, value in metadata.items():
                        if value is not None:
                            payload_extra[key] = value
                error_payload = ServiceErrorPayload(
                    service="vllm",
                    endpoint=url,
                    model=service_model,
                    prompt_count=len(prompts),
                    payload_chars=prompt_chars,
                    payload_size_bytes=payload_size_bytes,
                    status_code=last_status,
                    attempt=attempt + 1,
                    max_attempts=max_retries,
                    exception_type=type(err).__name__,
                    exception_message=str(err),
                    request_id=effective_request_id,
                    extra=payload_extra,
                )
                raise GenerationServiceError(
                    f"safe_generate failed after {max_retries} attempts",
                    error_payload,
                ) from err
    raise RuntimeError("vLLM request exhausted without response")


def _collect_stream_texts(
    response: RequestsResponse, num_prompts: int
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
    chunk_counts: List[int] = [0 for _ in range(num_prompts)]
    total_bytes = 0
    for line in response.iter_lines():
        if not line:
            continue
        total_bytes += len(line)
        row = json.loads(line.decode())
        idx = int(row.get("prompt_index", 0))
        if 0 <= idx < num_prompts:
            texts[idx].append(row.get("text", ""))
            chunk_counts[idx] += 1
        # else: ignore malformed index
    LOG.debug(
        "vLLM stream collected | prompts=%d | chunks=%d | bytes=%d | per_prompt_chunks=%s",
        num_prompts,
        sum(chunk_counts),
        total_bytes,
        chunk_counts[:8],
    )
    return [["".join(parts)] for parts in texts]


def _build_vllm_headers(
    prompts: List[str], client_tag: Optional[str] = None
) -> Dict[str, str]:
    """Construct optional headers used by TRL's vLLM RPC helpers.

    :param prompts: Prompt batch used to derive deterministic group IDs.
    :type prompts: list[str]
    :param client_tag: Optional client/rank identifier propagated to the server.
    :type client_tag: str | None
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
    if client_tag:
        headers["X-VLLM-Client-Tag"] = client_tag
    custom_headers = os.environ.get("VLLM_EXTRA_HEADERS")
    if custom_headers:
        try:
            parsed = json.loads(custom_headers)
            if isinstance(parsed, dict):
                headers.update({str(k): str(v) for k, v in parsed.items()})
        except json.JSONDecodeError:
            LOG.warning("Invalid VLLM_EXTRA_HEADERS JSON; ignoring custom headers.")
    return headers
