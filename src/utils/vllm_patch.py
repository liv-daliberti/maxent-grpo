"""
Helpers for talking to an external ``trl vllm-serve`` instance.

Copyright 2025 Liv d'Aliberti

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at::

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Notes
-----
- vLLM ≤ 0.8.5 sometimes returns token IDs by default, for example::

    {"completion_ids": [[ids...]], "prompt_ids": [[ids...]]}

  This module detects that schema and decodes it if a tokenizer is provided.
"""
from __future__ import annotations
import json
import time
from typing import List
import requests


# ─────────────────── generic GET helper ──────────────────────────────────────
def safe_request(url: str, max_retries: int = 3, backoff: float = 1.0, timeout: float = 10.0):
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
    :raises RuntimeError: If a non‑200 response is received.
    :raises requests.ConnectionError: On connection errors after retries.
    :raises requests.Timeout: On timeouts after retries.
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
def _parse_nonstream_json(data: dict, tokenizer=None) -> List[List[str]]:
    """Normalize non‑streaming vLLM JSON response into grouped texts.

    Supports OpenAI-compatible schema (``choices``), newer vLLM ``results``,
    batched ``text`` lists, and legacy ``completion_ids`` which require a
    tokenizer to decode.

    :param data: Raw JSON response.
    :type data: dict
    :param tokenizer: Optional tokenizer to decode token ID arrays.
    :type tokenizer: Any
    :returns: List of per‑prompt lists of texts (shape: ``[B][N]``).
    :rtype: list[list[str]]
    :raises RuntimeError: If an unknown schema is encountered or decoding is
        required but no tokenizer is supplied.
    """
    # OpenAI route
    if "choices" in data:
        return [[c["text"] for c in data["choices"]]]
    # Plain /generate route (newer default)
    if "results" in data:
        return [[r["text"] for r in data["results"]]]
    # vLLM 0.8.x batched output
    if "text" in data and isinstance(data["text"], list):
        return [[t] for t in data["text"]]
    # vLLM 0.8.x token-ID output
    if "completion_ids" in data:
        if tokenizer is None:
            raise RuntimeError(
                "Server returned token IDs but no tokenizer was supplied to safe_generate()."
            )
        return [[tokenizer.decode(ids, skip_special_tokens=True)] for ids in data["completion_ids"]]
    raise RuntimeError(f"Unknown vLLM response format: {data}")


# ─────────────────── POST /generate helper ────────────────────────────────────
def safe_generate(
    *,
    prompts: List[str],
    url: str = "http://localhost:8000/generate",
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    n: int = 1,
    stream: bool = False,
    tokenizer=None,                 # ← new optional arg
    max_retries: int = 3,
    backoff: float = 1.0,
    timeout: float = 30.0,
) -> List[List[str]]:
    """Robust POST to ``/generate`` with retry + schema‑agnostic decoding.

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
    :param n: Number of completions per prompt.
    :type n: int
    :param stream: Whether to use chunked streaming responses.
    :type stream: bool
    :param tokenizer: Optional tokenizer to decode token ID arrays.
    :type tokenizer: Any
    :param max_retries: Number of attempts before surfacing the error.
    :type max_retries: int
    :param backoff: Base backoff in seconds; exponential across attempts.
    :type backoff: float
    :param timeout: Per‑request timeout in seconds.
    :type timeout: float
    :returns: List of per‑prompt lists of texts.
    :rtype: list[list[str]]
    :raises RuntimeError: When the server returns a non‑200 response or after
        exhausting retries.
    """
    payload = dict(
        prompts=prompts, temperature=temperature, top_p=top_p,
        n=n, max_tokens=max_tokens, stream=stream,
    )

    for attempt in range(max_retries):
        try:
            r = requests.post(url, json=payload, timeout=timeout, stream=stream)
            if r.status_code == 200:
                if stream:
                    return _collect_stream_texts(r, len(prompts))
                return _parse_nonstream_json(r.json(), tokenizer)
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:120]}")
        except (requests.ConnectionError, requests.Timeout, RuntimeError) as err:
            if attempt < max_retries - 1:
                time.sleep(backoff * (2**attempt))
            else:
                raise RuntimeError(f"safe_generate failed: {err}") from err


def _collect_stream_texts(response, num_prompts: int) -> List[List[str]]:
    """Collect and join streaming response chunks per prompt index.

    :param response: Requests response object streaming chunked JSON lines.
    :type response: requests.Response
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
