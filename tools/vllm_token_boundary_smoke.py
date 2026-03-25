#!/usr/bin/env python
"""Boundary smoke test for the live vLLM /generate endpoint."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any, Iterable, List

from transformers import AutoConfig, AutoTokenizer


def _normalize_generate_url(raw: str) -> str:
    base = str(raw or "").strip()
    if not base:
        raise ValueError("missing URL")
    if base.endswith("/generate/"):
        return base
    if base.endswith("/generate"):
        return f"{base}/"
    return f"{base.rstrip('/')}/generate/"


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    data = json.loads(body)
    if not isinstance(data, dict):
        raise ValueError(f"unexpected response payload type: {type(data)!r}")
    return data


def _extract_completion_ids(payload: dict[str, Any]) -> List[List[int]]:
    raw = payload.get("completion_ids")
    if not isinstance(raw, list):
        raise ValueError("response missing completion_ids")
    completion_ids: List[List[int]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, list):
            raise ValueError(f"completion_ids[{idx}] is not a list")
        completion_ids.append([int(token_id) for token_id in item])
    return completion_ids


def _flatten(sequences: Iterable[Iterable[int]]) -> List[int]:
    return [int(token_id) for seq in sequences for token_id in seq]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that the live vLLM server honors token-boundary controls "
            "before training starts."
        )
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--sample-prompts", type=int, default=4)
    parser.add_argument("--sample-max-tokens", type=int, default=24)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    config = AutoConfig.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    tokenizer_limit = max(
        int(getattr(tokenizer, "vocab_size", 0) or 0),
        int(len(tokenizer)),
    )
    model_limit = int(getattr(config, "vocab_size", 0) or 0)
    if tokenizer_limit <= 0 or model_limit <= 0:
        raise SystemExit(
            f"could not resolve tokenizer/model limits: tokenizer={tokenizer_limit} model={model_limit}"
        )
    if model_limit <= tokenizer_limit:
        print(
            f"[vllm-token-boundary-smoke] tokenizer_limit={tokenizer_limit} model_limit={model_limit} no tail to test"
        )
        return 0

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(eos_token_id, int) or eos_token_id < 0:
        eos_token_id = getattr(tokenizer, "pad_token_id", None)
    if not isinstance(eos_token_id, int) or eos_token_id < 0:
        raise SystemExit("tokenizer is missing eos_token_id/pad_token_id")

    url = _normalize_generate_url(args.url)
    blocked_tail = list(range(int(tokenizer_limit), int(model_limit)))

    print(
        "[vllm-token-boundary-smoke] "
        f"model={args.model_name} url={url} tokenizer_limit={tokenizer_limit} model_limit={model_limit} "
        f"blocked_tail={len(blocked_tail)} eos_token_id={eos_token_id}"
    )

    allow_payload = {
        "prompts": ["Boundary smoke allowlist probe."],
        "temperature": 0.0,
        "top_p": 1.0,
        "n": 1,
        "max_tokens": 1,
        "allowed_token_ids": [int(eos_token_id)],
        "request_id": "token-boundary-allow-eos",
    }
    allow_response = _post_json(url, allow_payload, timeout=float(args.timeout))
    allow_sequences = _extract_completion_ids(allow_response)
    allow_flat = _flatten(allow_sequences)
    if any(int(token_id) != int(eos_token_id) for token_id in allow_flat):
        raise SystemExit(
            "allowlist probe failed: "
            f"expected only eos_token_id={eos_token_id}, got {allow_sequences}"
        )
    print(
        "[vllm-token-boundary-smoke] allowlist probe passed "
        f"completion_ids={allow_sequences}"
    )

    prompts = [
        f"Boundary smoke blocked-tail probe {idx}."
        for idx in range(max(int(args.sample_prompts), 1))
    ]
    blocked_payload = {
        "prompts": prompts,
        "temperature": 1.0,
        "top_p": 1.0,
        "n": 1,
        "max_tokens": int(args.sample_max_tokens),
        "blocked_token_ids": blocked_tail,
        "request_id": "token-boundary-blocked-tail",
    }
    blocked_response = _post_json(url, blocked_payload, timeout=float(args.timeout))
    blocked_sequences = _extract_completion_ids(blocked_response)
    blocked_flat = _flatten(blocked_sequences)
    invalid_tokens = [
        int(token_id)
        for token_id in blocked_flat
        if int(token_id) < 0 or int(token_id) >= int(tokenizer_limit)
    ]
    if invalid_tokens:
        sample = invalid_tokens[:16]
        raise SystemExit(
            "blocked-tail probe failed: "
            f"sampled tokens outside tokenizer limit {tokenizer_limit}: {sample}"
        )
    print(
        "[vllm-token-boundary-smoke] blocked-tail probe passed "
        f"sampled_tokens={len(blocked_flat)}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        print(f"[vllm-token-boundary-smoke] HTTPError: {exc}\n{message}", file=sys.stderr)
        raise
    except urllib.error.URLError as exc:
        print(f"[vllm-token-boundary-smoke] URLError: {exc}", file=sys.stderr)
        raise
