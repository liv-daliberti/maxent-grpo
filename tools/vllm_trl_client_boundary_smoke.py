#!/usr/bin/env python
"""Boundary smoke test for the patched TRL VLLMClient.generate path."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, List

from transformers import AutoConfig, AutoTokenizer

from maxent_grpo.training import baseline


def _normalize_base_url(raw: str) -> str:
    base = str(raw or "").strip()
    if not base:
        raise ValueError("missing URL")
    if base.endswith("/generate/"):
        return base[: -len("/generate/")]
    if base.endswith("/generate"):
        return base[: -len("/generate")]
    return base.rstrip("/")


def _flatten(sequences: Iterable[Iterable[int]]) -> List[int]:
    return [int(token_id) for seq in sequences for token_id in seq]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that the patched TRL VLLMClient.generate path enforces "
            "the tokenizer/model token boundary before training starts."
        )
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--sample-prompts", type=int, default=4)
    parser.add_argument("--sample-max-tokens", type=int, default=24)
    args = parser.parse_args()

    os.environ["MAXENT_VLLM_SERVER_MODEL_NAME"] = args.model_name

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

    base_url = _normalize_base_url(args.url)
    print(
        "[vllm-trl-client-boundary-smoke] "
        f"model={args.model_name} base_url={base_url} tokenizer_limit={tokenizer_limit} model_limit={model_limit}"
    )

    baseline._patch_trl_vllm_client_init()
    from trl.extras.vllm_client import VLLMClient

    client = VLLMClient(base_url=base_url, connection_timeout=float(args.timeout))
    prompts = [
        f"TRL client boundary smoke prompt {idx}."
        for idx in range(max(int(args.sample_prompts), 1))
    ]
    completion_ids = client.generate(
        prompts,
        n=1,
        temperature=1.0,
        top_p=1.0,
        max_tokens=int(args.sample_max_tokens),
    )
    flat = _flatten(completion_ids)
    invalid_tokens = [
        int(token_id)
        for token_id in flat
        if int(token_id) < 0 or int(token_id) >= int(tokenizer_limit)
    ]
    if invalid_tokens:
        sample = invalid_tokens[:16]
        raise SystemExit(
            "patched TRL client smoke failed: "
            f"sampled tokens outside tokenizer limit {tokenizer_limit}: {sample}"
        )
    print(
        "[vllm-trl-client-boundary-smoke] passed "
        f"sampled_tokens={len(flat)}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[vllm-trl-client-boundary-smoke] {exc}", file=sys.stderr)
        raise
