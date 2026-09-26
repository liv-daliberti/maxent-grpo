#!/usr/bin/env python3
"""Fail-closed E14 engine/tokenizer identity check before Slurm submission."""

from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path

import vllm
import vllm.envs as vllm_envs
from transformers import AutoTokenizer
from vllm.model_executor.layers.sampler import get_sampler

from oat_drgrpo.canonical_actions import resolve_graph_color_action_token_ids


EXPECTED_VLLM_VERSION = "0.8.4"
EXPECTED_TOKENIZER_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
EXPECTED_TOKENIZER_FILES_HASH = (
    "caa4fecabf4ddfe3d6678b909ca31e73337cfbfd9aa6befc935a9d1d90ca089d"
)
EXPECTED_TOKENIZER_VOCAB_HASH = (
    "698c955b0b438a535083d1771ef8b41069afba4b3d51a482558bdb68ea55e800"
)
EXPECTED_MODEL_WEIGHTS_HASH = (
    "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if vllm.__version__ != EXPECTED_VLLM_VERSION:
        raise SystemExit(
            "E14 requires frozen vLLM version "
            f"{EXPECTED_VLLM_VERSION}; got {vllm.__version__}"
        )
    configured_engine = os.environ.get("VLLM_USE_V1")
    if configured_engine != "0" or bool(vllm_envs.VLLM_USE_V1):
        raise SystemExit(
            "E14 requires VLLM_USE_V1=0 before importing vLLM; "
            f"environment={configured_engine!r} resolved={vllm_envs.VLLM_USE_V1!r}"
        )
    sampler_module = type(get_sampler()).__module__
    if sampler_module != "vllm.model_executor.layers.sampler":
        raise SystemExit(f"E14 expected the V0 sampler, got {sampler_module}")

    tokenizer_name = "Qwen/Qwen2.5-0.5B-Instruct"
    cache_root = Path(os.environ["TRANSFORMERS_CACHE"])
    cached_model = cache_root / "models--Qwen--Qwen2.5-0.5B-Instruct"
    revision = (cached_model / "refs" / "main").read_text().strip()
    if revision != EXPECTED_TOKENIZER_REVISION:
        raise SystemExit(
            "E14 tokenizer revision drifted: "
            f"expected={EXPECTED_TOKENIZER_REVISION} observed={revision}"
        )
    tokenizer_snapshot = cached_model / "snapshots" / revision
    required_files = (
        "config.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    )
    missing = [
        name for name in required_files if not (tokenizer_snapshot / name).is_file()
    ]
    if missing:
        raise SystemExit(
            f"E14 tokenizer snapshot {tokenizer_snapshot} is missing {missing}"
        )
    model_weights_hash = _file_sha256(tokenizer_snapshot / "model.safetensors")
    if model_weights_hash != EXPECTED_MODEL_WEIGHTS_HASH:
        raise SystemExit(
            "E14 model weights drifted: "
            f"expected={EXPECTED_MODEL_WEIGHTS_HASH} observed={model_weights_hash}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_snapshot),
        trust_remote_code=True,
        local_files_only=True,
    )
    action_token_ids = resolve_graph_color_action_token_ids(tokenizer)
    tokenizer_vocab_hash = sha256(
        json.dumps(
            sorted(
                (str(token), int(token_id))
                for token, token_id in tokenizer.get_vocab().items()
            ),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    tokenizer_files_hash = sha256()
    for name in required_files:
        tokenizer_files_hash.update(name.encode("utf-8"))
        tokenizer_files_hash.update((tokenizer_snapshot / name).read_bytes())
    tokenizer_files_digest = tokenizer_files_hash.hexdigest()
    if tokenizer_files_digest != EXPECTED_TOKENIZER_FILES_HASH:
        raise SystemExit(
            "E14 tokenizer file identity drifted: "
            f"expected={EXPECTED_TOKENIZER_FILES_HASH} "
            f"observed={tokenizer_files_digest}"
        )
    if tokenizer_vocab_hash != EXPECTED_TOKENIZER_VOCAB_HASH:
        raise SystemExit(
            "E14 tokenizer vocabulary identity drifted: "
            f"expected={EXPECTED_TOKENIZER_VOCAB_HASH} "
            f"observed={tokenizer_vocab_hash}"
        )
    if action_token_ids != (16, 17, 18):
        raise SystemExit(f"E14 action token ids drifted: {action_token_ids}")
    print(
        json.dumps(
            {
                "action_count": 3,
                "action_token_ids": list(action_token_ids),
                "canonical_training_sampler": (
                    "learner_hf_restricted_inverse_cdf_"
                    "fixed_shape_causal_placeholder"
                ),
                "engine": "v0",
                "max_action_entropy_nats": 3.295836866004329,
                "model_weights_hash": model_weights_hash,
                "sampler_module": sampler_module,
                "tokenizer": tokenizer_name,
                "tokenizer_class": type(tokenizer).__name__,
                "tokenizer_files_hash": tokenizer_files_digest,
                "tokenizer_revision": revision,
                "tokenizer_vocab_hash": tokenizer_vocab_hash,
                "tokenizer_vocab_size": len(tokenizer),
                "vllm_version": vllm.__version__,
                "vllm_role": "evaluation_only",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
