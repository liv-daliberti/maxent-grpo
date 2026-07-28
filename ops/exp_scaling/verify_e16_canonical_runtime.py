#!/usr/bin/env python3
"""Fail-closed runtime/tokenizer identity for E16's two canonical policies."""

from __future__ import annotations

import json
import os
import importlib.metadata
from hashlib import sha256
from pathlib import Path
import sys

import vllm
import vllm.envs as vllm_envs
from transformers import AutoTokenizer
from vllm.model_executor.layers.sampler import get_sampler

from oat_drgrpo.canonical_actions import resolve_canonical_action_space

try:  # package import in tests
    from .verify_e14_runtime import (
        EXPECTED_MODEL_WEIGHTS_HASH,
        EXPECTED_TOKENIZER_FILES_HASH,
        EXPECTED_TOKENIZER_REVISION,
        EXPECTED_TOKENIZER_VOCAB_HASH,
        EXPECTED_VLLM_VERSION,
        _file_sha256,
    )
except ImportError:  # direct script execution
    from verify_e14_runtime import (  # type: ignore[no-redef]
        EXPECTED_MODEL_WEIGHTS_HASH,
        EXPECTED_TOKENIZER_FILES_HASH,
        EXPECTED_TOKENIZER_REVISION,
        EXPECTED_TOKENIZER_VOCAB_HASH,
        EXPECTED_VLLM_VERSION,
        _file_sha256,
    )


EXPECTED_PYTHON = (
    "/n/fs/similarity/maxent-grpo/var/seed_paper_eval/paper310/bin/python3.10"
)
EXPECTED_PYTHON_VERSION_PREFIX = "3.10.20 | packaged by conda-forge"
EXPECTED_PYTHON_SHA256 = (
    "dd6930668bcd2a57281a8ffed1da12f543d2751b551ee5c3a7dbcc3916259a52"
)
EXPECTED_DISTRIBUTIONS = {
    "torch": (
        "2.6.0",
        "5e44aece2eac2265d3ecf180fe1906e6610218d7ea7b4f066fdbbdad36186901",
    ),
    "transformers": (
        "4.51.3",
        "f5d6e41625a6d70ba8614b478dd3a1a7f36cfcda8b49bdaeeec5cc6eb78134f5",
    ),
    "deepspeed": (
        "0.16.8",
        "4e8de5806db45055f976fe353cdfcc4f4bb3fef6bc4275beafd29cca1a5dadb2",
    ),
    "oat-llm": (
        "0.1.3.post1",
        "959ef5076aab4e6c455923f0fa09700a3f10acebec077f4199bd8af801ab2edc",
    ),
}
EXPECTED_OAT_SOURCE_SHA256 = (
    "8d6578f0598f5c79060b55e3073558ab7a3ac55705500d62594a2472894cafb4"
)


def _installed_distribution_identity(name: str) -> dict[str, str]:
    distribution = importlib.metadata.distribution(name)
    record = distribution.read_text("RECORD")
    if record is None:
        raise ValueError(f"{name} has no installed RECORD identity")
    return {
        "record_sha256": sha256(record.encode("utf-8")).hexdigest(),
        "version": distribution.version,
    }


def _python_tree_hash(root: Path) -> tuple[int, str]:
    digest = sha256()
    files = sorted(
        root.rglob("*.py"), key=lambda path: path.relative_to(root).as_posix()
    )
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(_file_sha256(path)))
    return len(files), digest.hexdigest()


def verify_runtime() -> dict[str, object]:
    if vllm.__version__ != EXPECTED_VLLM_VERSION:
        raise ValueError(
            f"vLLM drifted: expected={EXPECTED_VLLM_VERSION} got={vllm.__version__}"
        )
    configured_engine = os.environ.get("VLLM_USE_V1")
    if configured_engine != "0" or bool(vllm_envs.VLLM_USE_V1):
        raise ValueError(
            "E16 requires VLLM_USE_V1=0 before importing vLLM; "
            f"environment={configured_engine!r} resolved={vllm_envs.VLLM_USE_V1!r}"
        )
    sampler_module = type(get_sampler()).__module__
    if sampler_module != "vllm.model_executor.layers.sampler":
        raise ValueError(f"E16 expected the V0 sampler, got {sampler_module}")

    python_path = Path(sys.executable).resolve()
    python_hash = _file_sha256(python_path)
    if str(python_path) != EXPECTED_PYTHON or python_hash != EXPECTED_PYTHON_SHA256:
        raise ValueError(
            f"Python executable drifted: path={python_path} sha256={python_hash}"
        )
    if not sys.version.startswith(EXPECTED_PYTHON_VERSION_PREFIX):
        raise ValueError(f"Python runtime drifted: {sys.version}")
    distributions = {
        name: _installed_distribution_identity(name)
        for name in EXPECTED_DISTRIBUTIONS
    }
    for name, (version, record_hash) in EXPECTED_DISTRIBUTIONS.items():
        if distributions[name] != {
            "record_sha256": record_hash,
            "version": version,
        }:
            raise ValueError(f"installed {name} identity drifted")
    oat_root = Path(
        importlib.metadata.distribution("oat-llm").locate_file("oat")
    ).resolve()
    oat_source_file_count, oat_source_hash = _python_tree_hash(oat_root)
    if oat_source_hash != EXPECTED_OAT_SOURCE_SHA256:
        raise ValueError("installed OAT Python source drifted")

    cache_root = Path(os.environ["TRANSFORMERS_CACHE"])
    cached_model = cache_root / "models--Qwen--Qwen2.5-0.5B-Instruct"
    revision = (cached_model / "refs" / "main").read_text().strip()
    if revision != EXPECTED_TOKENIZER_REVISION:
        raise ValueError(
            "tokenizer revision drifted: "
            f"expected={EXPECTED_TOKENIZER_REVISION} observed={revision}"
        )
    snapshot = cached_model / "snapshots" / revision
    required_files = (
        "config.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    )
    missing = [name for name in required_files if not (snapshot / name).is_file()]
    if missing:
        raise ValueError(f"tokenizer snapshot is missing {missing}")
    weights_hash = _file_sha256(snapshot / "model.safetensors")
    if weights_hash != EXPECTED_MODEL_WEIGHTS_HASH:
        raise ValueError("model weights drifted")
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), trust_remote_code=True, local_files_only=True
    )
    vocab_hash = sha256(
        json.dumps(
            sorted(
                (str(token), int(token_id))
                for token, token_id in tokenizer.get_vocab().items()
            ),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    files_digest = sha256()
    for name in required_files:
        files_digest.update(name.encode("utf-8"))
        files_digest.update((snapshot / name).read_bytes())
    if files_digest.hexdigest() != EXPECTED_TOKENIZER_FILES_HASH:
        raise ValueError("tokenizer files drifted")
    if vocab_hash != EXPECTED_TOKENIZER_VOCAB_HASH:
        raise ValueError("tokenizer vocabulary drifted")

    graph = resolve_canonical_action_space(tokenizer, "graph_coloring")
    countdown = resolve_canonical_action_space(tokenizer, "countdown")
    if graph.token_ids_by_position != ((16, 17, 18),) * 3:
        raise ValueError(f"graph action token IDs drifted: {graph.token_ids_by_position}")
    if countdown.token_ids_by_position != (
        (16, 17, 18, 19, 20, 21),
        (16, 17, 18),
        (16, 17, 18, 19, 20, 21),
    ):
        raise ValueError(
            f"Countdown action token IDs drifted: {countdown.token_ids_by_position}"
        )
    return {
        "canonical_training_sampler": (
            "learner_hf_restricted_inverse_cdf_fixed_shape_causal_placeholder"
        ),
        "controller_entropy_metric": "canonical_exact_sequence_entropy",
        "controller_entropy_units": "canonical_action_nats_exact_v1",
        "countdown": {
            "max_action_entropy_nats": countdown.max_sequence_entropy,
            "sequence_count": countdown.sequence_count,
            "token_ids_by_position": countdown.token_ids_by_position,
        },
        "engine": "v0",
        "graph_coloring": {
            "max_action_entropy_nats": graph.max_sequence_entropy,
            "sequence_count": graph.sequence_count,
            "token_ids_by_position": graph.token_ids_by_position,
        },
        "model_weights_hash": weights_hash,
        "installed_distributions": distributions,
        "oat_python_source": {
            "file_count": oat_source_file_count,
            "root": str(oat_root),
            "sha256": oat_source_hash,
        },
        "python": {
            "executable": str(python_path),
            "executable_sha256": python_hash,
            "version": sys.version,
        },
        "sampler_module": sampler_module,
        "schema": "e16_canonical_runtime_identity_v1",
        "tokenizer": "Qwen/Qwen2.5-0.5B-Instruct",
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_files_hash": files_digest.hexdigest(),
        "tokenizer_revision": revision,
        "tokenizer_vocab_hash": vocab_hash,
        "tokenizer_vocab_size": len(tokenizer),
        "vllm_role": "evaluation_only",
        "vllm_version": vllm.__version__,
    }


def main() -> None:
    try:
        identity = verify_runtime()
    except (FileNotFoundError, KeyError, ValueError) as error:
        raise SystemExit(f"E16 canonical runtime rejected: {error}") from error
    print(json.dumps(identity, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
