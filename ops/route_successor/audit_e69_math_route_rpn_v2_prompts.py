#!/usr/bin/env python3
"""Freeze the RPN-v2 prompt contract over the sealed E69 route-dev rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from ops.math500 import materialize_e39_math12k_384 as e39
from oat_drgrpo.templates import apply_qwen_math_route_template


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "var/data/math12k_384_route_dev128_v1"
PARENT_MANIFEST = DATA_ROOT / "MATERIALIZATION_MANIFEST.json"
OUTPUT = DATA_ROOT / "RPN_V2_PROMPT_MANIFEST.json"
PROMPT_MAX_LENGTH = 1_024
EXPECTED_DEV_ROWS = 128
EXPECTED_ORDERED_PROBLEM_HASH = (
    "2f37f517a6ab6badce6c8d2fcc4e05483449a06d3aee89d87b5c4ecb7400fa43"
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest() -> dict[str, Any]:
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    parent = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    dataset_dict = load_from_disk(str(DATA_ROOT / "eval"))
    if set(dataset_dict) != {"math_dev"}:
        raise RuntimeError("sealed E69 eval root is not exactly math_dev")
    dataset = dataset_dict["math_dev"]
    if len(dataset) != EXPECTED_DEV_ROWS:
        raise RuntimeError("sealed E69 route-dev split is not 128 rows")
    problems = [str(dataset[index]["problem"]) for index in range(len(dataset))]
    problem_hashes = [
        hashlib.sha256(problem.encode("utf-8")).hexdigest() for problem in problems
    ]
    ordered_problem_hash = _json_sha256(problem_hashes)
    if ordered_problem_hash != EXPECTED_ORDERED_PROBLEM_HASH:
        raise RuntimeError("RPN-v2 prompt population differs from the sealed rows")
    parent_problem_hash = parent["audit"]["dev_identity"][
        "ordered_problem_sha256"
    ]
    if parent_problem_hash != ordered_problem_hash:
        raise RuntimeError("parent manifest and RPN-v2 population disagree")

    e39._verify_files(e39.TOKENIZER_ROOT, e39.TOKENIZER_FILES, label="tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        str(e39.TOKENIZER_ROOT),
        local_files_only=True,
    )
    prompts = [apply_qwen_math_route_template(problem) for problem in problems]
    token_ids = tokenizer(
        prompts,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]
    lengths = [len(row) for row in token_ids]
    if max(lengths) > PROMPT_MAX_LENGTH:
        raise RuntimeError("an RPN-v2 route-dev prompt exceeds 1,024 tokens")
    prompt_hashes = [
        hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in prompts
    ]
    return {
        "schema": "e69_math_route_rpn_v2_prompt_manifest_v1",
        "created_at": "2026-07-28",
        "population": {
            "data_root": str(DATA_ROOT.relative_to(ROOT)),
            "split": "math_dev",
            "rows": len(problems),
            "ordered_problem_sha256": ordered_problem_hash,
            "parent_materialization_manifest_sha256": _sha256(PARENT_MANIFEST),
        },
        "prompt_contract": {
            "template": "qwen_math_route_rpn_v2",
            "route_language": "math-route-rpn-v2",
            "prompt_max_length": PROMPT_MAX_LENGTH,
            "minimum_tokens": min(lengths),
            "maximum_tokens": max(lengths),
            "mean_tokens": sum(lengths) / len(lengths),
            "ordered_token_lengths_sha256": _json_sha256(lengths),
            "ordered_prompt_sha256": _json_sha256(prompt_hashes),
        },
        "provenance": {
            "generator": str(Path(__file__).resolve().relative_to(ROOT)),
            "generator_sha256": _sha256(Path(__file__).resolve()),
            "templates_sha256": _sha256(ROOT / "src/oat_drgrpo/templates.py"),
            "tokenizer_root": str(e39.TOKENIZER_ROOT.relative_to(ROOT)),
        },
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"fresh RPN-v2 prompt manifest required: {output}")
    payload = build_manifest()
    _atomic_json(output, payload)
    print(json.dumps(payload["prompt_contract"], sort_keys=True))


if __name__ == "__main__":
    main()
