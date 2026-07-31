#!/usr/bin/env python3
"""Materialize the frozen ConstructiveCode v7 train-only SFT corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V5_ROOT = ROOT / "var/data/constructive_code_v5"
V6_MANIFEST = ROOT / "var/data/constructive_code_v6/manifest.json"
V6_GATE = ROOT / "var/artifacts/constructive_code_v6_gate_audit.json"
PROTOCOL = ROOT / "paper/preregistration/constructive_code_v7_train_only_sft_20260730.md"
OUTPUT_ROOT = ROOT / "var/data/constructive_code_v7_sft"
EXAMPLES = OUTPUT_ROOT / "examples.jsonl"
MANIFEST = OUTPUT_ROOT / "manifest.json"
TRAIN = {
    "327_B": "327_b",
    "659_C": "659_c",
    "1283_C": "1283_c",
    "1102_B": "1102_b",
}
DEVELOPMENT = ("359_B", "988_A", "1399_D")
EVALUATION = ("361_B", "1294_C", "149_C")
SYSTEM_MESSAGE = (
    "Write a complete Python 3 program that solves the problem. Return only "
    "the program source, without Markdown fences or explanation."
)


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def canonical_sha(value: Any) -> str:
    return sha_bytes(json.dumps(
        value, allow_nan=False, ensure_ascii=True,
        separators=(",", ":"), sort_keys=True,
    ).encode("ascii"))


def atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(temporary, path)


def prompt(statement: str) -> str:
    return (
        "<|im_start|>system\n" + SYSTEM_MESSAGE
        + "<|im_end|>\n<|im_start|>user\n" + statement
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for path in (V6_MANIFEST, V6_GATE, PROTOCOL):
        if not path.is_file():
            raise FileNotFoundError(path)
    v6 = json.loads(V6_MANIFEST.read_text())
    gate = json.loads(V6_GATE.read_text())
    if (
        v6.get("status") != "admitted_pre_model"
        or v6.get("split_assignment") != {
            **{key: "train" for key in TRAIN},
            **{key: "development" for key in DEVELOPMENT},
            **{key: "evaluation" for key in EVALUATION},
        }
        or v6.get("evaluation_rows_loaded") is not False
        or v6.get("language_model_sampling") is not False
        or gate.get("status") != "pass"
        or gate.get("expected_replay_count") != 960
        or gate.get("observed_replay_count") != 960
        or gate.get("evaluation_rows_loaded") is not False
        or gate.get("language_model_sampling") is not False
    ):
        raise RuntimeError("ConstructiveCode v6 antecedent is not an exact pre-model pass")

    examples = []
    task_summary = []
    for problem_id, relative in TRAIN.items():
        task_root = V5_ROOT / relative
        task_path = task_root / "task.json"
        replay_path = task_root / "py3_replays.jsonl"
        task = json.loads(task_path.read_text())
        rows = [json.loads(line) for line in replay_path.read_text().splitlines() if line]
        correct = [row for row in rows if row.get("known_label") == "correct"]
        incorrect = [row for row in rows if row.get("known_label") == "incorrect"]
        if (
            task.get("source_problem_id") != problem_id
            or len(correct) != 48 or len(incorrect) != 48
            or any(row.get("language") != "py3" for row in rows)
        ):
            raise RuntimeError(f"{problem_id} replay admission drift")
        for row in correct:
            code = row.get("code")
            if not isinstance(code, str) or not code.strip():
                raise RuntimeError(f"{problem_id} has empty correct code")
            if sha_bytes(code.encode("utf-8")) != row.get("submission_sha256"):
                raise RuntimeError(f"{problem_id} correct source hash mismatch")
        selected = sorted(
            correct,
            key=lambda row: (len(row["code"].encode("utf-8")), row["submission_sha256"]),
        )[:16]
        statement = task.get("statement")
        if not isinstance(statement, str) or not statement.strip():
            raise RuntimeError(f"{problem_id} statement missing")
        for rank, row in enumerate(selected):
            code = row["code"]
            examples.append({
                "schema": "constructive-code-v7-sft-example-v1",
                "source_problem_id": problem_id,
                "problem_key": task["problem_key"],
                "witness_family": task["witness_family"],
                "selection_rank": rank,
                "statement": statement,
                "statement_sha256": sha_bytes(statement.encode("utf-8")),
                "prompt": prompt(statement),
                "prompt_sha256": sha_bytes(prompt(statement).encode("utf-8")),
                "code": code,
                "code_utf8_bytes": len(code.encode("utf-8")),
                "submission_sha256": row["submission_sha256"],
            })
        task_summary.append({
            "source_problem_id": problem_id,
            "problem_key": task["problem_key"],
            "witness_family": task["witness_family"],
            "selected_examples": 16,
            "selected_submission_sha256": [row["submission_sha256"] for row in selected],
            "maximum_selected_code_utf8_bytes": max(len(row["code"].encode("utf-8")) for row in selected),
            "task_json_sha256": sha(task_path),
            "replays_sha256": sha(replay_path),
        })
    if len(examples) != 64 or len({row["submission_sha256"] for row in examples}) != 64:
        raise RuntimeError("v7 SFT corpus is not 64 unique train-only programs")
    if any(row["source_problem_id"] in (*DEVELOPMENT, *EVALUATION) for row in examples):
        raise RuntimeError("v7 SFT crossed the train-only boundary")

    lines = "".join(json.dumps(row, allow_nan=False, sort_keys=True) + "\n" for row in examples)
    manifest = {
        "schema": "constructive-code-v7-sft-manifest-v1",
        "status": "pass",
        "selection_rule": "first 16 by (UTF-8 code byte length, submission_sha256)",
        "train_problem_ids": list(TRAIN),
        "development_problem_ids_loaded": [],
        "evaluation_problem_ids_loaded": [],
        "example_count": 64,
        "examples_per_task": 16,
        "unique_submission_count": 64,
        "task_summary": task_summary,
        "examples_sha256": sha_bytes(lines.encode("utf-8")),
        "examples_canonical_sha256": canonical_sha(examples),
        "v6_manifest_sha256": sha(V6_MANIFEST),
        "v6_gate_sha256": sha(V6_GATE),
        "protocol_sha256": sha(PROTOCOL),
        "language_model_sampling": False,
        "evaluation_rows_loaded": False,
    }
    rendered_manifest = json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not EXAMPLES.is_file() or EXAMPLES.read_text() != lines:
            raise RuntimeError("materialized v7 SFT examples drift")
        if not MANIFEST.is_file() or MANIFEST.read_text() != rendered_manifest:
            raise RuntimeError("materialized v7 SFT manifest drift")
        print("[constructive-v7-sft] check passed; no model sampled")
        return
    if EXAMPLES.exists() or MANIFEST.exists():
        raise FileExistsError("fresh ConstructiveCode v7 SFT corpus required")
    atomic(EXAMPLES, lines)
    atomic(MANIFEST, rendered_manifest)
    print("[constructive-v7-sft] materialized 64 train-only examples; no model sampled")


if __name__ == "__main__":
    main()
