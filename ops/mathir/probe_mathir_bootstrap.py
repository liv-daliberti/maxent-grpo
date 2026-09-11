#!/usr/bin/env python3
"""Measure whether the frozen 0.5B policy supplies usable MathIR reward."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oat_drgrpo.math_grader import (  # noqa: E402
    boxed_reward_fn,
    validated_modebench_outcome_key,
)
from oat_drgrpo.templates import apply_qwen_boxed_template  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_rows(dataset: Any, rows_per_family: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    families = sorted(set(str(value) for value in dataset["mathir_family"]))
    for row in dataset:
        family = str(row["mathir_family"])
        if counts[family] >= int(rows_per_family):
            continue
        selected.append(dict(row))
        counts[family] += 1
        if all(counts[family_name] >= int(rows_per_family) for family_name in families):
            break
    if any(counts[family] != int(rows_per_family) for family in families):
        raise RuntimeError(f"insufficient rows per family: {dict(counts)}")
    return selected


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-per-family", type=int, default=4)
    parser.add_argument("--samples-per-row", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=450051)
    parser.add_argument("--cpu-threads", type=int, default=16)
    args = parser.parse_args()

    if args.rows_per_family <= 0 or args.samples_per_row <= 0:
        raise SystemExit("probe row and sample counts must be positive")
    model_root = args.model.resolve()
    data_root = args.data_root.resolve()
    dataset = load_from_disk(str(data_root / "eval"))["multi_answer"]
    rows = _select_rows(dataset, args.rows_per_family)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_root),
        local_files_only=True,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    torch.set_num_threads(max(int(args.cpu_threads), 1))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_root),
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    model.eval()

    prompts: list[str] = []
    references: list[str] = []
    row_indices: list[int] = []
    families: list[str] = []
    for row_index, row in enumerate(rows):
        for _ in range(int(args.samples_per_row)):
            prompts.append(apply_qwen_boxed_template(str(row["problem"])))
            references.append(str(row["answer"]))
            row_indices.append(row_index)
            families.append(str(row["mathir_family"]))

    torch.manual_seed(int(args.seed))
    completions: list[str] = []
    for start in range(0, len(prompts), int(args.batch_size)):
        batch = prompts[start : start + int(args.batch_size)]
        inputs = tokenizer(batch, return_tensors="pt", padding=True)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                max_new_tokens=int(args.max_new_tokens),
                pad_token_id=tokenizer.pad_token_id,
            )
        completions.extend(
            tokenizer.batch_decode(
                outputs[:, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
        )

    attempts: list[dict[str, Any]] = []
    correct_by_row: Counter[int] = Counter()
    correct_by_family: Counter[str] = Counter()
    attempts_by_family: Counter[str] = Counter()
    nonseed_by_family: Counter[str] = Counter()
    distinct_by_row: dict[int, set[str]] = defaultdict(set)
    for completion, reference, row_index, family in zip(
        completions,
        references,
        row_indices,
        families,
    ):
        _info, reward = boxed_reward_fn(completion, reference, fast=True)
        key = validated_modebench_outcome_key(completion, reference)
        spec = json.loads(reference)
        correct = bool(float(reward) > 0.0)
        nonseed = bool(correct and key != spec["public_seed_key"])
        attempts_by_family[family] += 1
        if correct:
            correct_by_row[row_index] += 1
            correct_by_family[family] += 1
            assert key is not None
            distinct_by_row[row_index].add(key)
        if nonseed:
            nonseed_by_family[family] += 1
        attempts.append(
            {
                "row_index": row_index,
                "family": family,
                "correct": correct,
                "nonseed": nonseed,
                "canonical_key": key,
                "completion": completion,
            }
        )

    row_any = [
        float(correct_by_row[row_index] > 0) for row_index in range(len(rows))
    ]
    family_summary: dict[str, Any] = {}
    for family in sorted(attempts_by_family):
        family_rows = [
            row_index
            for row_index, row in enumerate(rows)
            if str(row["mathir_family"]) == family
        ]
        family_summary[family] = {
            "attempts": attempts_by_family[family],
            "correct": correct_by_family[family],
            "correct_fraction": (
                correct_by_family[family] / attempts_by_family[family]
            ),
            "groups_any_correct_fraction": sum(
                float(correct_by_row[index] > 0) for index in family_rows
            )
            / len(family_rows),
            "nonseed_correct": nonseed_by_family[family],
            "mean_distinct_correct_per_group": sum(
                len(distinct_by_row[index]) for index in family_rows
            )
            / len(family_rows),
        }
    total_correct = sum(correct_by_family.values())
    total_attempts = len(attempts)
    gates = {
        "total_correct_fraction_at_least_0p10": (
            total_correct / total_attempts >= 0.10
        ),
        "group_any_correct_fraction_at_least_0p75": (
            sum(row_any) / len(row_any) >= 0.75
        ),
        "every_family_group_any_at_least_0p50": all(
            summary["groups_any_correct_fraction"] >= 0.50
            for summary in family_summary.values()
        ),
    }
    payload = {
        "schema": "e45_mathir_bootstrap_probe_v1",
        "model_root": str(model_root),
        "model_config_sha256": _sha256(model_root / "config.json"),
        "data_root": str(data_root),
        "data_identity_sha256": _sha256(data_root / "identity.json"),
        "rows_per_family": int(args.rows_per_family),
        "samples_per_row": int(args.samples_per_row),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "max_new_tokens": int(args.max_new_tokens),
        "seed": int(args.seed),
        "total_attempts": total_attempts,
        "total_correct": total_correct,
        "total_correct_fraction": total_correct / total_attempts,
        "groups_any_correct_fraction": sum(row_any) / len(row_any),
        "family_summary": family_summary,
        "gates": gates,
        "passed": all(gates.values()),
        "attempts": attempts,
    }
    _write_json_atomic(args.output.resolve(), payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "attempts"}, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit("MathIR bootstrap probe failed")


if __name__ == "__main__":
    main()
