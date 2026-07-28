#!/usr/bin/env python3
"""Probe untouched-model support for executable MathIR action-menu modes."""

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
    families = sorted(set(str(value) for value in dataset["mathir_family"]))
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in dataset:
        family = str(row["mathir_family"])
        if counts[family] >= rows_per_family:
            continue
        selected.append(dict(row))
        counts[family] += 1
        if all(counts[family] >= rows_per_family for family in families):
            break
    if any(counts[family] != rows_per_family for family in families):
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
    parser.add_argument("--rows-per-family", type=int, default=2)
    parser.add_argument("--samples-per-row", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=590051)
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()

    if args.rows_per_family <= 0 or args.samples_per_row <= 0:
        raise SystemExit("probe row and sample counts must be positive")
    model_root = args.model.resolve()
    data_root = args.data_root.resolve()
    dataset = load_from_disk(str(data_root / "eval"))["multi_answer"]
    rows = _select_rows(dataset, int(args.rows_per_family))

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_root),
        local_files_only=True,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(int(args.cpu_threads), 1))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_root),
        local_files_only=True,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    prompts: list[str] = []
    references: list[str] = []
    row_indices: list[int] = []
    families: list[str] = []
    for row_index, row in enumerate(rows):
        for _sample_index in range(int(args.samples_per_row)):
            prompts.append(apply_qwen_boxed_template(str(row["problem"])))
            references.append(str(row["answer"]))
            row_indices.append(row_index)
            families.append(str(row["mathir_family"]))

    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))
    completions: list[str] = []
    for start in range(0, len(prompts), int(args.batch_size)):
        batch = prompts[start : start + int(args.batch_size)]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
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
    distinct_by_row: dict[int, set[str]] = defaultdict(set)
    for completion, reference, row_index, family in zip(
        completions,
        references,
        row_indices,
        families,
    ):
        _info, reward = boxed_reward_fn(completion, reference, fast=True)
        key = validated_modebench_outcome_key(completion, reference)
        correct = bool(float(reward) > 0.0)
        if correct != (key is not None):
            raise RuntimeError("reward/key parity failure")
        attempts_by_family[family] += 1
        if correct:
            correct_by_row[row_index] += 1
            correct_by_family[family] += 1
            assert key is not None
            distinct_by_row[row_index].add(key)
        attempts.append(
            {
                "row_index": row_index,
                "family": family,
                "correct": correct,
                "canonical_key": key,
                "completion": completion,
            }
        )

    family_summary: dict[str, Any] = {}
    for family in sorted(attempts_by_family):
        family_rows = [
            index
            for index, row in enumerate(rows)
            if str(row["mathir_family"]) == family
        ]
        family_summary[family] = {
            "attempts": attempts_by_family[family],
            "correct": correct_by_family[family],
            "correct_fraction": (
                correct_by_family[family] / attempts_by_family[family]
            ),
            "groups_any_correct": sum(
                int(correct_by_row[index] > 0) for index in family_rows
            ),
            "groups_two_modes": sum(
                int(len(distinct_by_row[index]) >= 2) for index in family_rows
            ),
        }

    total_attempts = len(attempts)
    total_correct = sum(correct_by_row.values())
    groups_any = sum(
        int(correct_by_row[index] > 0) for index in range(len(rows))
    )
    groups_two_modes = sum(
        int(len(distinct_by_row[index]) >= 2) for index in range(len(rows))
    )
    families_any = sum(
        int(summary["correct"] > 0) for summary in family_summary.values()
    )
    gates = {
        "at_least_one_verified_generation": total_correct >= 1,
        "at_least_three_families_with_reward": families_any >= 3,
        "at_least_quarter_groups_with_reward": groups_any >= len(rows) / 4,
        "at_least_one_group_with_two_modes": groups_two_modes >= 1,
    }
    payload = {
        "schema": "mathir_action_menu_base_probe_v1",
        "model_root": str(model_root),
        "model_config_sha256": _sha256(model_root / "config.json"),
        "data_root": str(data_root),
        "data_identity_sha256": _sha256(data_root / "identity.json"),
        "device": str(device),
        "rows_per_family": int(args.rows_per_family),
        "samples_per_row": int(args.samples_per_row),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "max_new_tokens": int(args.max_new_tokens),
        "seed": int(args.seed),
        "total_attempts": total_attempts,
        "total_correct": total_correct,
        "total_correct_fraction": total_correct / total_attempts,
        "groups_any_correct": groups_any,
        "groups_two_modes": groups_two_modes,
        "families_with_reward": families_any,
        "family_summary": family_summary,
        "gates": gates,
        "passed": all(gates.values()),
        "attempts": attempts,
    }
    _write_json_atomic(args.output.resolve(), payload)
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "attempts"},
            indent=2,
            sort_keys=True,
        )
    )
    if not payload["passed"]:
        raise SystemExit("MathIR action-menu base probe failed")


if __name__ == "__main__":
    main()
