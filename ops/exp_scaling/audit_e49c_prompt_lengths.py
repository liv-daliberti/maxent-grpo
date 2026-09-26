#!/usr/bin/env python3
"""Fail closed if an E49C menu prompt would be dropped or truncated."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import tempfile

from datasets import load_from_disk
from transformers import AutoTokenizer


ROOT = pathlib.Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "src"))
from oat_drgrpo.math_strategy_menu import parse_strategy_menu
from oat_drgrpo.templates import apply_qwen_math_template


MODEL = (
    ROOT
    / "var/cache/huggingface/transformers/"
    "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
    "7ae557604adf67be50417f59c2c2f167def9a775"
)


def _write(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_name, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL), local_files_only=True
    )
    records = []
    for tree_name in ("train", "eval"):
        dataset_dict = load_from_disk(str(args.data / tree_name))
        split = next(iter(dataset_dict))
        for index, row in enumerate(dataset_dict[split]):
            problem = str(row["problem"])
            menu = parse_strategy_menu(problem)
            if menu is None:
                raise RuntimeError(f"{tree_name}:{index} has no finite menu")
            rendered = apply_qwen_math_template(problem)
            token_count = len(
                tokenizer(
                    rendered,
                    add_special_tokens=False,
                    truncation=False,
                )["input_ids"]
            )
            records.append(
                {
                    "tree": tree_name,
                    "split": split,
                    "index": index,
                    "tokens": token_count,
                    "menu_sha256": menu.sha256,
                }
            )
    over_limit = [
        row for row in records if row["tokens"] > args.max_prompt_tokens
    ]
    report = {
        "schema": "e49c_prompt_length_audit_v1",
        "pass": not over_limit,
        "rows": len(records),
        "max_prompt_tokens": args.max_prompt_tokens,
        "observed_max_tokens": max(row["tokens"] for row in records),
        "observed_mean_tokens": (
            sum(row["tokens"] for row in records) / len(records)
        ),
        "over_limit": over_limit,
        "longest": sorted(
            records, key=lambda row: row["tokens"], reverse=True
        )[:10],
    }
    _write(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if over_limit:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
