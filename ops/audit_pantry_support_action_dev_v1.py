#!/usr/bin/env python3
"""Independently replay-audit the Pantry support-action development receipt."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import sys
import tempfile


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _hash_tree(root: Path) -> str:
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = "./" + path.relative_to(root).as_posix()
        lines.append(f"{_sha256(path)}  {relative}\n")
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def _atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str((args.source_root / "src").resolve()))
    from datasets import load_from_disk
    from oat_drgrpo.pantry_plan import parse_pantry_plan_spec
    from oat_drgrpo.pantry_support_action import validate_pantry_support_action

    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    identity = json.loads(args.identity.read_text(encoding="utf-8"))
    dataset_identity = json.loads(
        (args.data_root / "identity.json").read_text(encoding="utf-8")
    )
    rows = load_from_disk(str(args.data_root / "dev"))["multi_answer"].to_list()
    errors: list[str] = []
    checks = {
        "job_identity_matches": receipt.get("job_id") == identity.get("job_id"),
        "source_hash_matches": (
            receipt.get("source_hash") == identity.get("source_hash")
            == _hash_tree(args.source_root / "src")
        ),
        "execution_hash_matches": (
            receipt.get("execution_hash") == identity.get("execution_hash")
            == _hash_tree(args.execution_root)
        ),
        "protocol_hash_matches_recorded_empty_file": (
            _sha256(args.protocol) == receipt.get("protocol_sha256")
            == identity.get("protocol_sha256")
            == hashlib.sha256(b"").hexdigest()
        ),
        "dataset_identity_matches": (
            receipt.get("dataset_identity_sha256") == _canonical(dataset_identity)
        ),
        "development_rows_match": receipt.get("data_split_sha256") == _canonical(rows),
        "attempt_count_is_4096": len(receipt.get("attempts", [])) == 4096,
        "information_boundary_declared": receipt.get("information_boundary")
        == {
            "action_space_contains_all_support_combinations": True,
            "certified_feasible_supports_in_context": False,
            "development_only": True,
            "evaluation_prompts_loaded": False,
            "quantity_projection_uses_only_prompt_local_constraints": True,
            "reference_allocation_in_context": False,
        },
    }

    replay_prompt_results = []
    attempts_by_row = {index: [] for index in range(len(rows))}
    for attempt in receipt.get("attempts", []):
        row_index = int(attempt["row_index"])
        if row_index not in attempts_by_row:
            errors.append(f"out-of-range row index {row_index}")
            continue
        attempts_by_row[row_index].append(attempt)
    for row_index, row in enumerate(rows):
        spec = json.loads(row["answer"])
        parsed = parse_pantry_plan_spec(spec)
        ingredient_ids = sorted(item.ingredient_id for item in parsed.ingredients)
        action_space = {
            " ".join(support)
            for width in range(parsed.min_ingredients, parsed.max_ingredients + 1)
            for support in combinations(ingredient_ids, width)
        }
        attempts = sorted(
            attempts_by_row[row_index], key=lambda item: int(item["sample_index"])
        )
        if [int(item["sample_index"]) for item in attempts] != list(range(1, 65)):
            errors.append(f"row {row_index} sample indices differ from 1..64")
        keys = []
        cache = {}
        for attempt in attempts:
            text = str(attempt["text"])
            if text not in action_space:
                errors.append(f"row {row_index} emitted text outside syntax space")
            if text not in cache:
                validation = validate_pantry_support_action(text, spec)
                cache[text] = validation.canonical_key if validation else None
            key = cache[text]
            keys.append(key)
            if key != attempt.get("canonical_key") or bool(key) != bool(
                attempt.get("verified")
            ):
                errors.append(f"row {row_index} replay mismatch")
        prefix = [key for key in keys[:16] if key]
        full = [key for key in keys if key]
        counts = Counter(full)
        replay_prompt_results.append(
            {
                "verified_in_prefix": len(prefix),
                "verified_in_full_sample": len(full),
                "distinct_keys_in_full_sample": len(counts),
            }
        )
        claimed = receipt["prompt_results"][row_index]
        for field, value in replay_prompt_results[-1].items():
            if claimed.get(field) != value:
                errors.append(f"row {row_index} summary mismatch for {field}")

    summary = {
        "prompt_count": len(rows),
        "prefix_success_prompts": sum(
            item["verified_in_prefix"] > 0 for item in replay_prompt_results
        ),
        "multimode_prompts": sum(
            item["distinct_keys_in_full_sample"] >= 2
            for item in replay_prompt_results
        ),
        "verified_completions": sum(
            item["verified_in_full_sample"] for item in replay_prompt_results
        ),
    }
    checks["summary_replays_exactly"] = summary == receipt.get("summary")
    for name, passed in checks.items():
        if not passed:
            errors.append(f"failed check: {name}")
    payload = {
        "schema_version": "pantry-support-action-development-audit-v1",
        "status": "pass_with_provenance_defect" if not errors else "fail",
        "receipt_sha256": _sha256(args.receipt),
        "identity_sha256": _sha256(args.identity),
        "checks": checks,
        "replayed_summary": summary,
        "provenance_defect": (
            "The prose protocol was zero bytes at submission. The held-job "
            "identity and immutable evaluator still fixed seed, model, split, "
            "sampling, criteria, action space, and information boundary."
        ),
        "errors": errors,
    }
    _atomic(args.output, payload)
    print(json.dumps({"status": payload["status"], "summary": summary}))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
