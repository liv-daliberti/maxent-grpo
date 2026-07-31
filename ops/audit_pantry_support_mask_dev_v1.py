#!/usr/bin/env python3
"""Independently replay-audit the Pantry six-bit support-mask receipt."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value) -> str:
    return hashlib.sha256(json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")).hexdigest()


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
    parser.add_argument("--admission-audit", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("fresh Pantry mask audit output is required")
    sys.path.insert(0, str((args.source_root / "src").resolve()))
    from datasets import load_from_disk
    from oat_drgrpo.pantry_plan import validate_pantry_plan
    from oat_drgrpo.pantry_support_action import (
        PANTRY_SUPPORT_MASK_INVALID,
        decode_pantry_support_mask,
    )

    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    identity = json.loads(args.identity.read_text(encoding="utf-8"))
    dataset_identity = json.loads(
        (args.data_root / "identity.json").read_text(encoding="utf-8")
    )
    rows = load_from_disk(str(args.data_root / "dev"))["multi_answer"].to_list()
    expected_masks = {"".join(bits) for bits in product("01", repeat=6)}
    evaluator = args.execution_root / "evaluate_pantry_support_mask_viability_dev.py"
    slurm = args.execution_root / "evaluate_pantry_support_mask_viability_dev.slurm"
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
        "evaluator_hash_matches": _sha256(evaluator) == identity.get("evaluator_sha256"),
        "slurm_hash_matches": _sha256(slurm) == identity.get("slurm_sha256"),
        "protocol_hash_matches": (
            _sha256(args.protocol) == receipt.get("protocol_sha256")
            == identity.get("protocol_sha256")
        ),
        "admission_hash_matches": (
            _sha256(args.admission_audit) == receipt.get("admission_audit_sha256")
            == identity.get("admission_audit_sha256")
        ),
        "dataset_identity_matches": (
            receipt.get("dataset_identity_sha256") == _canonical(dataset_identity)
        ),
        "development_rows_match": receipt.get("data_split_sha256") == _canonical(rows),
        "attempt_count_is_4096": len(receipt.get("attempts", [])) == 4096,
        "criteria_are_frozen": receipt.get("criteria") == {
            "minimum_multimode_prompts": 16,
            "minimum_prefix_success_prompts": 32,
        },
        "sampling_is_frozen": receipt.get("sampling") == {
            "action_adapter": "pantry_support_mask_projected_v1",
            "assistant_prefix": "",
            "bit_token_ids": [15, 16],
            "fixed_action_space": "six binary inclusion decisions",
            "max_model_len": 2048,
            "max_tokens": 6,
            "prefix_count": 16,
            "prompt_repair": "none",
            "prompt_template": "qwen_pantry_support_mask",
            "sample_count": 64,
            "seed": 76102,
            "temperature": 1.0,
            "top_p": 1.0,
        },
        "information_boundary_declared": receipt.get("information_boundary") == {
            "action_space_contains_all_64_binary_masks": True,
            "certified_feasible_supports_in_context": False,
            "development_only": True,
            "evaluation_prompts_loaded": False,
            "quantity_projection_uses_only_prompt_local_constraints": True,
            "reference_allocation_in_context": False,
            "valid_support_width_is_not_masked": True,
        },
    }
    antecedents = {
        "v2_receipt_sha256": "var/artifacts/pantry_plan_05b_viability_v2.json",
        "v3_receipt_sha256": "var/artifacts/pantry_plan_05b_viability_v3.json",
        "support_action_receipt_sha256": "var/artifacts/pantry_support_action_dev_v1.json",
        "support_action_audit_sha256": "var/artifacts/pantry_support_action_dev_v1_audit.json",
        "interactive_receipt_sha256": "var/artifacts/pantry_plan_interactive_05b_viability_v1_r1.json",
        "interactive_runtime_protocol_sha256": "paper/preregistration/pantry_plan_interactive_05b_viability_v1_runtime_r1_20260729.md",
    }
    checks["antecedent_hashes_match"] = all(
        _sha256(args.repo_root / relative) == identity.get(field)
        for field, relative in antecedents.items()
    )

    attempts_by_row = {index: [] for index in range(len(rows))}
    for attempt in receipt.get("attempts", []):
        row_index = int(attempt["row_index"])
        if row_index not in attempts_by_row:
            errors.append(f"out-of-range row index {row_index}")
            continue
        attempts_by_row[row_index].append(attempt)

    replay_prompt_results = []
    for row_index, row in enumerate(rows):
        spec = json.loads(row["answer"])
        attempts = sorted(
            attempts_by_row[row_index], key=lambda item: int(item["sample_index"])
        )
        if [int(item["sample_index"]) for item in attempts] != list(range(1, 65)):
            errors.append(f"row {row_index} sample indices differ from 1..64")
        keys = []
        cache = {}
        for attempt in attempts:
            text = str(attempt["text"])
            if text not in expected_masks:
                errors.append(f"row {row_index} emitted text outside six-bit space")
            if int(attempt.get("token_count", -1)) != 6:
                errors.append(f"row {row_index} recorded non-six-token action")
            if text not in cache:
                allocation = decode_pantry_support_mask(text, spec)
                validation = (
                    None if allocation == PANTRY_SUPPORT_MASK_INVALID
                    else validate_pantry_plan(allocation, spec)
                )
                cache[text] = validation.canonical_key if validation else None
            key = cache[text]
            keys.append(key)
            if key != attempt.get("canonical_key") or bool(key) != bool(attempt.get("verified")):
                errors.append(f"row {row_index} sample {attempt['sample_index']} replay mismatch")
        prefix = [key for key in keys[:16] if key]
        full = [key for key in keys if key]
        counts = Counter(full)
        replayed = {
            "verified_in_prefix": len(prefix),
            "verified_in_full_sample": len(full),
            "distinct_keys_in_full_sample": len(counts),
        }
        replay_prompt_results.append(replayed)
        claimed = receipt["prompt_results"][row_index]
        for field, value in replayed.items():
            if claimed.get(field) != value:
                errors.append(f"row {row_index} summary mismatch for {field}")

    summary = {
        "prompt_count": len(rows),
        "prefix_success_prompts": sum(item["verified_in_prefix"] > 0 for item in replay_prompt_results),
        "multimode_prompts": sum(item["distinct_keys_in_full_sample"] >= 2 for item in replay_prompt_results),
        "verified_completions": sum(item["verified_in_full_sample"] for item in replay_prompt_results),
    }
    checks["summary_replays_exactly"] = summary == receipt.get("summary")
    checks["frozen_decision_is_pass"] = (
        receipt.get("status") == "pass"
        and receipt.get("decision") == "development_interface_signal_pass"
        and summary["prefix_success_prompts"] >= 32
        and summary["multimode_prompts"] >= 16
    )
    for name, passed in checks.items():
        if not passed:
            errors.append(f"failed check: {name}")
    payload = {
        "schema_version": "pantry-support-mask-development-audit-v1",
        "status": "pass" if not errors else "fail",
        "receipt_sha256": _sha256(args.receipt),
        "identity_sha256": _sha256(args.identity),
        "checks": checks,
        "replayed_summary": summary,
        "replayed_prompt_results": replay_prompt_results,
        "errors": errors,
    }
    _atomic(args.output, payload)
    print(json.dumps({"status": payload["status"], "summary": summary}))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
