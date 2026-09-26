#!/usr/bin/env python3
"""Score E49T's frozen declared-combo mismatch veto."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import tempfile
from collections import defaultdict
from typing import Any

from oat_drgrpo.math_strategy_canonicalizer import MathStrategyCanonicalizer
from oat_drgrpo.math_strategy_menu import parse_strategy_menu


ROOT = pathlib.Path(__file__).resolve().parents[2]
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e49t_declaration_mismatch_calibration_20260726.md"
)
SCRIPT = pathlib.Path(__file__).resolve()
CANONICALIZER = ROOT / "src/oat_drgrpo/math_strategy_canonicalizer.py"


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def score(artifact: pathlib.Path, endpoint_path: pathlib.Path, output: pathlib.Path) -> None:
    if output.exists():
        raise RuntimeError("fresh mismatch result required")
    identity_path = artifact / "frozen_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    cohort_path = artifact / "cohort.jsonl"
    labels_path = artifact / "private/labels.jsonl"
    if (
        identity["cohort_sha256"] != _sha(cohort_path)
        or identity["private_labels_sha256"] != _sha(labels_path)
        or identity["protocol_sha256"] != _sha(PROTOCOL)
    ):
        raise RuntimeError("mismatch cohort identity failed")
    endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
    expected = {
        "model": "qwen2.5-72b",
        "tensor_parallel_size": 4,
        "max_model_len": 32768,
        "max_num_seqs": 8,
        "enforce_eager": True,
        "structured_output_backend": "guidance",
        "checkpoint_revision": "698703eae6604af048a3d2f509995dc302088217",
    }
    if any(endpoint.get(key) != value for key, value in expected.items()):
        raise RuntimeError("mismatch endpoint identity failed")
    cohort = _read_jsonl(cohort_path)
    private = {row["item_id"]: row for row in _read_jsonl(labels_path)}
    if len(cohort) != 48 or len(private) != 48:
        raise RuntimeError("mismatch cohort size failed")
    canonicalizer = MathStrategyCanonicalizer(
        endpoint=f"http://{endpoint['node']}:{endpoint['port']}/v1",
        model="qwen2.5-72b",
        timeout_seconds=600,
        max_workers=4,
        max_item_chars=4000,
        allow_unstructured_menu_inference=True,
    )
    keys, diagnostics = canonicalizer.canonicalize(
        prompt_token_ids=[[row["group_index"] + 1] for row in cohort],
        prompt_texts=[row["problem"] for row in cohort],
        response_texts=[row["response"] for row in cohort],
        task_reward_positive=[True] * 48,
        active_mask=[True] * 48,
        num_samples=4,
    )
    correct_matched = wrong_matched = mismatch_accepted = open_set = 0
    correct_by_group: dict[str, set[str]] = defaultdict(set)
    decisions = []
    for row, key in zip(cohort, keys, strict=True):
        label = private[row["item_id"]]
        if hashlib.sha256(row["response"].encode()).hexdigest() != label[
            "response_sha256"
        ]:
            raise RuntimeError("mismatch response identity failed")
        menu = parse_strategy_menu(row["problem"])
        assert menu is not None
        allowed = {
            MathStrategyCanonicalizer._menu_strategy_key(
                menu, strategy.strategy_id
            ): strategy.strategy_id
            for strategy in menu.strategies
        }
        assigned = allowed.get(key) if key is not None else None
        open_set += int(key is not None and key not in allowed)
        if label["kind"] == "mismatched_declaration":
            mismatch_accepted += int(key is not None)
        elif key is not None:
            if assigned == label["expected_strategy_id"]:
                correct_matched += 1
                correct_by_group[label["group_id"]].add(assigned)
            else:
                wrong_matched += 1
        decisions.append(
            {
                "item_id": row["item_id"],
                "kind": label["kind"],
                "declared_strategy_id": label["declared_strategy_id"],
                "executed_strategy_id": label["executed_strategy_id"],
                "assigned_strategy_id": assigned,
                "canonical_key": key,
            }
        )
    dual = sum(values == {"S1", "S2"} for values in correct_by_group.values())
    checks = {
        "zero_mismatched_declaration_acceptance": mismatch_accepted == 0,
        "zero_matched_declaration_misassignment": wrong_matched == 0,
        "zero_open_set_keys": open_set == 0,
        "matched_acceptance_at_least_75pct": correct_matched / 24 >= 0.75,
        "dual_route_prompts_at_least_10": dual >= 10,
        "zero_judge_format_failures": diagnostics.judge_format_failure_rows == 0,
    }
    result = {
        "schema": "e49t_declaration_mismatch_result_v1",
        "pass": all(checks.values()),
        "checks": checks,
        "counts": {
            "matched_count": 24,
            "correct_matched": correct_matched,
            "wrong_matched": wrong_matched,
            "mismatched_count": 24,
            "mismatch_accepted": mismatch_accepted,
            "open_set": open_set,
            "dual_route_prompt_count": dual,
        },
        "diagnostics": diagnostics.__dict__,
        "decisions": decisions,
        "identities": {
            "cohort_identity_sha256": _sha(identity_path),
            "endpoint_record_sha256": _sha(endpoint_path),
            "canonicalizer_sha256": _sha(CANONICALIZER),
            "scorer_sha256": _sha(SCRIPT),
        },
    }
    _write(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=pathlib.Path, required=True)
    parser.add_argument("--endpoint-record", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    score(args.artifact_dir.resolve(), args.endpoint_record.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
