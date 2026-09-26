#!/usr/bin/env python3
"""Score the frozen E49T route-confusion cohort with unanimous 72B audits."""

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
    "e49t_finite_menu_route_confusion_calibration_20260726.md"
)
SCRIPT = pathlib.Path(__file__).resolve()
CANONICALIZER = ROOT / "src/oat_drgrpo/math_strategy_canonicalizer.py"


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _validate_endpoint(record: dict[str, Any]) -> str:
    expected = {
        "model": "qwen2.5-72b",
        "tensor_parallel_size": 4,
        "max_model_len": 32768,
        "max_num_seqs": 8,
        "enforce_eager": True,
        "structured_output_backend": "guidance",
        "checkpoint_revision": "698703eae6604af048a3d2f509995dc302088217",
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise RuntimeError("unexpected E49T 72B endpoint identity")
    node = record.get("node")
    port = record.get("port")
    if not isinstance(node, str) or not node or not isinstance(port, int):
        raise RuntimeError("E49T endpoint lacks node/port")
    return f"http://{node}:{port}/v1"


def score(
    cohort_dir: pathlib.Path,
    endpoint_record_path: pathlib.Path,
    output: pathlib.Path,
) -> None:
    if output.exists():
        raise RuntimeError(f"fresh E49T result required: {output}")
    identity_path = cohort_dir / "frozen_identity.json"
    cohort_path = cohort_dir / "cohort.jsonl"
    labels_path = cohort_dir / "private/labels.jsonl"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if (
        identity.get("schema")
        != "e49t_route_confusion_frozen_identity_v1"
        or identity.get("cohort_sha256") != _sha256(cohort_path)
        or identity.get("private_labels_sha256") != _sha256(labels_path)
        or identity.get("protocol_sha256") != _sha256(PROTOCOL)
    ):
        raise RuntimeError("E49T frozen cohort identity mismatch")
    cohort = _read_jsonl(cohort_path)
    labels = _read_jsonl(labels_path)
    if len(cohort) != 60 or len(labels) != 60:
        raise RuntimeError("E49T frozen cohort has wrong size")
    private = {row["item_id"]: row for row in labels}
    if set(private) != {row["item_id"] for row in cohort}:
        raise RuntimeError("E49T public/private cohort mismatch")
    for row in cohort:
        label = private[row["item_id"]]
        if hashlib.sha256(row["response"].encode()).hexdigest() != label[
            "response_sha256"
        ]:
            raise RuntimeError("E49T response identity mismatch")

    endpoint_record = json.loads(
        endpoint_record_path.read_text(encoding="utf-8")
    )
    endpoint = _validate_endpoint(endpoint_record)
    canonicalizer = MathStrategyCanonicalizer(
        endpoint=endpoint,
        model="qwen2.5-72b",
        timeout_seconds=600,
        max_workers=4,
        max_item_chars=4000,
        allow_unstructured_menu_inference=True,
    )
    keys, diagnostics = canonicalizer.canonicalize(
        prompt_token_ids=[
            [int(row["group_index"]) + 1] for row in cohort
        ],
        prompt_texts=[str(row["problem"]) for row in cohort],
        response_texts=[str(row["response"]) for row in cohort],
        task_reward_positive=[True] * len(cohort),
        active_mask=[True] * len(cohort),
        num_samples=5,
    )

    decisions = []
    correct_positive = 0
    rejected_positive = 0
    misassigned_positive = 0
    accepted_negative = 0
    open_set_count = 0
    accepted_route_keys: dict[tuple[str, str], set[str]] = defaultdict(set)
    correctly_accepted_strategies: dict[str, set[str]] = defaultdict(set)
    for row, key in zip(cohort, keys, strict=True):
        label = private[row["item_id"]]
        menu = parse_strategy_menu(row["problem"])
        if menu is None:
            raise RuntimeError("E49T cohort menu disappeared")
        allowed = {
            MathStrategyCanonicalizer._menu_strategy_key(
                menu, strategy.strategy_id
            ): strategy.strategy_id
            for strategy in menu.strategies
        }
        assigned_strategy = allowed.get(key) if key is not None else None
        if key is not None and key not in allowed:
            open_set_count += 1
        expected = label["expected_strategy_id"]
        if label["kind"] == "positive":
            if key is None:
                rejected_positive += 1
            elif assigned_strategy == expected:
                correct_positive += 1
                correctly_accepted_strategies[label["group_id"]].add(expected)
            else:
                misassigned_positive += 1
            if key is not None:
                accepted_route_keys[(label["group_id"], expected)].add(key)
        elif key is not None:
            accepted_negative += 1
        decisions.append(
            {
                "item_id": row["item_id"],
                "kind": label["kind"],
                "expected_strategy_id": expected,
                "assigned_strategy_id": assigned_strategy,
                "canonical_key": key,
            }
        )

    duplicate_false_new_count = sum(
        len(route_keys) > 1 for route_keys in accepted_route_keys.values()
    )
    dual_route_prompt_count = sum(
        strategy_ids == {"S1", "S2"}
        for strategy_ids in correctly_accepted_strategies.values()
    )
    positive_count = 48
    acceptance_fraction = correct_positive / positive_count
    checks = {
        "no_open_set_keys": open_set_count == 0,
        "no_false_route_assignments": misassigned_positive == 0,
        "no_duplicate_false_new": duplicate_false_new_count == 0,
        "no_answer_only_acceptance": accepted_negative == 0,
        "positive_acceptance_at_least_75pct": acceptance_fraction >= 0.75,
        "dual_route_prompts_at_least_10": dual_route_prompt_count >= 10,
        "no_judge_format_failures": (
            diagnostics.judge_format_failure_rows == 0
        ),
    }
    payload = {
        "schema": "e49t_route_confusion_calibration_result_v1",
        "pass": all(checks.values()),
        "checks": checks,
        "counts": {
            "group_count": 12,
            "positive_count": positive_count,
            "correct_positive": correct_positive,
            "rejected_positive": rejected_positive,
            "misassigned_positive": misassigned_positive,
            "positive_acceptance_fraction": acceptance_fraction,
            "negative_count": 12,
            "accepted_negative": accepted_negative,
            "open_set_count": open_set_count,
            "duplicate_false_new_count": duplicate_false_new_count,
            "dual_route_prompt_count": dual_route_prompt_count,
        },
        "diagnostics": diagnostics.__dict__,
        "decisions": decisions,
        "identities": {
            "cohort_identity_sha256": _sha256(identity_path),
            "endpoint_record_sha256": _sha256(endpoint_record_path),
            "canonicalizer_sha256": _sha256(CANONICALIZER),
            "scoring_script_sha256": _sha256(SCRIPT),
            "protocol_sha256": _sha256(PROTOCOL),
        },
    }
    _write(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["pass"]:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-dir", type=pathlib.Path, required=True)
    parser.add_argument("--endpoint-record", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    score(
        args.cohort_dir.resolve(),
        args.endpoint_record.resolve(),
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()
