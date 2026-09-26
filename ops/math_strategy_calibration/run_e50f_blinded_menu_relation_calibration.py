#!/usr/bin/env python3
"""Calibrate the menu-level relation decision on frozen blinded labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import random
import sys
import tempfile
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = pathlib.Path(__file__).resolve()
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e50f_blinded_menu_relation_calibration_20260726.md"
)
PACKET = (
    ROOT
    / "var/artifacts/e49r_combined_manual_audit_v1/"
    "manual_audit_packet.jsonl"
)
LABELS = (
    ROOT
    / "var/artifacts/e49r_combined_manual_audit_v1/"
    "manual_audit_labels.json"
)
ENDPOINT_RECORD = (
    ROOT / "var/artifacts/e49t_qwen72_node302_v1/qwen72_endpoint.json"
)
HELPER = (
    ROOT
    / "ops/math_strategy_calibration/"
    "run_e49w_bottom_up_route_calibration.py"
)
EXPECTED = {
    "packet": "603a497f44286380e073fb183ee73926f26f7bcc616e259421837fa29d09f2ca",
    "labels": "b7a82a2f3b57888413376407136283b02a8a88469ddecad3535f8a67360762fa",
    "endpoint": "964ce3bda1d2cac7dddf2065451fd2b456433478356df9d493e98f9e23c66ac3",
}
SEEDS = (500761, 500762)
BATCH_SIZE = 13
SYSTEM = (
    "You are a conservative mathematical strategy-boundary auditor. "
    "Return only valid JSON matching the supplied schema."
)

sys.path.insert(0, str(ROOT / "ops/math_strategy_calibration"))
from run_e49w_bottom_up_route_calibration import _endpoint, _post  # noqa: E402


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _compact_route(route: dict[str, Any]) -> dict[str, Any]:
    return {
        "plan": str(route["plan"]),
        "action_combo": str(route["action_combo"]),
        "actions": [
            {
                "action_id": str(action["action_id"]),
                "operation": str(action["operation"]),
            }
            for action in route["actions"]
        ],
    }


def _schema(case_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["assessments"],
        "properties": {
            "assessments": {
                "type": "array",
                "minItems": len(case_ids),
                "maxItems": len(case_ids),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "case_id",
                        "relation",
                        "route_a_decisive_operation",
                        "route_b_decisive_operation",
                        "brief_check",
                    ],
                    "properties": {
                        "case_id": {"type": "string", "enum": case_ids},
                        "relation": {
                            "type": "string",
                            "enum": ["same", "different", "ambiguous"],
                        },
                        "route_a_decisive_operation": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 300,
                        },
                        "route_b_decisive_operation": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 300,
                        },
                        "brief_check": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 500,
                        },
                    },
                },
            }
        },
    }


def _render(cases: list[dict[str, Any]]) -> str:
    rendered = []
    for case in cases:
        rendered.append(
            "CASE "
            + case["pair_id"]
            + "\nPROBLEM:\n"
            + case["problem"]
            + "\nROUTE A:\n"
            + json.dumps(case["route_a"], sort_keys=True)
            + "\nROUTE B:\n"
            + json.dumps(case["route_b"], sort_keys=True)
        )
    return "\n\n".join(rendered)


def _prompt(cases: list[dict[str, Any]]) -> str:
    return """Classify each pair of finite mathematical routes by its
ESSENTIAL DECISIVE STRATEGY. Route validity is audited separately; here,
classify the mathematical route actually written.

Return SAME when both routes use the same decisive identity, theorem,
invariant, construction, counted object, substitution, or algorithm and
differ only in wording, notation, action IDs, step order, routine algebra,
arithmetic detail, expanded versus factored rendering of the same
calculation, unit conversion, or a redundant check.

Return DIFFERENT when they use genuinely different central mathematical
operations or proof constructions. A shared problem, final answer, final
equality, or final arithmetic step is never evidence of sameness. Direct
versus complementary counting, coefficient expansion versus strategic
function evaluation, synthetic versus coordinate geometry, inequality versus
calculus, closed form versus dynamic programming, and number-theoretic
construction versus exhaustive search are different when actually executed.

First name each route's decisive operation. If your own comparison names
different central operations, relation must be DIFFERENT unless brief_check
explains concretely why those operations are merely routine
reparameterizations of one calculation. Return AMBIGUOUS only when the
written routes do not permit a reliable decision. Do not use compactness,
efficiency, correctness, or stylistic quality as the relation.

Do not omit, duplicate, or invent case IDs.

""" + _render(cases)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    output = args.output_root.resolve()
    result_path = output / "result.json"
    if result_path.exists():
        raise RuntimeError(f"fresh E50F output required: {result_path}")
    for name, path in (
        ("packet", PACKET),
        ("labels", LABELS),
        ("endpoint", ENDPOINT_RECORD),
    ):
        if _sha256(path) != EXPECTED[name]:
            raise RuntimeError(f"E50F frozen {name} drifted")

    packets = _read_jsonl(PACKET)
    label_payload = json.loads(LABELS.read_text(encoding="utf-8"))
    labels = {
        str(row["pair_id"]): (
            "different"
            if row["genuinely_distinct_decisive_strategy"] is True
            else "same"
        )
        for row in label_payload["labels"]
    }
    if (
        len(packets) != 26
        or set(labels) != {str(row["pair_id"]) for row in packets}
        or sum(value == "different" for value in labels.values()) != 16
        or sum(value == "same" for value in labels.values()) != 10
    ):
        raise RuntimeError("E50F expected 16 distinct and 10 same pairs")
    cases = [
        {
            "pair_id": str(row["pair_id"]),
            "problem": str(row["problem"]),
            "route_a": _compact_route(row["route_a"]),
            "route_b": _compact_route(row["route_b"]),
        }
        for row in packets
    ]

    endpoint, model = _endpoint(ENDPOINT_RECORD)
    records = []
    by_pass: list[dict[str, str]] = []
    for pass_index, seed in enumerate(SEEDS):
        order = list(range(len(cases)))
        random.Random(seed).shuffle(order)
        ordered = []
        for position, case_index in enumerate(order):
            case = json.loads(json.dumps(cases[case_index]))
            if (position + pass_index) % 2:
                case["route_a"], case["route_b"] = (
                    case["route_b"],
                    case["route_a"],
                )
            ordered.append(case)
        decisions: dict[str, str] = {}
        for batch_index in range(0, len(ordered), BATCH_SIZE):
            batch = ordered[batch_index : batch_index + BATCH_SIZE]
            case_ids = [row["pair_id"] for row in batch]
            prompt = _prompt(batch)
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 4096,
                "seed": seed,
                "stream": False,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "e50f_blinded_route_relation",
                        "strict": True,
                        "schema": _schema(case_ids),
                    },
                },
            }
            response, assessment = _post(
                endpoint, payload, timeout=args.timeout
            )
            rows = assessment.get("assessments")
            if (
                not isinstance(rows, list)
                or len(rows) != len(case_ids)
                or {str(row.get("case_id")) for row in rows} != set(case_ids)
            ):
                raise RuntimeError("E50F malformed assessment coverage")
            response_id = str(response.get("id") or "")
            finish_reason = str(
                ((response.get("choices") or [{}])[0]).get("finish_reason")
                or ""
            )
            if not response_id or finish_reason != "stop":
                raise RuntimeError("E50F nonterminal judge response")
            for row in rows:
                pair_id = str(row["case_id"])
                if pair_id in decisions:
                    raise RuntimeError("E50F duplicate relation decision")
                decisions[pair_id] = str(row["relation"])
            records.append(
                {
                    "pass_index": pass_index,
                    "seed": seed,
                    "batch_index": batch_index // BATCH_SIZE,
                    "case_ids": case_ids,
                    "prompt_sha256": _sha256_text(prompt),
                    "response_id": response_id,
                    "finish_reason": finish_reason,
                    "response_content_sha256": _sha256_text(
                        str(
                            ((response.get("choices") or [{}])[0].get(
                                "message"
                            )
                            or {}).get("content")
                            or ""
                        )
                    ),
                    "assessments": rows,
                }
            )
            print(
                json.dumps(
                    {
                        "phase": "audit",
                        "pass_index": pass_index,
                        "batch_index": batch_index // BATCH_SIZE,
                        "case_count": len(batch),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if set(decisions) != set(labels):
            raise RuntimeError("E50F pass did not cover every pair")
        by_pass.append(decisions)

    pair_results = []
    for pair_id in sorted(labels):
        decisions = [row[pair_id] for row in by_pass]
        pair_results.append(
            {
                "pair_id": pair_id,
                "expected_relation": labels[pair_id],
                "decisions": decisions,
                "unanimous": len(set(decisions)) == 1,
                "correct": all(value == labels[pair_id] for value in decisions),
            }
        )
    false_new_by_pass = [
        sum(
            decisions[pair_id] == "different"
            for pair_id, label in labels.items()
            if label == "same"
        )
        for decisions in by_pass
    ]
    false_merge_by_pass = [
        sum(
            decisions[pair_id] != "different"
            for pair_id, label in labels.items()
            if label == "different"
        )
        for decisions in by_pass
    ]
    checks = {
        "zero_false_new_each_pass": max(false_new_by_pass) == 0,
        "at_most_two_false_merges_each_pass": max(false_merge_by_pass) <= 2,
        "all_pairs_unanimous": all(row["unanimous"] for row in pair_results),
        "all_requests_terminal_and_well_formed": True,
    }
    records_path = output / "judge_records.jsonl"
    _write_jsonl(records_path, records)
    result = {
        "schema": "e50f_blinded_menu_relation_calibration_v1",
        "pass": all(checks.values()),
        "checks": checks,
        "counts": {
            "pair_count": 26,
            "distinct_pair_count": 16,
            "same_pair_count": 10,
            "false_new_by_pass": false_new_by_pass,
            "false_merge_by_pass": false_merge_by_pass,
            "unanimous_pair_count": sum(
                row["unanimous"] for row in pair_results
            ),
        },
        "pair_results": pair_results,
        "identity": {
            "packet_sha256": _sha256(PACKET),
            "labels_sha256": _sha256(LABELS),
            "endpoint_record_sha256": _sha256(ENDPOINT_RECORD),
            "protocol_sha256": _sha256(PROTOCOL),
            "script_sha256": _sha256(SCRIPT),
            "helper_sha256": _sha256(HELPER),
            "judge_records_sha256": _sha256(records_path),
            "seeds": list(SEEDS),
        },
    }
    _write_json(result_path, result)
    print(result_path)


if __name__ == "__main__":
    main()
