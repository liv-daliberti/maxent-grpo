#!/usr/bin/env python3
"""Calibrate the E49I executable algorithm-signature pair veto."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
BASE_PATH = HERE / "calibrate_e49g_hardened_pair_veto.py"
BASE_SPEC = importlib.util.spec_from_file_location(
    "e49i_e49g_transport",
    BASE_PATH,
)
if BASE_SPEC is None or BASE_SPEC.loader is None:
    raise RuntimeError("cannot import E49G transport")
base = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(base)


ASSESSMENTS = (
    ("algorithm_signature_editor", 493201, False),
    ("algorithm_signature_editor", 493202, True),
    ("false_new_prosecutor", 493211, False),
    ("false_new_prosecutor", 493212, True),
)
SCHEMA = "e49i_algorithm_signature_assessment_v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: pathlib.Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


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
                    json.dumps(
                        row,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _schema() -> dict[str, Any]:
    required = [
        "route_a_sound_and_self_contained",
        "route_b_sound_and_self_contained",
        "route_a_operator_signature",
        "route_b_operator_signature",
        "same_primitive_signature",
        "routine_signature_translation_exists",
        "routine_signature_translation_witness",
        "different_labels_or_granularity_only",
        "route_a_exclusive_operator",
        "route_b_exclusive_operator",
        "relation",
        "rationale",
    ]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": {
            "route_a_sound_and_self_contained": {"type": "boolean"},
            "route_b_sound_and_self_contained": {"type": "boolean"},
            "route_a_operator_signature": {
                "type": "array",
                "minItems": 1,
                "maxItems": 12,
                "items": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 300,
                },
            },
            "route_b_operator_signature": {
                "type": "array",
                "minItems": 1,
                "maxItems": 12,
                "items": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 300,
                },
            },
            "same_primitive_signature": {"type": "boolean"},
            "routine_signature_translation_exists": {"type": "boolean"},
            "routine_signature_translation_witness": {
                "type": "string",
                "maxLength": 1200,
            },
            "different_labels_or_granularity_only": {"type": "boolean"},
            "route_a_exclusive_operator": {
                "type": "string",
                "maxLength": 800,
            },
            "route_b_exclusive_operator": {
                "type": "string",
                "maxLength": 800,
            },
            "relation": {
                "type": "string",
                "enum": [
                    "equivalent",
                    "distinct",
                    "unsound",
                    "ambiguous",
                ],
            },
            "rationale": {
                "type": "string",
                "minLength": 1,
                "maxLength": 1600,
            },
        },
    }


def _well_formed(record: Any) -> bool:
    required = set(_schema()["required"])
    if not isinstance(record, dict) or set(record) != required:
        return False
    for key in (
        "route_a_sound_and_self_contained",
        "route_b_sound_and_self_contained",
        "same_primitive_signature",
        "routine_signature_translation_exists",
        "different_labels_or_granularity_only",
    ):
        if type(record[key]) is not bool:
            return False
    for key in ("route_a_operator_signature", "route_b_operator_signature"):
        values = record[key]
        if (
            not isinstance(values, list)
            or not 1 <= len(values) <= 12
            or any(
                not isinstance(value, str)
                or not value.strip()
                or len(value) > 300
                for value in values
            )
        ):
            return False
    for key, limit, allow_empty in (
        ("routine_signature_translation_witness", 1200, True),
        ("route_a_exclusive_operator", 800, True),
        ("route_b_exclusive_operator", 800, True),
        ("rationale", 1600, False),
    ):
        value = record[key]
        if (
            not isinstance(value, str)
            or len(value) > limit
            or (not allow_empty and not value.strip())
        ):
            return False
    return record["relation"] in {
        "equivalent",
        "distinct",
        "unsound",
        "ambiguous",
    }


def _strict_distinct(record: dict[str, Any]) -> bool:
    if not _well_formed(record):
        return False
    left = record["route_a_exclusive_operator"].strip().casefold()
    right = record["route_b_exclusive_operator"].strip().casefold()
    return bool(
        record["route_a_sound_and_self_contained"]
        and record["route_b_sound_and_self_contained"]
        and record["same_primitive_signature"] is False
        and record["routine_signature_translation_exists"] is False
        and record["different_labels_or_granularity_only"] is False
        and record["relation"] == "distinct"
        and left
        and right
        and left != right
    )


def _prompt(
    pair: dict[str, Any],
    *,
    role: str,
    swapped: bool,
) -> str:
    left = pair["route_b"] if swapped else pair["route_a"]
    right = pair["route_a"] if swapped else pair["route_b"]
    role_text = {
        "algorithm_signature_editor": (
            "Act as an algorithm-signature editor. Canonicalize what each "
            "route literally executes, not merely the final identity both "
            "routes prove."
        ),
        "false_new_prosecutor": (
            "Act as a false-new prosecutor. Try to align the executable "
            "signatures, but do not erase a necessary route-exclusive "
            "algorithm just because both algorithms have the same answer."
        ),
    }[role]
    return f"""Classify two proposed MATH solution routes conservatively.

{role_text}

First verify both traces. Then erase prose, action IDs, declared kernel
labels, arithmetic bookkeeping, redundant checks, and unnecessary detours.
For each route return the ordered minimal EXECUTED OPERATOR SIGNATURE: named
theorem actually invoked, constructed object, invariant, search/counting
space, representation-specific algorithm, and optimality/globality proof.

Equivalent is the default. It includes ordinary algebraic rearrangement,
notation, action granularity, unit conversion, relabeling, reordered
bookkeeping, decimals versus fractions, and a picture that establishes only
the same equation. Calling independent selections a Cartesian product is the
same multiplication principle. Solving one inverse-variation invariant by a
ratio or by naming its constant is the same signature. An unsound,
incomplete, or hidden step cannot establish novelty.

Distinct means both routes are sound and each executes a necessary primitive
operator absent from the other after this cleanup. Sharing a final identity,
answer, or broad topic does NOT by itself make signatures equivalent.
Important boundary examples:

- direct radix grouping/mapping versus conversion to a decimal intermediate
  followed by repeated division are distinct algorithms;
- invoking gcd(a,b)lcm(a,b)=ab as a shortcut versus independently prime
  factorizing both inputs and constructing gcd and lcm is distinct;
- differentiating a generating function versus shifting/subtracting the
  target series is distinct;
- directly counting a strict interior product space versus counting a larger
  space and subtracting a separately characterized boundary is distinct.

Those examples do not license label games: the exclusive operator must
actually appear in the action executions and be necessary to that route.
Write a concrete ordinary translation if one signature reduces to the other.
If unsure, return ambiguous, which is not a novelty vote.

PROBLEM:
{pair["problem"]}

REFERENCE ANSWER FOR SOUNDNESS ONLY:
{pair["reference_answer"]}

DISPLAYED ROUTE A:
{json.dumps(left, sort_keys=True)}

DISPLAYED ROUTE B:
{json.dumps(right, sort_keys=True)}
"""


def _request_one(
    *,
    pair: dict[str, Any],
    endpoint: str,
    model: str,
    role: str,
    seed: int,
    swapped: bool,
    timeout: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a conservative executable-algorithm auditor. "
                    "Return valid JSON only."
                ),
            },
            {
                "role": "user",
                "content": _prompt(pair, role=role, swapped=swapped),
            },
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 3072,
        "seed": seed,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "e49i_algorithm_signature_veto",
                "strict": True,
                "schema": _schema(),
            },
        },
    }
    response, content = base._post(endpoint, payload, timeout=timeout)
    common = {
        "schema": SCHEMA,
        "pair_id": pair["pair_id"],
        "packet_row_sha256": _canonical_sha256(pair),
        "role": role,
        "seed": seed,
        "display_swapped": swapped,
        "response_id": response.get("id"),
        "finish_reason": (
            ((response.get("choices") or [{}])[0]).get("finish_reason")
        ),
        "content_sha256": _sha256_bytes(content.encode("utf-8")),
    }
    try:
        assessment = json.loads(content)
    except json.JSONDecodeError as exc:
        return {
            **common,
            "assessment": None,
            "completed_invalid": True,
            "error": str(exc),
            "distinct_vote": False,
        }
    valid = bool(
        common["finish_reason"] == "stop" and _well_formed(assessment)
    )
    return {
        **common,
        "assessment": assessment,
        "completed_invalid": not valid,
        "distinct_vote": bool(valid and _strict_distinct(assessment)),
    }


def _cache_path(
    output: pathlib.Path,
    pair_id: str,
    role: str,
    seed: int,
) -> pathlib.Path:
    return output / "request_cache" / pair_id / f"{role}-{seed}.json"


def run(args: argparse.Namespace) -> None:
    packet = _read_jsonl(args.packet)
    if len(packet) != 29 or len({row["pair_id"] for row in packet}) != 29:
        raise RuntimeError("E49I packet is not the frozen 29-pair cohort")
    endpoint, model = base._endpoint(args.endpoint)
    work = [
        (pair, role, seed, swapped)
        for pair in packet
        for role, seed, swapped in ASSESSMENTS
    ]

    def execute(item):
        pair, role, seed, swapped = item
        path = _cache_path(args.output, pair["pair_id"], role, seed)
        if path.is_file():
            record = json.loads(path.read_text(encoding="utf-8"))
            if (
                record.get("schema") != SCHEMA
                or record.get("packet_row_sha256")
                != _canonical_sha256(pair)
                or record.get("role") != role
                or record.get("seed") != seed
                or record.get("display_swapped") != swapped
            ):
                raise RuntimeError(f"E49I cache changed: {path}")
            return record
        record = _request_one(
            pair=pair,
            endpoint=endpoint,
            model=model,
            role=role,
            seed=seed,
            swapped=swapped,
            timeout=args.timeout,
        )
        _write_json(path, record)
        return record

    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute, item) for item in work]
        for future in as_completed(futures):
            completed.append(future.result())
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for record in completed:
        by_pair.setdefault(record["pair_id"], []).append(record)
    rows = []
    for pair in packet:
        audits = sorted(
            by_pair[pair["pair_id"]],
            key=lambda row: row["seed"],
        )
        if len(audits) != 4:
            raise RuntimeError("E49I assessment coverage is incomplete")
        distinct_votes = sum(audit["distinct_vote"] for audit in audits)
        rows.append(
            {
                "schema": "e49i_algorithm_signature_decision_v1",
                "pair_id": pair["pair_id"],
                "packet_row_sha256": _canonical_sha256(pair),
                "audits": audits,
                "distinct_vote_count": distinct_votes,
                "predicted_distinct": distinct_votes >= 3,
                "complete": all(
                    audit["completed_invalid"] is False for audit in audits
                ),
            }
        )
    decisions = args.output / "pair_decisions.jsonl"
    if decisions.exists():
        raise RuntimeError("E49I final decisions already exist")
    _write_jsonl(decisions, rows)
    _write_json(
        args.output / "run_summary.json",
        {
            "schema": "e49i_algorithm_signature_run_summary_v1",
            "packet_sha256": _sha256(args.packet),
            "endpoint_sha256": _sha256(args.endpoint),
            "pair_count": len(rows),
            "request_count": len(completed),
            "complete_pair_count": sum(row["complete"] for row in rows),
            "predicted_distinct_count": sum(
                row["predicted_distinct"] for row in rows
            ),
            "decisions_sha256": _sha256(decisions),
        },
    )


def analyze(args: argparse.Namespace) -> None:
    labels_payload = json.loads(args.labels.read_text(encoding="utf-8"))
    labels = {
        row["pair_id"]: row for row in labels_payload["labels"]
    }
    decisions = {
        row["pair_id"]: row
        for row in _read_jsonl(args.output / "pair_decisions.jsonl")
    }
    private = {
        row["pair_id"]: row for row in _read_jsonl(args.private_key)
    }
    if not set(labels) == set(decisions) == set(private):
        raise RuntimeError("E49I analysis cohorts do not match")
    rows = []
    for pair_id in sorted(labels):
        manual = labels[pair_id]
        expected = bool(
            manual["route_a_sound_and_self_contained"]
            and manual["route_b_sound_and_self_contained"]
            and manual["genuinely_distinct_decisive_strategy"]
        )
        predicted = bool(decisions[pair_id]["predicted_distinct"])
        rows.append(
            {
                "pair_id": pair_id,
                "kind": private[pair_id]["kind"],
                "expected_distinct": expected,
                "predicted_distinct": predicted,
                "distinct_vote_count": decisions[pair_id][
                    "distinct_vote_count"
                ],
                "complete": decisions[pair_id]["complete"],
                "correct": expected == predicted,
            }
        )
    true_positive = sum(
        row["expected_distinct"] and row["predicted_distinct"]
        for row in rows
    )
    false_positive = sum(
        not row["expected_distinct"] and row["predicted_distinct"]
        for row in rows
    )
    false_negative = sum(
        row["expected_distinct"] and not row["predicted_distinct"]
        for row in rows
    )
    true_negative = sum(
        not row["expected_distinct"] and not row["predicted_distinct"]
        for row in rows
    )
    controls = [
        row for row in rows if row["kind"] == "blinded_equivalent_control"
    ]
    checks = {
        "all_requests_complete": all(row["complete"] for row in rows),
        "false_new_exactly_zero": false_positive == 0,
        "all_hidden_equivalent_controls_rejected": (
            len(controls) == 3
            and all(not row["predicted_distinct"] for row in controls)
        ),
        "at_least_three_of_four_true_distinct_recovered": (
            true_positive >= 3
            and sum(row["expected_distinct"] for row in rows) == 4
        ),
    }
    report = {
        "schema": "e49i_algorithm_signature_calibration_report_v1",
        "pass": all(checks.values()),
        "checks": checks,
        "packet_sha256": _sha256(args.packet),
        "labels_sha256": _sha256(args.labels),
        "private_key_sha256": _sha256(args.private_key),
        "decisions_sha256": _sha256(
            args.output / "pair_decisions.jsonl"
        ),
        "counts": {
            "pair_count": len(rows),
            "manual_distinct": sum(
                row["expected_distinct"] for row in rows
            ),
            "predicted_distinct": sum(
                row["predicted_distinct"] for row in rows
            ),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
        },
        "false_new_rate": false_positive
        / max(1, sum(not row["expected_distinct"] for row in rows)),
        "distinct_recall": true_positive
        / max(1, sum(row["expected_distinct"] for row in rows)),
        "distinct_precision": true_positive
        / max(1, sum(row["predicted_distinct"] for row in rows)),
        "rows": rows,
    }
    path = args.output / "calibration_report.json"
    if path.exists():
        raise RuntimeError("E49I calibration report already exists")
    _write_json(path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "analyze"))
    parser.add_argument("--packet", type=pathlib.Path, required=True)
    parser.add_argument("--endpoint", type=pathlib.Path)
    parser.add_argument("--labels", type=pathlib.Path)
    parser.add_argument("--private-key", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if args.action == "run":
        if args.endpoint is None:
            raise RuntimeError("--endpoint is required for run")
        run(args)
    else:
        if args.labels is None or args.private_key is None:
            raise RuntimeError(
                "--labels and --private-key are required for analyze"
            )
        analyze(args)


if __name__ == "__main__":
    main()
