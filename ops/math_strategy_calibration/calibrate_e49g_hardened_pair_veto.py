#!/usr/bin/env python3
"""Run and score E49G's conservative four-way Qwen72 pair veto."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


ASSESSMENTS = (
    ("equivalence_reducer", 493101, False),
    ("equivalence_reducer", 493102, True),
    ("theorem_signature_falsifier", 493111, False),
    ("theorem_signature_falsifier", 493112, True),
)
SCHEMA = "e49g_hardened_pair_assessment_v1"


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
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _append_jsonl(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(payload, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _endpoint(record_path: pathlib.Path) -> tuple[str, str]:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    expected = {
        "model": "qwen2.5-72b",
        "node": "node105",
        "tensor_parallel_size": 4,
        "max_model_len": 32768,
        "enforce_eager": True,
        "structured_output_backend": "guidance",
        "checkpoint_revision": "698703eae6604af048a3d2f509995dc302088217",
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise RuntimeError("unexpected E49G Qwen72 endpoint identity")
    host = os.environ.get("E49G_QWEN_HOST_OVERRIDE", "").strip()
    if not host:
        host = str(record["node"])
    return f"http://{host}:{int(record['port'])}/v1", str(record["model"])


def _schema() -> dict[str, Any]:
    required = [
        "route_a_sound_and_self_contained",
        "route_b_sound_and_self_contained",
        "route_a_minimal_core",
        "route_b_minimal_core",
        "same_decisive_core",
        "routine_reduction_exists",
        "routine_reduction_witness",
        "different_labels_or_granularity_only",
        "route_a_exclusive_necessary_fact",
        "route_b_exclusive_necessary_fact",
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
            "route_a_minimal_core": {"type": "string"},
            "route_b_minimal_core": {"type": "string"},
            "same_decisive_core": {"type": "boolean"},
            "routine_reduction_exists": {"type": "boolean"},
            "routine_reduction_witness": {"type": "string"},
            "different_labels_or_granularity_only": {"type": "boolean"},
            "route_a_exclusive_necessary_fact": {"type": "string"},
            "route_b_exclusive_necessary_fact": {"type": "string"},
            "relation": {
                "type": "string",
                "enum": ["equivalent", "distinct", "ambiguous"],
            },
            "rationale": {"type": "string"},
        },
    }


def _bounded_text(record: dict[str, Any], key: str, limit: int) -> bool:
    value = record.get(key)
    return (
        isinstance(value, str)
        and bool(value.strip())
        and len(value) <= limit
    )


def _well_formed(record: Any) -> bool:
    required = set(_schema()["required"])
    return bool(
        isinstance(record, dict)
        and set(record) == required
        and type(record["route_a_sound_and_self_contained"]) is bool
        and type(record["route_b_sound_and_self_contained"]) is bool
        and type(record["same_decisive_core"]) is bool
        and type(record["routine_reduction_exists"]) is bool
        and type(record["different_labels_or_granularity_only"]) is bool
        and record["relation"] in {"equivalent", "distinct", "ambiguous"}
        and all(
            _bounded_text(record, key, limit)
            for key, limit in (
                ("route_a_minimal_core", 1000),
                ("route_b_minimal_core", 1000),
                ("routine_reduction_witness", 1200),
                ("route_a_exclusive_necessary_fact", 800),
                ("route_b_exclusive_necessary_fact", 800),
                ("rationale", 1600),
            )
        )
    )


def _strict_distinct(record: dict[str, Any]) -> bool:
    if not _well_formed(record):
        return False
    left = record["route_a_exclusive_necessary_fact"].strip().casefold()
    right = record["route_b_exclusive_necessary_fact"].strip().casefold()
    return bool(
        record["route_a_sound_and_self_contained"]
        and record["route_b_sound_and_self_contained"]
        and record["same_decisive_core"] is False
        and record["routine_reduction_exists"] is False
        and record["different_labels_or_granularity_only"] is False
        and record["relation"] == "distinct"
        and left != right
        and left not in {"none", "n/a", "not applicable"}
        and right not in {"none", "n/a", "not applicable"}
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
        "equivalence_reducer": (
            "Act as an equivalence reducer. Try as hard as possible to map "
            "both traces to one minimal sufficient derivation."
        ),
        "theorem_signature_falsifier": (
            "Act as a skeptical theorem-signature editor. Try to falsify "
            "every claimed route-exclusive fact and every novelty claim."
        ),
    }[role]
    return f"""Conservatively classify two proposed MATH solution routes.

{role_text}

Equivalent is the default. First erase action IDs, declared kernel labels,
step count, prose style, and redundant checks. Reduce each route to the
smallest sequence of mathematical facts that is sufficient for the answer.
Then explicitly try ordinary substitution, algebraic rearrangement,
representation conversion, reindexing, a coordinate shift, unit conversion,
or deletion of redundant steps to turn one core into the other.

The following are equivalent, not distinct: multiplication principle versus
calling the same tuples a Cartesian product; an inverse-variation ratio
versus solving the same constant product; decimals versus fractions; direct
expansion versus naming the identical formula; a unit-conversion detour;
renaming the same theorem; or adding a geometric picture that establishes
only the same equation. A route with a false, incomplete, or hidden decisive
step cannot support novelty.

Distinct requires all of the following: both routes are sound and
self-contained; each establishes a different route-exclusive fact necessary
for its derivation; neither exclusive fact follows from the other route by
routine algebra or representation; and the difference is not labels,
granularity, or an unnecessary detour. Examples of potentially distinct
cores include an exact theorem shortcut versus independent prime
factorization, direct counting versus complementary counting, and
differentiating a generating function versus shifting/subtracting a series.

If unsure, return ambiguous. Do not reward variety of wording.

PROBLEM:
{pair["problem"]}

REFERENCE ANSWER FOR SOUNDNESS ONLY:
{pair["reference_answer"]}

DISPLAYED ROUTE A:
{json.dumps(left, sort_keys=True)}

DISPLAYED ROUTE B:
{json.dumps(right, sort_keys=True)}
"""


def _post(
    endpoint: str,
    payload: dict[str, Any],
    *,
    timeout: int,
) -> tuple[dict[str, Any], str]:
    request = urllib.request.Request(
        f"{endpoint.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    error: Exception | None = None
    for attempt in range(3):
        try:
            with opener.open(request, timeout=timeout) as response:
                decoded = json.loads(response.read().decode("utf-8"))
            choices = decoded.get("choices") or []
            if not choices:
                raise ValueError("Qwen72 returned no choices")
            content = str(
                (choices[0].get("message") or {}).get("content") or ""
            )
            return decoded, content
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            error = exc
            if attempt < 2:
                time.sleep(2**attempt)
    raise RuntimeError(f"E49G Qwen72 request failed: {error}")


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
                    "You are a conservative mathematical equivalence "
                    "auditor. Return valid JSON only."
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
                "name": "e49g_hardened_pair_veto",
                "strict": True,
                "schema": _schema(),
            },
        },
    }
    response, content = _post(endpoint, payload, timeout=timeout)
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
        raise RuntimeError("E49G packet is not the frozen 29-pair cohort")
    endpoint, model = _endpoint(args.endpoint)
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
                raise RuntimeError(f"E49G cache changed: {path}")
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
        futures = {pool.submit(execute, item): item for item in work}
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
            raise RuntimeError("E49G assessment coverage is incomplete")
        rows.append(
            {
                "schema": "e49g_hardened_pair_decision_v1",
                "pair_id": pair["pair_id"],
                "packet_row_sha256": _canonical_sha256(pair),
                "audits": audits,
                "predicted_distinct": all(
                    audit["distinct_vote"] for audit in audits
                ),
                "complete": all(
                    audit["completed_invalid"] is False for audit in audits
                ),
            }
        )
    decisions = args.output / "pair_decisions.jsonl"
    if decisions.exists():
        raise RuntimeError("E49G final decisions already exist")
    for row in rows:
        _append_jsonl(decisions, row)
    _write_json(
        args.output / "run_summary.json",
        {
            "schema": "e49g_hardened_pair_run_summary_v1",
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
        raise RuntimeError("E49G analysis cohorts do not match")
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
        "schema": "e49g_hardened_pair_calibration_report_v1",
        "pass": all(checks.values()),
        "checks": checks,
        "packet_sha256": _sha256(args.packet),
        "labels_sha256": _sha256(args.labels),
        "private_key_sha256": _sha256(args.private_key),
        "decision_sha256": _sha256(
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
        raise RuntimeError("E49G calibration report already exists")
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
