#!/usr/bin/env python3
"""Calibrate finite strategy families with one problem per judge request."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = pathlib.Path(__file__).resolve()
BASE_SCRIPT = (
    ROOT
    / "ops/math_strategy_calibration/"
    "run_e50f3_finite_family_canonicalization.py"
)
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e50f4_pairwise_finite_family_calibration_20260726.md"
)
SEEDS = (500791, 500792)


def _load_base() -> Any:
    spec = importlib.util.spec_from_file_location("e50f4_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load E50F3 finite-family base")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base()


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def _one_call(
    *,
    endpoint: str,
    model: str,
    case: dict[str, Any],
    pass_index: int,
    seed: int,
    timeout: int,
) -> dict[str, Any]:
    routes = list(case["routes"])
    if pass_index == 1:
        routes.reverse()
    route_ids = [row["route_id"] for row in routes]
    prompt = BASE._prompt(routes)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": BASE.SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1024,
        "seed": seed,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "e50f4_pairwise_finite_strategy_family",
                "strict": True,
                "schema": BASE._schema(route_ids),
            },
        },
    }
    response, assessment = BASE._post(
        endpoint, payload, timeout=timeout
    )
    rows = assessment.get("assignments")
    if (
        not isinstance(rows, list)
        or len(rows) != 2
        or {str(row.get("route_id")) for row in rows} != set(route_ids)
    ):
        raise RuntimeError("E50F4 malformed route coverage")
    choice = (response.get("choices") or [{}])[0]
    response_id = str(response.get("id") or "")
    finish_reason = str(choice.get("finish_reason") or "")
    content = str((choice.get("message") or {}).get("content") or "")
    if not response_id or finish_reason != "stop":
        raise RuntimeError("E50F4 nonterminal judge response")
    return {
        "pair_id": case["pair_id"],
        "pass_index": pass_index,
        "seed": seed,
        "assignments": {
            str(row["route_id"]): str(row["family"]) for row in rows
        },
        "raw_assignments": rows,
        "prompt_sha256": _sha256_text(prompt),
        "response_id": response_id,
        "finish_reason": finish_reason,
        "response_content_sha256": _sha256_text(content),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    output = args.output_root.resolve()
    result_path = output / "result.json"
    if result_path.exists():
        raise RuntimeError(f"fresh E50F4 output required: {result_path}")
    for name, path in (
        ("packet", BASE.PACKET),
        ("labels", BASE.LABELS),
        ("endpoint", BASE.ENDPOINT_RECORD),
    ):
        if _sha256(path) != BASE.EXPECTED[name]:
            raise RuntimeError(f"E50F4 frozen {name} drifted")

    packets = {
        str(row["pair_id"]): row for row in BASE._read_jsonl(BASE.PACKET)
    }
    label_payload = json.loads(BASE.LABELS.read_text(encoding="utf-8"))
    cases = []
    for label in label_payload["labels"]:
        if (
            label["route_a_sound_and_self_contained"] is not True
            or label["route_b_sound_and_self_contained"] is not True
        ):
            continue
        pair_id = str(label["pair_id"])
        packet = packets[pair_id]
        cases.append(
            {
                "pair_id": pair_id,
                "expected_relation": (
                    "different"
                    if label["genuinely_distinct_decisive_strategy"] is True
                    else "same"
                ),
                "routes": [
                    {
                        "route_id": f"{pair_id}_A",
                        "problem": str(packet["problem"]),
                        "route": BASE._compact_route(packet["route_a"]),
                    },
                    {
                        "route_id": f"{pair_id}_B",
                        "problem": str(packet["problem"]),
                        "route": BASE._compact_route(packet["route_b"]),
                    },
                ],
            }
        )
    if (
        len(cases) != 18
        or sum(row["expected_relation"] == "different" for row in cases)
        != 12
        or sum(row["expected_relation"] == "same" for row in cases) != 6
    ):
        raise RuntimeError("E50F4 expected 12 distinct and six same pairs")

    endpoint, model = BASE._endpoint(BASE.ENDPOINT_RECORD)
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                _one_call,
                endpoint=endpoint,
                model=model,
                case=case,
                pass_index=pass_index,
                seed=seed,
                timeout=args.timeout,
            )
            for case in cases
            for pass_index, seed in enumerate(SEEDS)
        ]
        for future in as_completed(futures):
            row = future.result()
            records.append(row)
            print(
                json.dumps(
                    {
                        "phase": "family",
                        "pair_id": row["pair_id"],
                        "pass_index": row["pass_index"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    records.sort(key=lambda row: (row["pair_id"], row["pass_index"]))
    records_path = output / "judge_records.jsonl"
    _write_jsonl(records_path, records)
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_pair.setdefault(str(row["pair_id"]), []).append(row)

    pair_results = []
    for case in sorted(cases, key=lambda row: row["pair_id"]):
        pair_records = by_pair[case["pair_id"]]
        if len(pair_records) != 2:
            raise RuntimeError("E50F4 pair did not receive two passes")
        left_id = case["routes"][0]["route_id"]
        right_id = case["routes"][1]["route_id"]
        left = [row["assignments"][left_id] for row in pair_records]
        right = [row["assignments"][right_id] for row in pair_records]
        stable = len(set(left)) == 1 and len(set(right)) == 1
        relation = (
            "different"
            if (
                stable
                and left[0] != "other"
                and right[0] != "other"
                and left[0] != right[0]
            )
            else "same_or_uncertain"
        )
        pair_results.append(
            {
                "pair_id": case["pair_id"],
                "expected_relation": case["expected_relation"],
                "relation": relation,
                "route_a_families": left,
                "route_b_families": right,
                "stable": stable,
                "correct": (
                    relation == "different"
                    if case["expected_relation"] == "different"
                    else relation != "different"
                ),
            }
        )
    false_new = sum(
        row["expected_relation"] == "same" and row["relation"] == "different"
        for row in pair_results
    )
    false_merge = sum(
        row["expected_relation"] == "different"
        and row["relation"] != "different"
        for row in pair_results
    )
    checks = {
        "zero_false_new": false_new == 0,
        "at_most_two_false_merges": false_merge <= 2,
        "all_route_families_stable": all(
            row["stable"] for row in pair_results
        ),
        "all_requests_terminal_and_well_formed": True,
    }
    result = {
        "schema": "e50f4_pairwise_finite_family_calibration_v1",
        "pass": all(checks.values()),
        "checks": checks,
        "counts": {
            "sound_pair_count": 18,
            "distinct_pair_count": 12,
            "same_pair_count": 6,
            "false_new_count": false_new,
            "false_merge_count": false_merge,
            "stable_pair_count": sum(
                row["stable"] for row in pair_results
            ),
        },
        "families": list(BASE.FAMILIES),
        "pair_results": pair_results,
        "identity": {
            "packet_sha256": _sha256(BASE.PACKET),
            "labels_sha256": _sha256(BASE.LABELS),
            "endpoint_record_sha256": _sha256(BASE.ENDPOINT_RECORD),
            "base_script_sha256": _sha256(BASE_SCRIPT),
            "protocol_sha256": _sha256(PROTOCOL),
            "script_sha256": _sha256(SCRIPT),
            "judge_records_sha256": _sha256(records_path),
            "seeds": list(SEEDS),
        },
    }
    _write_json(result_path, result)
    print(result_path)


if __name__ == "__main__":
    main()
