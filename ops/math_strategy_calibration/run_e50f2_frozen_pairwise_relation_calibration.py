#!/usr/bin/env python3
"""Test the exact frozen E47W pairwise veto on sound manual route pairs."""

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
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e50f2_frozen_pairwise_relation_calibration_20260726.md"
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
PAIRWISE_SOURCE = (
    ROOT
    / "var/artifacts/source_snapshots/"
    "e49b_math_strategy_ae06b1023ade504c7b8a07e901874e5b563b9e8fad618e5fefe519337458ebfc/"
    "src/oat_drgrpo/math_strategy_canonicalizer.py"
)
HELPER = (
    ROOT
    / "ops/math_strategy_calibration/"
    "run_e49w_bottom_up_route_calibration.py"
)
E47W_ANALYSIS = (
    ROOT
    / "var/artifacts/e47w_pairwise_math_strategy_calibration_v1/"
    "analysis.json"
)
EXPECTED = {
    "packet": "603a497f44286380e073fb183ee73926f26f7bcc616e259421837fa29d09f2ca",
    "labels": "b7a82a2f3b57888413376407136283b02a8a88469ddecad3535f8a67360762fa",
    "endpoint": "964ce3bda1d2cac7dddf2065451fd2b456433478356df9d493e98f9e23c66ac3",
    "pairwise_source": "91a1a8fdc3b29fa49b1089154f5955ec2d9e759cc85e94b35bc7d101d81bf988",
    "e47w_analysis": "3ad930355d4ef3cc30f153035db3cfdca5bf8290566b00fe3383111c4c7455d6",
}
SEEDS = (500771, 500772)

sys.path.insert(0, str(ROOT / "ops/math_strategy_calibration"))
from run_e49w_bottom_up_route_calibration import _endpoint  # noqa: E402


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


def _load_pairwise_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "e50f2_frozen_pairwise", PAIRWISE_SOURCE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen E47W pairwise source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _route_text(route: dict[str, Any]) -> str:
    compact = {
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
    return json.dumps(compact, sort_keys=True, separators=(",", ":"))


def _one_call(
    *,
    module: Any,
    endpoint: str,
    model: str,
    case: dict[str, Any],
    pass_index: int,
    seed: int,
    timeout: int,
) -> dict[str, Any]:
    judge = module.MathStrategyCanonicalizer(
        endpoint=endpoint,
        model=model,
        timeout_seconds=timeout,
        max_workers=1,
        max_item_chars=4000,
    )
    captured: dict[str, Any] = {}
    original_post = judge._post

    def recording_post(payload: dict[str, Any]) -> dict[str, Any]:
        response = original_post(payload)
        captured["payload"] = payload
        captured["response"] = response
        return response

    judge._post = recording_post
    pair_id = "PAIR_0000"
    relations = judge._judge_relations(
        problem=case["problem"],
        base_items=[
            ("ROUTE_A", case["route_a"]),
            ("ROUTE_B", case["route_b"]),
        ],
        pairs=[(pair_id, "ROUTE_A", "ROUTE_B")],
        seed=seed,
    )
    response = captured["response"]
    payload = captured["payload"]
    choice = (response.get("choices") or [{}])[0]
    finish_reason = str(choice.get("finish_reason") or "")
    response_id = str(response.get("id") or "")
    content = str((choice.get("message") or {}).get("content") or "")
    if finish_reason != "stop" or not response_id:
        raise RuntimeError("E50F2 relation response was nonterminal")
    return {
        "pair_id": case["pair_id"],
        "pass_index": pass_index,
        "seed": seed,
        "relation": relations[pair_id],
        "prompt_sha256": _sha256_text(
            json.dumps(payload["messages"], sort_keys=True)
        ),
        "response_id": response_id,
        "finish_reason": finish_reason,
        "response_content_sha256": _sha256_text(content),
        "response_content": content,
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
        raise RuntimeError(f"fresh E50F2 output required: {result_path}")
    for name, path in (
        ("packet", PACKET),
        ("labels", LABELS),
        ("endpoint", ENDPOINT_RECORD),
        ("pairwise_source", PAIRWISE_SOURCE),
        ("e47w_analysis", E47W_ANALYSIS),
    ):
        if _sha256(path) != EXPECTED[name]:
            raise RuntimeError(f"E50F2 frozen {name} drifted")
    e47w = json.loads(E47W_ANALYSIS.read_text(encoding="utf-8"))
    required_checks = (
        "exact_duplicate_false_new_zero",
        "manual_same_false_new_at_most_0_05",
        "semantic_regressions_pass",
        "overall_false_new_at_most_0_05",
    )
    if e47w.get("gate_status") != "pass" or not all(
        e47w["gate_checks"].get(name) is True for name in required_checks
    ):
        raise RuntimeError("E50F2 requires the passing E47W calibration")

    packet_by_id = {
        str(row["pair_id"]): row for row in _read_jsonl(PACKET)
    }
    labels_payload = json.loads(LABELS.read_text(encoding="utf-8"))
    sound_labels = [
        row
        for row in labels_payload["labels"]
        if row["route_a_sound_and_self_contained"] is True
        and row["route_b_sound_and_self_contained"] is True
    ]
    cases = []
    for label in sound_labels:
        pair_id = str(label["pair_id"])
        packet = packet_by_id[pair_id]
        cases.append(
            {
                "pair_id": pair_id,
                "problem": str(packet["problem"]),
                "route_a": _route_text(packet["route_a"]),
                "route_b": _route_text(packet["route_b"]),
                "expected_relation": (
                    "different"
                    if label["genuinely_distinct_decisive_strategy"] is True
                    else "same"
                ),
            }
        )
    if (
        len(cases) != 18
        or sum(row["expected_relation"] == "different" for row in cases)
        != 12
        or sum(row["expected_relation"] == "same" for row in cases) != 6
    ):
        raise RuntimeError("E50F2 expected 12 distinct and six same pairs")

    module = _load_pairwise_module()
    endpoint, model = _endpoint(ENDPOINT_RECORD)
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                _one_call,
                module=module,
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
            record = future.result()
            records.append(record)
            print(
                json.dumps(
                    {
                        "phase": "relation",
                        "pair_id": record["pair_id"],
                        "pass_index": record["pass_index"],
                        "relation": record["relation"],
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
        decisions = [
            row["relation"] for row in by_pair[case["pair_id"]]
        ]
        if len(decisions) != 2:
            raise RuntimeError("E50F2 pair did not receive two decisions")
        pair_results.append(
            {
                "pair_id": case["pair_id"],
                "expected_relation": case["expected_relation"],
                "decisions": decisions,
                "unanimous": len(set(decisions)) == 1,
                "correct": all(
                    value == case["expected_relation"] for value in decisions
                ),
            }
        )
    false_new_by_pass = [
        sum(
            by_pair[case["pair_id"]][pass_index]["relation"] == "different"
            for case in cases
            if case["expected_relation"] == "same"
        )
        for pass_index in range(2)
    ]
    false_merge_by_pass = [
        sum(
            by_pair[case["pair_id"]][pass_index]["relation"] != "different"
            for case in cases
            if case["expected_relation"] == "different"
        )
        for pass_index in range(2)
    ]
    checks = {
        "zero_false_new_each_pass": max(false_new_by_pass) == 0,
        "at_most_two_false_merges_each_pass": max(false_merge_by_pass) <= 2,
        "all_pairs_unanimous": all(row["unanimous"] for row in pair_results),
        "all_requests_terminal_and_well_formed": True,
    }
    result = {
        "schema": "e50f2_frozen_pairwise_relation_calibration_v1",
        "pass": all(checks.values()),
        "checks": checks,
        "counts": {
            "sound_pair_count": 18,
            "distinct_pair_count": 12,
            "same_pair_count": 6,
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
            "pairwise_source_sha256": _sha256(PAIRWISE_SOURCE),
            "e47w_analysis_sha256": _sha256(E47W_ANALYSIS),
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
