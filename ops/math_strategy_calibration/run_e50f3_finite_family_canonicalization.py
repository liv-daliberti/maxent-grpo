#!/usr/bin/env python3
"""Calibrate a finite 72B strategy-family canonicalizer."""

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
    "e50f3_finite_family_canonicalization_20260726.md"
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
SEEDS = (500781, 500782)
FAMILIES = (
    "polynomial_expansion",
    "strategic_value_or_invariant",
    "quadratic_or_root_formula",
    "vieta_or_coefficient_relations",
    "symbolic_equation_manipulation",
    "factorization_or_divisor_structure",
    "euclidean_algorithm",
    "prime_factorization",
    "crt_or_congruence_construction",
    "modular_or_residue_analysis",
    "calculus_or_derivative",
    "inequality_or_extremal_bound",
    "finite_candidate_search",
    "dynamic_programming_or_recurrence",
    "closed_form_probability_or_count",
    "direct_combinatorial_count",
    "incidence_or_double_counting",
    "complementary_counting",
    "inclusion_exclusion_or_direct_union",
    "coordinate_or_vector_geometry",
    "synthetic_geometry",
    "trigonometric_geometry",
    "representation_or_numeric_comparison",
    "other",
)
SYSTEM = (
    "You are a conservative canonicalizer of mathematical solution "
    "strategies. Return only valid JSON matching the supplied schema."
)

sys.path.insert(0, str(ROOT / "ops/math_strategy_calibration"))
from run_e49w_bottom_up_route_calibration import (  # noqa: E402
    _endpoint,
    _post,
)


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


def _schema(route_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["assignments"],
        "properties": {
            "assignments": {
                "type": "array",
                "minItems": len(route_ids),
                "maxItems": len(route_ids),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "route_id",
                        "family",
                        "decisive_operation",
                    ],
                    "properties": {
                        "route_id": {"type": "string", "enum": route_ids},
                        "family": {"type": "string", "enum": list(FAMILIES)},
                        "decisive_operation": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 300,
                        },
                    },
                },
            }
        },
    }


def _prompt(routes: list[dict[str, Any]]) -> str:
    rendered = "\n\n".join(
        "ROUTE "
        + row["route_id"]
        + "\nPROBLEM:\n"
        + row["problem"]
        + "\nDECLARED ROUTE:\n"
        + json.dumps(row["route"], sort_keys=True)
        for row in routes
    )
    return """Assign every validated finite route to exactly one canonical
strategy family from the supplied enum. Choose the route's decisive
mathematical engine, not its topic, notation, answer, surface ordering, or
final arithmetic.

Canonical boundary rules:
- Tree, grid, table, nested-loop, tuple-filter, multiplication-principle, and
  falling-factorial renderings of the same Cartesian product or injective
  assignment count are direct_combinatorial_count. Merely drawing the product
  does not create a new strategy.
- finite_candidate_search means generating candidates and testing a separate
  mathematical constraint; it does not mean displaying a direct product
  count.
- Decimal, common-denominator, or equivalent exact-value renderings used only
  to compare the same quantities are representation_or_numeric_comparison.
- A closed formula and dynamic programming are different families.
- Complementary counting and a direct union/inclusion-exclusion calculation
  are different families when those are the decisive probability arguments.
- Vieta/coefficient relations and explicitly solving roots with a root
  formula are different families.
- Polynomial coefficient expansion and avoiding expansion by strategic value
  evaluation are different families.
- Calculus and a sharp inequality argument are different families.
- Euclidean algorithm, prime-factor intersection, CRT construction, and
  exhaustive constraint search are different families.
- Coordinate/vector and synthetic geometric constructions are different
  families.
- Use other conservatively when no enum member clearly applies. Do not invent
  a family or use compactness/correctness as a family distinction.

Do not omit, duplicate, or invent route IDs.

""" + rendered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    output = args.output_root.resolve()
    result_path = output / "result.json"
    if result_path.exists():
        raise RuntimeError(f"fresh E50F3 output required: {result_path}")
    for name, path in (
        ("packet", PACKET),
        ("labels", LABELS),
        ("endpoint", ENDPOINT_RECORD),
    ):
        if _sha256(path) != EXPECTED[name]:
            raise RuntimeError(f"E50F3 frozen {name} drifted")

    packets = {
        str(row["pair_id"]): row for row in _read_jsonl(PACKET)
    }
    label_payload = json.loads(LABELS.read_text(encoding="utf-8"))
    sound_labels = [
        row
        for row in label_payload["labels"]
        if row["route_a_sound_and_self_contained"] is True
        and row["route_b_sound_and_self_contained"] is True
    ]
    cases = []
    routes = []
    for label in sound_labels:
        pair_id = str(label["pair_id"])
        packet = packets[pair_id]
        route_ids = (f"{pair_id}_A", f"{pair_id}_B")
        cases.append(
            {
                "pair_id": pair_id,
                "route_ids": list(route_ids),
                "expected_relation": (
                    "different"
                    if label["genuinely_distinct_decisive_strategy"] is True
                    else "same"
                ),
            }
        )
        routes.extend(
            [
                {
                    "route_id": route_ids[0],
                    "problem": str(packet["problem"]),
                    "route": _compact_route(packet["route_a"]),
                },
                {
                    "route_id": route_ids[1],
                    "problem": str(packet["problem"]),
                    "route": _compact_route(packet["route_b"]),
                },
            ]
        )
    if (
        len(cases) != 18
        or sum(row["expected_relation"] == "different" for row in cases)
        != 12
        or sum(row["expected_relation"] == "same" for row in cases) != 6
        or len(routes) != 36
    ):
        raise RuntimeError("E50F3 expected 18 sound pairs and 36 routes")

    endpoint, model = _endpoint(ENDPOINT_RECORD)
    records = []
    assignments_by_pass = []
    for pass_index, seed in enumerate(SEEDS):
        ordered = list(routes)
        random.Random(seed).shuffle(ordered)
        route_ids = [row["route_id"] for row in ordered]
        prompt = _prompt(ordered)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 6144,
            "seed": seed,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "e50f3_finite_strategy_family",
                    "strict": True,
                    "schema": _schema(route_ids),
                },
            },
        }
        response, assessment = _post(
            endpoint, payload, timeout=args.timeout
        )
        rows = assessment.get("assignments")
        if (
            not isinstance(rows, list)
            or len(rows) != len(route_ids)
            or {str(row.get("route_id")) for row in rows} != set(route_ids)
        ):
            raise RuntimeError("E50F3 malformed route coverage")
        response_id = str(response.get("id") or "")
        choice = (response.get("choices") or [{}])[0]
        finish_reason = str(choice.get("finish_reason") or "")
        content = str((choice.get("message") or {}).get("content") or "")
        if not response_id or finish_reason != "stop":
            raise RuntimeError("E50F3 nonterminal judge response")
        assignments = {
            str(row["route_id"]): str(row["family"]) for row in rows
        }
        assignments_by_pass.append(assignments)
        records.append(
            {
                "pass_index": pass_index,
                "seed": seed,
                "route_ids": route_ids,
                "prompt_sha256": _sha256_text(prompt),
                "response_id": response_id,
                "finish_reason": finish_reason,
                "response_content_sha256": _sha256_text(content),
                "assignments": rows,
            }
        )
        print(
            json.dumps(
                {
                    "phase": "family",
                    "pass_index": pass_index,
                    "route_count": len(rows),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    route_stable = {
        route["route_id"]: (
            assignments_by_pass[0][route["route_id"]]
            == assignments_by_pass[1][route["route_id"]]
        )
        for route in routes
    }
    pair_results = []
    for case in sorted(cases, key=lambda row: row["pair_id"]):
        left, right = case["route_ids"]
        left_families = [
            assignments[left] for assignments in assignments_by_pass
        ]
        right_families = [
            assignments[right] for assignments in assignments_by_pass
        ]
        stable = route_stable[left] and route_stable[right]
        relation = (
            "different"
            if (
                stable
                and left_families[0] != "other"
                and right_families[0] != "other"
                and left_families[0] != right_families[0]
            )
            else "same_or_uncertain"
        )
        pair_results.append(
            {
                "pair_id": case["pair_id"],
                "expected_relation": case["expected_relation"],
                "relation": relation,
                "route_a_families": left_families,
                "route_b_families": right_families,
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
        "all_route_families_stable": all(route_stable.values()),
        "all_requests_terminal_and_well_formed": True,
    }
    records_path = output / "judge_records.jsonl"
    _write_jsonl(records_path, records)
    result = {
        "schema": "e50f3_finite_family_canonicalization_v1",
        "pass": all(checks.values()),
        "checks": checks,
        "counts": {
            "sound_pair_count": 18,
            "distinct_pair_count": 12,
            "same_pair_count": 6,
            "false_new_count": false_new,
            "false_merge_count": false_merge,
            "stable_route_count": sum(route_stable.values()),
            "route_count": len(routes),
        },
        "families": list(FAMILIES),
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
