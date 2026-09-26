#!/usr/bin/env python3
"""Calibrate a restricted-enum MathIR executable strategy signature."""

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
    "e49j_e49g_transport",
    BASE_PATH,
)
if BASE_SPEC is None or BASE_SPEC.loader is None:
    raise RuntimeError("cannot import E49G transport")
base = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(base)

ASSESSMENTS = (
    ("mathir_canonical_parser", 493301, False),
    ("mathir_canonical_parser", 493302, True),
    ("adversarial_mathir_parser", 493311, False),
    ("adversarial_mathir_parser", 493312, True),
)
SCHEMA = "e49j_restricted_mathir_assessment_v1"

GENERIC_OPERATORS = frozenset(
    {
        "READ_GIVENS",
        "SUBSTITUTE_VALUES",
        "ARITHMETIC",
        "ALGEBRA_REARRANGE",
        "SOLVE_EQUATION",
        "VERIFY_RESULT",
    }
)
OPERATOR_IDS = (
    *sorted(GENERIC_OPERATORS),
    "ARITHMETIC_MEAN",
    "ARITHMETIC_SEQUENCE_SUM",
    "AM_GM_INEQUALITY",
    "CALCULUS_GLOBAL_EXTREMUM",
    "CENTRAL_ISOSCELES_GEOMETRY",
    "COMBINATION_PRODUCT",
    "CONIC_ASYMPTOTE_FACTOR_EQUATION",
    "CRT_RESIDUE_CLASS",
    "COTANGENT_IDENTITY_DIFFERENTIATION",
    "CUBIC_VOLUME_SCALING",
    "DEGREE_MAX_EXPONENT",
    "DIFFERENCE_OF_SQUARES",
    "DIRECT_INTERVAL_DYNAMIC_PROGRAMMING",
    "DIRECT_RATE_DEFINITION",
    "DIVISOR_GCD_ENUMERATION",
    "EUCLIDEAN_ALGORITHM",
    "EVENT_UNION_IDENTITY",
    "EXPANDED_PYTHAGOREAN",
    "EXPONENTIAL_GROWTH",
    "FACTORIAL_NORMALIZED_RECURRENCE",
    "FINITE_CASE_ENUMERATION",
    "FRACTION_MAGNITUDE_COMPARE",
    "GCD_LCM_PRODUCT_THEOREM",
    "GEOMETRIC_SERIES_DIFFERENTIATION",
    "HYPERBOLA_AXIS_ASYMPTOTE_GEOMETRY",
    "INDEPENDENT_CHOICE_PRODUCT",
    "INTEGER_CONVEXITY_PARITY",
    "INTEGER_FACTOR_LOCALIZATION",
    "INTERIOR_PRODUCT_COUNT",
    "INVERSE_VARIATION_INVARIANT",
    "LITERAL_RECURRENCE_SIMULATION",
    "MATRIX_FAST_POWER",
    "NUMERIC_ORDER_BOUND",
    "ORTHOCENTER_SUPPLEMENT",
    "PLANE_CROSS_PRODUCT",
    "POLYGON_EXTERIOR_TURNING",
    "POLYNOMIAL_FACTOR_DIVISIBILITY",
    "POLYNOMIAL_QUADRATIC_FORMULA",
    "POLYNOMIAL_QUOTIENT_RING_POWER",
    "POLYNOMIAL_ROOT_FACTOR_EVALUATION",
    "POLYNOMIAL_TRANSLATION_EXPANSION",
    "POLYNOMIAL_VALUE_THEOREM",
    "PRIME_EXPONENT_ACCUMULATION",
    "PRIME_FACTOR_GCD_LCM",
    "RADIX_GROUP_MAP",
    "RADIX_POSITIONAL_EXPANSION",
    "RADIX_REPEATED_DIVISION",
    "RECIPROCAL_TRIG_SUM_IDENTITY",
    "REMAINDER_THEOREM",
    "SERIES_SHIFT_SUBTRACTION",
    "SIDE_LENGTH_ALTITUDE_AREA",
    "SIGN_INVARIANT_BOUND",
    "SYMBOLIC_BASE_FACTOR_IDENTITY",
    "TOTAL_MINUS_BOUNDARY_COUNT",
    "TRIANGLE_SIMILARITY_AREA",
    "AFFINE_COORDINATE_AREA",
    "UNIT_COMPLEX_ALGEBRA",
    "VECTOR_CROSS_PRODUCT_AREA",
    "VECTOR_EQUILATERAL_GEOMETRY",
    "PISANO_PERIOD",
    "ORDERED_ENUMERATION",
    "SECTOR_TO_CONE_GEOMETRY",
)


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
        "route_a_signature_complete",
        "route_b_signature_complete",
        "route_a_mathir_signature",
        "route_b_mathir_signature",
        "rationale",
    ]
    signature = {
        "type": "array",
        "minItems": 1,
        "maxItems": 8,
        "items": {
            "type": "string",
            "enum": list(OPERATOR_IDS),
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": {
            "route_a_sound_and_self_contained": {"type": "boolean"},
            "route_b_sound_and_self_contained": {"type": "boolean"},
            "route_a_signature_complete": {"type": "boolean"},
            "route_b_signature_complete": {"type": "boolean"},
            "route_a_mathir_signature": signature,
            "route_b_mathir_signature": signature,
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
        "route_a_signature_complete",
        "route_b_signature_complete",
    ):
        if type(record[key]) is not bool:
            return False
    for key in ("route_a_mathir_signature", "route_b_mathir_signature"):
        values = record[key]
        if (
            not isinstance(values, list)
            or not 1 <= len(values) <= 8
            or any(value not in OPERATOR_IDS for value in values)
        ):
            return False
    return bool(
        isinstance(record["rationale"], str)
        and 0 < len(record["rationale"]) <= 1600
    )


def _canonical_signature(values: list[str]) -> tuple[str, ...]:
    return tuple(value for value in values if value not in GENERIC_OPERATORS)


def _distinct_vote(record: dict[str, Any]) -> bool:
    if not _well_formed(record):
        return False
    left = _canonical_signature(record["route_a_mathir_signature"])
    right = _canonical_signature(record["route_b_mathir_signature"])
    return bool(
        record["route_a_sound_and_self_contained"]
        and record["route_b_sound_and_self_contained"]
        and record["route_a_signature_complete"]
        and record["route_b_signature_complete"]
        and left
        and right
        and left != right
    )


def _ontology_text() -> str:
    return ", ".join(OPERATOR_IDS)


def _prompt(
    pair: dict[str, Any],
    *,
    role: str,
    swapped: bool,
) -> str:
    left = pair["route_b"] if swapped else pair["route_a"]
    right = pair["route_a"] if swapped else pair["route_b"]
    role_text = {
        "mathir_canonical_parser": (
            "Parse both routes into the smallest complete sequence of "
            "canonical executable operators."
        ),
        "adversarial_mathir_parser": (
            "Try to falsify soundness and try to assign the same canonical "
            "operator whenever two routes merely rename one computation."
        ),
    }[role]
    return f"""Canonicalize two MATH routes into a restricted MathIR.

{role_text}

Ignore action IDs, kernel labels, prose, redundant verification, and
unnecessary detours. Preserve every necessary decisive operation actually
executed. Use generic IDs only for bookkeeping around a decisive ID.

Mandatory aliases:
- ratio and named-constant inverse variation -> INVERSE_VARIATION_INVARIANT;
- multiplication principle and Cartesian-product cardinality ->
  INDEPENDENT_CHOICE_PRODUCT;
- two-event inclusion-exclusion and complement-of-neither after independence
  -> EVENT_UNION_IDENTITY;
- decimal and fractional comparison -> FRACTION_MAGNITUDE_COMPARE;
- direct and unit-converted cubic scaling -> CUBIC_VOLUME_SCALING;
- direct and renamed polynomial substitutions through the same root
  factorization -> POLYNOMIAL_ROOT_FACTOR_EVALUATION.
- decimal approximation and neighboring-square bounds for the same ceiling
  decision -> NUMERIC_ORDER_BOUND;
- expanded and exponent-combined versions of the same prime factorization ->
  PRIME_EXPONENT_ACCUMULATION;
- a claimed generating-function route that only executes binomial
  coefficients -> COMBINATION_PRODUCT;
- identical displacement-vector cross products for a plane ->
  PLANE_CROSS_PRODUCT;
- geometric-sequence language and the identical exponential update ->
  EXPONENTIAL_GROWTH;
- divisor listing phrased as congruences and divisor listing phrased via a
  gcd -> DIVISOR_GCD_ENUMERATION;
- combining like coefficients before reading degree and directly reading
  the maximal exponent -> DEGREE_MAX_EXPONENT.

Mandatory distinctions when literally executed:
- direct binary/octal digit grouping -> RADIX_GROUP_MAP, whereas conversion
  through decimal followed by repeated division uses
  RADIX_POSITIONAL_EXPANSION then RADIX_REPEATED_DIVISION;
- gcd-lcm product shortcut -> GCD_LCM_PRODUCT_THEOREM, whereas independently
  building gcd and lcm from prime exponents -> PRIME_FACTOR_GCD_LCM;
- differentiating a geometric series -> GEOMETRIC_SERIES_DIFFERENTIATION,
  whereas shift/subtract -> SERIES_SHIFT_SUBTRACTION;
- direct strict-coordinate count -> INTERIOR_PRODUCT_COUNT, whereas total
  points minus a separately counted boundary -> TOTAL_MINUS_BOUNDARY_COUNT.

A false, incomplete, or hidden decisive step makes that route sound=false or
signature_complete=false. Do not repair it. Return IDs only from this enum:
{_ontology_text()}

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
                    "You are a restricted MathIR parser and proof auditor. "
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
        "max_tokens": 2048,
        "seed": seed,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "e49j_restricted_mathir_signature",
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
            "canonical_route_a": [],
            "canonical_route_b": [],
            "distinct_vote": False,
        }
    valid = bool(
        common["finish_reason"] == "stop" and _well_formed(assessment)
    )
    left = (
        list(_canonical_signature(assessment["route_a_mathir_signature"]))
        if valid
        else []
    )
    right = (
        list(_canonical_signature(assessment["route_b_mathir_signature"]))
        if valid
        else []
    )
    return {
        **common,
        "assessment": assessment,
        "completed_invalid": not valid,
        "canonical_route_a": left,
        "canonical_route_b": right,
        "distinct_vote": bool(valid and _distinct_vote(assessment)),
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
        raise RuntimeError("E49J packet is not the frozen 29-pair cohort")
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
                raise RuntimeError(f"E49J cache changed: {path}")
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
            raise RuntimeError("E49J assessment coverage is incomplete")
        rows.append(
            {
                "schema": "e49j_restricted_mathir_decision_v1",
                "pair_id": pair["pair_id"],
                "packet_row_sha256": _canonical_sha256(pair),
                "audits": audits,
                "distinct_vote_count": sum(
                    audit["distinct_vote"] for audit in audits
                ),
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
        raise RuntimeError("E49J final decisions already exist")
    _write_jsonl(decisions, rows)
    _write_json(
        args.output / "run_summary.json",
        {
            "schema": "e49j_restricted_mathir_run_summary_v1",
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
        raise RuntimeError("E49J analysis cohorts do not match")
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
    tp = sum(
        row["expected_distinct"] and row["predicted_distinct"]
        for row in rows
    )
    fp = sum(
        not row["expected_distinct"] and row["predicted_distinct"]
        for row in rows
    )
    fn = sum(
        row["expected_distinct"] and not row["predicted_distinct"]
        for row in rows
    )
    tn = sum(
        not row["expected_distinct"] and not row["predicted_distinct"]
        for row in rows
    )
    controls = [
        row for row in rows if row["kind"] == "blinded_equivalent_control"
    ]
    checks = {
        "all_requests_complete": all(row["complete"] for row in rows),
        "false_new_exactly_zero": fp == 0,
        "all_hidden_equivalent_controls_rejected": (
            len(controls) == 3
            and all(not row["predicted_distinct"] for row in controls)
        ),
        "at_least_three_of_four_true_distinct_recovered": (
            tp >= 3
            and sum(row["expected_distinct"] for row in rows) == 4
        ),
    }
    report = {
        "schema": "e49j_restricted_mathir_calibration_report_v1",
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
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
        },
        "false_new_rate": fp
        / max(1, sum(not row["expected_distinct"] for row in rows)),
        "distinct_recall": tp
        / max(1, sum(row["expected_distinct"] for row in rows)),
        "distinct_precision": tp
        / max(1, sum(row["predicted_distinct"] for row in rows)),
        "rows": rows,
    }
    path = args.output / "calibration_report.json"
    if path.exists():
        raise RuntimeError("E49J calibration report already exists")
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
