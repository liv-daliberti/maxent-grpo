#!/usr/bin/env python3
"""Live regressions for E47S's reduced-set novelty veto."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
E47 = ROOT / "var/artifacts/e47_math_strategy_calibration_v1"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "ops/math_strategy_calibration"))

from e47_calibration import _validate
from e47m_semantic_regressions import (
    _canonicalize,
    _read_jsonl,
    _write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    endpoint = json.loads(args.endpoint.read_text(encoding="utf-8"))
    problems = {
        row["problem_id"]: row for row in _read_jsonl(E47 / "problems.jsonl")
    }
    policy = {
        row["sample_id"]: row
        for row in _read_jsonl(E47 / "validated_policy.blinded.jsonl")
    }
    injections = {
        row["sample_id"]: row
        for row in _read_jsonl(E47 / "injections.blinded.jsonl")
    }
    all_rows = {**injections, **policy}
    injection_keys: dict[str, dict[str, str]] = {}
    for row in _read_jsonl(E47 / "private/injection_key.jsonl"):
        injection_keys.setdefault(row["problem_id"], {})[
            row["injection_kind"]
        ] = row["sample_id"]

    records: list[dict[str, Any]] = []

    gcd_anchor = injections[injection_keys["p015"]["anchor"]]["text"]
    gcd_invalid = [
        policy["s_09ea047c9f33e571"]["text"],
        policy["s_af55e947400d8999"]["text"],
    ]
    gcd_keys = _canonicalize(
        endpoint=endpoint,
        problem=problems["p015"]["problem"],
        responses=[gcd_anchor, *gcd_invalid],
        label="invalid_derivations_rejected",
        records=records,
    )

    clock_pair_ids = [
        "s_eb9f508a29bbb6f4",
        "s_94bef21377f3ed3f",
    ]
    clock_keys = _canonicalize(
        endpoint=endpoint,
        problem=problems["p039"]["problem"],
        responses=[policy[sample_id]["text"] for sample_id in clock_pair_ids],
        label="routine_equivalence_merged",
        records=records,
    )

    # These are the exact blinded E47R contexts in which two direct
    # comparisons that passed above were nevertheless split. They were frozen
    # as E47S regressions only after E47R's labels had been committed.
    clock_context_ids = [
        "s_aa1491707d962237",
        "s_655db96a95c75010",
        "s_eb9f508a29bbb6f4",
        "s_c73cc860c263d375",
        "s_fac2991cda7b2e3b",
        "s_c33f20b2305a29f9",
        "s_94bef21377f3ed3f",
    ]
    clock_context_keys = _canonicalize(
        endpoint=endpoint,
        problem=problems["p039"]["problem"],
        responses=[
            all_rows[sample_id]["text"] for sample_id in clock_context_ids
        ],
        label="contextual_clock_equivalence_merged",
        records=records,
    )
    clock_left = clock_context_ids.index(clock_pair_ids[0])
    clock_right = clock_context_ids.index(clock_pair_ids[1])

    median_pair_ids = [
        "s_65fab8513825f4fb",
        "s_0ca9d720211c5c5a",
    ]
    median_context_ids = [
        "s_a2e25624a4ef6366",
        "s_05d4a468430f837e",
        "s_49256d2f4c0b9b63",
        "s_1e1aa4f644c59f4b",
        "s_78c08a9863d45fcf",
        "s_35a3621ef7b781aa",
        "s_8e34cb532b7dcd6a",
        "s_21cfa2d67656130e",
        "s_65fab8513825f4fb",
        "s_bece79164d84f508",
        "s_defe13daf8f42b8f",
        "s_c0683af605085564",
        "s_6ce1a71794054977",
        "s_f3943c31e1e2d14e",
        "s_8066242cd2352734",
        "s_0ca9d720211c5c5a",
    ]
    median_context_keys = _canonicalize(
        endpoint=endpoint,
        problem=problems["p009"]["problem"],
        responses=[
            all_rows[sample_id]["text"] for sample_id in median_context_ids
        ],
        label="contextual_median_equivalence_merged",
        records=records,
    )
    median_left = median_context_ids.index(median_pair_ids[0])
    median_right = median_context_ids.index(median_pair_ids[1])

    distinct_problem = (
        "When the graph of $y = 2x^2 - x + 7$ is shifted four "
        "units to the right, we obtain the graph of "
        "$y = ax^2 + bx + c$. Find $a + b + c$."
    )
    coefficient_expansion = r"""
After shifting four units right, substitute \(x-4\):
\[
y=2(x-4)^2-(x-4)+7=2x^2-17x+43.
\]
Thus \(a+b+c=2-17+43=\boxed{28}\).
""".strip()
    point_evaluation = r"""
For the shifted polynomial \(g(x)=ax^2+bx+c\), the requested
quantity is \(g(1)=a+b+c\). A four-unit right shift means
\(g(1)=f(1-4)=f(-3)\), where \(f(x)=2x^2-x+7\). Therefore
\[
g(1)=2(-3)^2-(-3)+7=18+3+7=\boxed{28}.
\]
""".strip()
    for response in (coefficient_expansion, point_evaluation):
        _, route_reward = _validate(response, "28")
        if route_reward != 1.0:
            raise RuntimeError("different-route proof failed validation")
    distinct_keys = _canonicalize(
        endpoint=endpoint,
        problem=distinct_problem,
        responses=[coefficient_expansion, point_evaluation],
        label="different_proofs_separated",
        records=records,
    )

    checks = {
        "valid_anchor_admitted": gcd_keys[0] is not None,
        "invalid_derivations_rejected": gcd_keys[1:] == [None, None],
        "routine_equivalence_merged": (
            clock_keys[0] is not None and clock_keys[0] == clock_keys[1]
        ),
        "contextual_clock_equivalence_merged": (
            clock_context_keys[clock_left] is not None
            and clock_context_keys[clock_left]
            == clock_context_keys[clock_right]
        ),
        "contextual_median_false_new_prevented": not (
            median_context_keys[median_left] is not None
            and median_context_keys[median_right] is not None
            and median_context_keys[median_left]
            != median_context_keys[median_right]
        ),
        "different_proofs_separated": (
            distinct_keys[0] is not None
            and distinct_keys[1] is not None
            and distinct_keys[0] != distinct_keys[1]
        ),
    }
    report = {
        "schema": "e47s_reduced_veto_regressions_v1",
        "checks": checks,
        "pass": all(checks.values()),
        "outcome_keys": {
            "invalid_derivations_rejected": gcd_keys,
            "routine_equivalence_merged": clock_keys,
            "contextual_clock_equivalence_merged": clock_context_keys,
            "contextual_median_false_new_prevented": median_context_keys,
            "different_proofs_separated": distinct_keys,
        },
        "judge_records": records,
    }
    _write_json(args.output, report)
    print(json.dumps({**checks, "pass": report["pass"]}, indent=2))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
