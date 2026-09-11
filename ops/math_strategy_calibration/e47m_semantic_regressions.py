#!/usr/bin/env python3
"""Frozen live semantic regressions for E47M's two-stage canonicalizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
E47 = ROOT / "var/artifacts/e47_math_strategy_calibration_v1"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "ops/math_strategy_calibration"))

from e47_calibration import _post_slurm_json, _validate
from oat_drgrpo.math_strategy_canonicalizer import MathStrategyCanonicalizer


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    Path(temporary).replace(path)


def _prompt_tokens(label: str) -> list[int]:
    return list(hashlib.sha256(label.encode("utf-8")).digest())


def _canonicalize(
    *,
    endpoint: dict[str, Any],
    problem: str,
    responses: list[str],
    label: str,
    records: list[dict[str, Any]],
) -> list[str | None]:
    records_lock = threading.Lock()

    def transport(payload: dict[str, Any]) -> dict[str, Any]:
        response = _post_slurm_json(endpoint, payload, 900)
        with records_lock:
            records.append(
                {
                    "case": label,
                    "seed": payload["seed"],
                    "messages": payload["messages"],
                    "response": response,
                }
            )
        return response

    canonicalizer = MathStrategyCanonicalizer(
        endpoint="slurm-relay://e47m/v1",
        timeout_seconds=900,
        max_workers=1,
        max_item_chars=4000,
        transport=transport,
    )
    keys, _ = canonicalizer.canonicalize(
        prompt_token_ids=[_prompt_tokens(label)] * len(responses),
        prompt_texts=[problem] * len(responses),
        response_texts=responses,
        task_reward_positive=[True] * len(responses),
        active_mask=[True] * len(responses),
        num_samples=len(responses),
    )
    return keys


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
    injection_keys: dict[str, dict[str, str]] = {}
    for row in _read_jsonl(E47 / "private/injection_key.jsonl"):
        injection_keys.setdefault(row["problem_id"], {})[
            row["injection_kind"]
        ] = row["sample_id"]

    records: list[dict[str, Any]] = []

    gcd_problem = problems["p015"]
    gcd_anchor = injections[injection_keys["p015"]["anchor"]]["text"]
    gcd_invalid = [
        policy["s_09ea047c9f33e571"]["text"],
        policy["s_af55e947400d8999"]["text"],
    ]
    gcd_keys = _canonicalize(
        endpoint=endpoint,
        problem=gcd_problem["problem"],
        responses=[gcd_anchor, *gcd_invalid],
        label="invalid_derivations_rejected",
        records=records,
    )

    clock_problem = problems["p039"]
    clock_responses = [
        policy["s_eb9f508a29bbb6f4"]["text"],
        policy["s_94bef21377f3ed3f"]["text"],
    ]
    clock_keys = _canonicalize(
        endpoint=endpoint,
        problem=clock_problem["problem"],
        responses=clock_responses,
        label="routine_equivalence_merged",
        records=records,
    )

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
            raise RuntimeError(
                "constructed different-route proof failed the MATH validator"
            )
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
            clock_keys[0] is not None
            and clock_keys[0] == clock_keys[1]
        ),
        "different_proofs_separated": (
            distinct_keys[0] is not None
            and distinct_keys[1] is not None
            and distinct_keys[0] != distinct_keys[1]
        ),
    }
    report = {
        "schema": "e47m_semantic_regressions_v1",
        "checks": checks,
        "pass": all(checks.values()),
        "outcome_keys": {
            "invalid_derivations_rejected": gcd_keys,
            "routine_equivalence_merged": clock_keys,
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
