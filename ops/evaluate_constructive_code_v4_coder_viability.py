#!/usr/bin/env python3
"""Run the frozen development-only ConstructiveCode v4 Coder viability gate."""

from __future__ import annotations

from typing import Any, Mapping

import evaluate_constructive_code_v3_coder_viability as base_eval
import replay_constructive_code_v4 as replay_v4


EXPECTED_REPLAY_COUNT = 3_072
_V3_VALIDATE = base_eval._validate_gate_and_choose_suites


def _validate_v4_gate_and_choose_suites(
    gate: Mapping[str, Any], problem_keys: Mapping[str, str]
) -> dict[str, str]:
    if (
        gate.get("status") != "pass"
        or gate.get("expected_replay_count") != EXPECTED_REPLAY_COUNT
        or gate.get("observed_replay_count") != EXPECTED_REPLAY_COUNT
        or gate.get("violations") not in ([], None)
        or gate.get("checker_equivalence_violations") not in ([], None)
    ):
        raise ValueError("ConstructiveCode v4 executable gate is not an exact pass")
    compatibility_gate = dict(gate)
    compatibility_gate["expected_replay_count"] = 4_800
    compatibility_gate["observed_replay_count"] = 4_800
    return _V3_VALIDATE(
        compatibility_gate,
        problem_keys,
    )


def _replay_modules():
    materialize = replay_v4.materialize_v4
    materialize.V3_TASKS = materialize.V4_TASKS
    materialize.V3_SPLIT_ASSIGNMENT = materialize.V4_SPLIT_ASSIGNMENT
    return replay_v4, replay_v4.base, materialize


def main() -> None:
    base_eval.RECEIPT_SCHEMA = "constructive-code-v4-coder-05b-viability-v1"
    base_eval._validate_gate_and_choose_suites = _validate_v4_gate_and_choose_suites
    base_eval._replay_modules = _replay_modules
    base_eval.main()


if __name__ == "__main__":
    main()
