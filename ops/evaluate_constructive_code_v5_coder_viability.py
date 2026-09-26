#!/usr/bin/env python3
"""Run the frozen development-only ConstructiveCode v5 Coder viability gate."""

from __future__ import annotations

from typing import Any, Mapping

import evaluate_constructive_code_v4_coder_viability as v4_eval
import replay_constructive_code_v5 as replay_v5


EXPECTED_REPLAY_COUNT = 2_304
_V4_VALIDATE = v4_eval._validate_v4_gate_and_choose_suites


def _validate_v5_gate_and_choose_suites(
    gate: Mapping[str, Any], problem_keys: Mapping[str, str]
) -> dict[str, str]:
    if (
        gate.get("status") != "pass"
        or gate.get("expected_replay_count") != EXPECTED_REPLAY_COUNT
        or gate.get("observed_replay_count") != EXPECTED_REPLAY_COUNT
        or gate.get("violations") not in ([], None)
        or gate.get("checker_equivalence_violations") not in ([], None)
    ):
        raise ValueError("ConstructiveCode v5 executable gate is not an exact pass")
    compatibility_gate = dict(gate)
    compatibility_gate["expected_replay_count"] = 3_072
    compatibility_gate["observed_replay_count"] = 3_072
    return _V4_VALIDATE(compatibility_gate, problem_keys)


def _replay_modules():
    materialize = replay_v5.materialize_v5
    materialize.V3_TASKS = materialize.V5_TASKS
    materialize.V3_SPLIT_ASSIGNMENT = materialize.V5_SPLIT_ASSIGNMENT
    return replay_v5, replay_v5.base, materialize


def main() -> None:
    base_eval = v4_eval.base_eval
    base_eval.RECEIPT_SCHEMA = "constructive-code-v5-coder-05b-viability-v1"
    base_eval._validate_gate_and_choose_suites = _validate_v5_gate_and_choose_suites
    base_eval._replay_modules = _replay_modules
    base_eval.main()


if __name__ == "__main__":
    main()
