#!/usr/bin/env python3
"""Run the preregistered ConstructiveCode 1.5B capacity retry."""

from __future__ import annotations

import evaluate_constructive_code_v3_coder_viability as base_eval
import evaluate_constructive_code_v6_coder_viability as v6


def main() -> None:
    base_eval.DEVELOPMENT_PROBLEMS = v6.DEVELOPMENT_PROBLEMS
    base_eval.EVALUATION_PROBLEMS = v6.EVALUATION_PROBLEMS
    base_eval.RECEIPT_SCHEMA = "constructive-code-v8-coder-15b-viability-v1"
    base_eval.EXPECTED_SEED = 77101
    base_eval._validate_gate_and_choose_suites = v6._validate_v6_gate_and_choose_suites
    base_eval._replay_modules = v6._replay_modules
    base_eval.main()


if __name__ == "__main__":
    main()
