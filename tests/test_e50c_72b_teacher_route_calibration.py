from __future__ import annotations

import importlib.util
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "ops/math_strategy_calibration/"
    "run_e50c_72b_teacher_route_calibration.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("e50c_teacher_routes", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e50c_frozen_cohort_and_sampling_contract():
    module = _load()
    problems = module._load_jsonl(module.E47_PROBLEMS)
    assert len(problems) == 50
    assert all(int(row["level"]) == 5 for row in problems)
    assert module.TEACHER_SAMPLES == 16
    assert module.FORCED_SAMPLES == 16
    assert module.UNFORCED_SAMPLES == 64


def test_e50c_protocol_requires_natural_two_route_support():
    module = _load()
    protocol = (
        ROOT
        / "paper/preregistration/"
        "e50c_72b_teacher_route_calibration_20260726.md"
    ).read_text(encoding="utf-8")
    assert module.E49AC_RESULT.name == "result.json"
    assert "E49AA, E49AB, E50A, and E49AC" in protocol
    assert "problem but not the reference" in protocol
    assert "at least two counted unforced responses" in protocol
    assert "without a forced exploration phase" in protocol
    assert "performs no policy update" in protocol
