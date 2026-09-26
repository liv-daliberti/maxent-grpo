#!/usr/bin/env python3
"""Materialize E50D from the passing safe-signature route calibration."""

from __future__ import annotations

import importlib.util
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE_PATH = (
    ROOT
    / "ops/math_strategy_calibration/"
    "materialize_e49x_accessible_math_toy.py"
)
SCRIPT = pathlib.Path(__file__).resolve()
PROTOCOL = (
    ROOT
    / "paper/preregistration/"
    "e50d_teacher_route_math_toy_20260726.md"
)
CALIBRATIONS = (
    (
        ROOT
        / "var/artifacts/"
        "e50g_safe_signature_teacher_route_calibration_v1/result.json",
        "e50g_safe_signature_teacher_route_calibration_v1",
        "e50g_safe_signature_teacher_05b_natural_supported",
    ),
)


def _load_base():
    spec = importlib.util.spec_from_file_location(
        "e50d_e49x_materializer", BASE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load base materializer: {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _select_calibration():
    import json

    for path, schema, origin in CALIBRATIONS:
        if not path.is_file():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if (
            result.get("schema") == schema
            and result.get("pass") is True
            and len(result.get("selected_source_indices") or []) == 10
            and int(result.get("bidirectionally_executable_count", 0)) >= 10
        ):
            return path, schema, origin
    raise RuntimeError("no E50 teacher-route calibration passed")


def main() -> None:
    calibration, schema, origin = _select_calibration()
    base = _load_base()
    base.CALIBRATION_RESULT = calibration
    base.CALIBRATION_SCHEMA = schema
    base.OUTPUT_SCHEMA = "e50d_teacher_route_math_toy_v1"
    base.OUTPUT_LABEL = "E50D"
    base.CALIBRATION_ORIGIN = origin
    base.SELECTED_IDS_FIELD = "selected_source_indices"
    base.PROBLEM_ID_FIELD = "source_index"
    base.WRITE_TRAIN_ROUTE_PROBE = True
    base.NEUTRAL_ROUTE_CHOICE = True
    base.PROTOCOL = PROTOCOL
    base.SCRIPT = SCRIPT
    base.BASE_SCRIPT = BASE_PATH
    base.main()


if __name__ == "__main__":
    main()
