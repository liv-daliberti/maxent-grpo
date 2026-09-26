#!/usr/bin/env python3
"""Materialize E49Z from E49Y's empirically executable route pairs."""

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
    "e49z_executable_route_math_toy_20260726.md"
)
CALIBRATION = (
    ROOT
    / "var/artifacts/"
    "e49y_compiled_bottom_up_route_calibration_v1/result.json"
)


def _load_base():
    spec = importlib.util.spec_from_file_location(
        "e49z_e49x_materializer", BASE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load E49X materializer: {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    base = _load_base()
    base.CALIBRATION_RESULT = CALIBRATION
    base.CALIBRATION_SCHEMA = (
        "e49y_compiled_bottom_up_route_calibration_v1"
    )
    base.OUTPUT_SCHEMA = "e49z_executable_route_math_toy_v1"
    base.OUTPUT_LABEL = "E49Z"
    base.CALIBRATION_ORIGIN = "e49y_compiled_bidirectional"
    base.PROTOCOL = PROTOCOL
    base.SCRIPT = SCRIPT
    base.BASE_SCRIPT = BASE_PATH
    base.main()


if __name__ == "__main__":
    main()
