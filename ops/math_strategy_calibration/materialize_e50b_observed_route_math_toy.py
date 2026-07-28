#!/usr/bin/env python3
"""Materialize E50B from the first passing preregistered route source."""

from __future__ import annotations

import importlib.util
import json
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
    "e50b_observed_route_math_toy_20260726.md"
)
SOURCES = (
    (
        ROOT
        / "var/artifacts/"
        "e49aa_calibrated_pairwise_observed_route_discovery_v1/result.json",
        "e49aa_calibrated_pairwise_observed_route_discovery_v1",
        "e49aa_calibrated_pairwise_observed",
    ),
    (
        ROOT
        / "var/artifacts/"
        "e49ab_all_observed_persistent_pairwise_route_discovery_v1/result.json",
        "e49ab_all_observed_persistent_pairwise_route_discovery_v1",
        "e49ab_all_observed_persistent_pairwise",
    ),
    (
        ROOT
        / "var/artifacts/"
        "e50a_consensus_observed_route_calibration_v1/result.json",
        "e50a_consensus_observed_route_calibration_v1",
        "e50a_consensus_observed",
    ),
    (
        ROOT
        / "var/artifacts/"
        "e49ac_confirmed_singleton_observed_route_discovery_v1/result.json",
        "e49ac_confirmed_singleton_observed_route_discovery_v1",
        "e49ac_confirmed_singleton_observed",
    ),
)


def _load_base():
    spec = importlib.util.spec_from_file_location(
        "e50b_e49x_materializer", BASE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load base materializer: {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _select_source() -> tuple[pathlib.Path, str, str]:
    for path, schema, origin in SOURCES:
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
    raise RuntimeError(
        "no preregistered E50B observed-route source has passed"
    )


def main() -> None:
    calibration, schema, origin = _select_source()
    base = _load_base()
    base.CALIBRATION_RESULT = calibration
    base.CALIBRATION_SCHEMA = schema
    base.OUTPUT_SCHEMA = "e50b_observed_route_math_toy_v1"
    base.OUTPUT_LABEL = "E50B"
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
