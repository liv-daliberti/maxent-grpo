from __future__ import annotations

import json
from pathlib import Path

from datasets import load_from_disk

from oat_drgrpo.point_maze_grid import (
    POINT_GRID_ACTION_VERSION,
    POINT_GRID_CONTROLLER_SHA256,
    adapt_point_maze_spec,
    find_point_grid_route_programs,
    parse_point_grid_program,
    parse_point_grid_spec,
)
from oat_drgrpo.point_maze_grid_process import PointGridVerifierProcess


ROOT = Path(__file__).resolve().parents[1]


def _first_dev_spec():
    row = load_from_disk("var/data/point_maze_modebench_v1/dev")[
        "multi_answer"
    ][0]
    return adapt_point_maze_spec(json.loads(row["answer"]))


def test_adapter_separates_grid_actions_and_binds_controller():
    raw = _first_dev_spec()
    spec = parse_point_grid_spec(raw)
    assert raw["grid_action_version"] == POINT_GRID_ACTION_VERSION
    assert raw["controller_sha256"] == POINT_GRID_CONTROLLER_SHA256
    assert spec.max_steps_per_action == 100
    assert spec.base_spec.map_id.endswith("_grid_dev_v1")


def test_graph_fixtures_find_two_prompt_visible_routes():
    raw = _first_dev_spec()
    programs = find_point_grid_route_programs(raw)
    assert len(programs) == 2
    for candidate in programs.values():
        tokens, cells = parse_point_grid_program(
            candidate, parse_point_grid_spec(raw)
        )
        assert len(tokens) == len(cells)


def test_external_feedback_controller_executes_two_distinct_routes():
    worker_python = ROOT / "var/maze_runtime/venv/bin/python"
    if not worker_python.is_file():
        return
    raw = _first_dev_spec()
    programs = find_point_grid_route_programs(raw)
    verifier = PointGridVerifierProcess(worker_python=worker_python)
    try:
        validations = [verifier.validate(program, raw) for program in programs.values()]
    finally:
        verifier.close()
    assert all(validation is not None for validation in validations)
    assert len({validation.canonical_key for validation in validations}) == 2
